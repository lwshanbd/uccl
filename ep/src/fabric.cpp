#include "fabric.hpp"

#include <cuda_runtime.h>
#include <cassert>
#include <cerrno>
#include <cstdio>
#include <cstdlib>
#include <cstring>

// ---------------------------------------------------------------------------
// Internal helpers
// ---------------------------------------------------------------------------

static void check_fi(int ret, char const* msg) {
  if (ret) {
    fprintf(stderr, "[FABRIC] %s: %s (ret=%d)\n", msg, fi_strerror(-ret), ret);
    fflush(stderr);
    std::abort();
  }
}

// ---------------------------------------------------------------------------
// Memory registration
// ---------------------------------------------------------------------------

FabricMR fabric_reg_gpu_mr(fid_domain* domain, fid_ep* ep,
                           void* gpu_buf, size_t bytes, int gpu_idx,
                           uint64_t access) {
  FabricMR result;
  fi_mr_attr attr = {};
  struct iovec iov = {gpu_buf, bytes};

  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access = access;
  attr.iface = FI_HMEM_CUDA;
  attr.device.cuda = gpu_idx;

  int ret = fi_mr_regattr(domain, &attr, 0, &result.mr);
  if (ret) {
    // GPU MR registration failed; fall back to host-path registration.
    // This can happen on some nodes where CXI HMEM is not fully functional.
    fprintf(stderr,
            "[FABRIC] fi_mr_regattr (GPU) failed: %s (ret=%d), "
            "falling back to non-HMEM registration\n",
            fi_strerror(-ret), ret);
    fflush(stderr);
    attr.iface = FI_HMEM_SYSTEM;
    ret = fi_mr_regattr(domain, &attr, 0, &result.mr);
    check_fi(ret, "fi_mr_regattr (GPU fallback to SYSTEM)");
  }

  // CXI requires EP to be enabled BEFORE fi_mr_bind.
  // Caller must ensure fi_enable(ep) was already called.
  ret = fi_mr_bind(result.mr, &ep->fid, 0);
  check_fi(ret, "fi_mr_bind (GPU)");

  ret = fi_mr_enable(result.mr);
  check_fi(ret, "fi_mr_enable (GPU)");

  result.desc = fi_mr_desc(result.mr);
  result.key = fi_mr_key(result.mr);

  return result;
}

FabricMR fabric_reg_host_mr(fid_domain* domain, fid_ep* ep,
                            void* buf, size_t bytes, uint64_t access) {
  FabricMR result;
  fi_mr_attr attr = {};
  struct iovec iov = {buf, bytes};

  attr.mr_iov = &iov;
  attr.iov_count = 1;
  attr.access = access;
  attr.iface = FI_HMEM_SYSTEM;

  int ret = fi_mr_regattr(domain, &attr, 0, &result.mr);
  check_fi(ret, "fi_mr_regattr (host)");

  ret = fi_mr_bind(result.mr, &ep->fid, 0);
  check_fi(ret, "fi_mr_bind (host)");

  ret = fi_mr_enable(result.mr);
  check_fi(ret, "fi_mr_enable (host)");

  result.desc = fi_mr_desc(result.mr);
  result.key = fi_mr_key(result.mr);

  (void)buf; // suppress unused warning in release
  return result;
}

// ---------------------------------------------------------------------------
// Initialization
// ---------------------------------------------------------------------------

void fabric_init(FabricCtx& ctx, void* gpu_buf, size_t gpu_buf_size,
                 void* atomic_buf, size_t atomic_buf_size,
                 int gpu_idx, int rank, int thread_idx, int local_rank) {
  int ret;
  ctx.gpu_idx = gpu_idx;
  ctx.gpu_buf_base = (uintptr_t)gpu_buf;

  // Ensure CUDA device is set on this thread (proxy threads start with
  // no CUDA context, which causes fi_mr_regattr with FI_HMEM_CUDA to fail).
  {
    cudaError_t ce = cudaSetDevice(gpu_idx);
    if (ce != cudaSuccess) {
      fprintf(stderr, "[FABRIC] cudaSetDevice(%d) failed: %s\n",
              gpu_idx, cudaGetErrorString(ce));
      fflush(stderr);
    }
    int cur_dev = -1;
    cudaGetDevice(&cur_dev);
    fprintf(stderr, "[FABRIC] rank=%d thread=%d: CUDA device=%d (requested %d)\n",
            rank, thread_idx, cur_dev, gpu_idx);
    fflush(stderr);
  }

  // ---- 1. fi_getinfo: discover CXI provider ----
  fi_info* hints = fi_allocinfo();
  assert(hints);

  // FI_REMOTE_CQ_DATA is a per-operation flag, not a capability; do not
  // include it in caps.  CQ data support is indicated by cq_data_size > 0
  // in the domain attributes (CXI returns 8).
  // FI_HMEM is only requested when we actually have GPU memory to register.
  hints->caps = FI_MSG | FI_RMA | FI_ATOMIC | FI_RMA_EVENT | FI_HMEM;
  hints->ep_attr->type = FI_EP_RDM;
  hints->fabric_attr->prov_name = strdup("cxi");
  hints->domain_attr->mr_mode =
      FI_MR_ALLOCATED | FI_MR_PROV_KEY | FI_MR_ENDPOINT;
  hints->domain_attr->threading = FI_THREAD_SAFE;

  ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                   nullptr, nullptr, 0, hints, &ctx.info);
  if (ret == -FI_ENODATA) {
    // FI_HMEM may not be available (e.g. login node without GPU).
    // Retry without it.
    fprintf(stderr, "[FABRIC] fi_getinfo with FI_HMEM failed, retrying "
            "without HMEM\n");
    hints->caps &= ~FI_HMEM;
    ret = fi_getinfo(FI_VERSION(FI_MAJOR_VERSION, FI_MINOR_VERSION),
                     nullptr, nullptr, 0, hints, &ctx.info);
  }
  check_fi(ret, "fi_getinfo");

  // Select the CXI NIC closest to our GPU by matching NUMA nodes.
  // fi_getinfo returns a linked list; pick the one on the same NUMA node.
  {
    // Get GPU's NUMA node.
    int gpu_numa = -1;
    {
      cudaDeviceProp prop;
      if (cudaGetDeviceProperties(&prop, gpu_idx) == cudaSuccess) {
        char path[256];
        snprintf(path, sizeof(path),
                 "/sys/bus/pci/devices/0000:%02x:%02x.0/numa_node",
                 prop.pciBusID, prop.pciDeviceID);
        FILE* f = fopen(path, "r");
        if (f) { fscanf(f, "%d", &gpu_numa); fclose(f); }
      }
    }

    // Walk fi_info list and pick NIC with smallest NUMA distance to GPU.
    fi_info* best = ctx.info;
    int best_dist = 9999;
    for (fi_info* p = ctx.info; p; p = p->next) {
      char path[256];
      snprintf(path, sizeof(path), "/sys/class/cxi/%s/device/numa_node",
               p->domain_attr->name);
      int nic_numa = -1;
      FILE* f = fopen(path, "r");
      if (f) { fscanf(f, "%d", &nic_numa); fclose(f); }
      int dist = (gpu_numa >= 0 && nic_numa >= 0)
                     ? abs(gpu_numa - nic_numa) : 9999;
      if (dist < best_dist) {
        best_dist = dist;
        best = p;
      }
    }

    // If we found a better match, use it (detach from list).
    if (best != ctx.info) {
      // We need to use 'best' but fi_fabric/fi_domain take a single fi_info.
      // Swap the first entry's attributes with the best match.
      // Simplest: just point ctx.info at the best one.
      ctx.info = best;
    }

    // Read NUMA node of the selected NIC.
    char npath[256];
    snprintf(npath, sizeof(npath), "/sys/class/cxi/%s/device/numa_node",
             ctx.info->domain_attr->name);
    FILE* f = fopen(npath, "r");
    if (f) {
      int n = -1;
      if (fscanf(f, "%d", &n) == 1) ctx.numa_node = n;
      fclose(f);
    }

    fprintf(stderr,
            "[FABRIC] rank=%d thread=%d: selected NIC %s (numa=%d) "
            "for GPU %d (numa=%d)\n",
            rank, thread_idx, ctx.info->domain_attr->name,
            ctx.numa_node, gpu_idx, gpu_numa);
  }

  // ---- 2. Open fabric and domain ----
  ret = fi_fabric(ctx.info->fabric_attr, &ctx.fabric, nullptr);
  check_fi(ret, "fi_fabric");

  ret = fi_domain(ctx.fabric, ctx.info, &ctx.domain, nullptr);
  check_fi(ret, "fi_domain");

  // ---- 3. Address Vector ----
  fi_av_attr av_attr = {};
  memset(&av_attr, 0, sizeof(av_attr));
  av_attr.type = FI_AV_MAP;
  av_attr.count = 2048;

  ret = fi_av_open(ctx.domain, &av_attr, &ctx.av, nullptr);
  check_fi(ret, "fi_av_open");

  // ---- 4. Completion Queues ----
  fi_cq_attr cq_attr = {};
  memset(&cq_attr, 0, sizeof(cq_attr));
  cq_attr.format = FI_CQ_FORMAT_DATA;
  cq_attr.size = 8192;
  cq_attr.wait_obj = FI_WAIT_NONE;

  ret = fi_cq_open(ctx.domain, &cq_attr, &ctx.tx_cq, nullptr);
  check_fi(ret, "fi_cq_open (TX)");

  ret = fi_cq_open(ctx.domain, &cq_attr, &ctx.rx_cq, nullptr);
  check_fi(ret, "fi_cq_open (RX)");

  // ---- 5. Data EPs (multi-rail) ----
  uint64_t mr_access = FI_REMOTE_WRITE | FI_REMOTE_READ |
                        FI_WRITE | FI_READ;
  bool have_hmem = (ctx.info->caps & FI_HMEM) != 0;

  // CXI requires: create EP -> bind CQ/AV -> fi_enable(EP) -> then
  // fi_mr_reg -> fi_mr_bind(MR, EP) -> fi_mr_enable(MR).
  for (int i = 0; i < kChannelPerProxy; i++) {
    ret = fi_endpoint(ctx.domain, ctx.info, &ctx.data_eps[i], nullptr);
    check_fi(ret, "fi_endpoint (data)");

    ret = fi_ep_bind(ctx.data_eps[i], &ctx.tx_cq->fid, FI_TRANSMIT);
    check_fi(ret, "fi_ep_bind TX CQ (data)");

    ret = fi_ep_bind(ctx.data_eps[i], &ctx.rx_cq->fid, FI_RECV);
    check_fi(ret, "fi_ep_bind RX CQ (data)");

    ret = fi_ep_bind(ctx.data_eps[i], &ctx.av->fid, 0);
    check_fi(ret, "fi_ep_bind AV (data)");

    ret = fi_enable(ctx.data_eps[i]);
    check_fi(ret, "fi_enable (data)");

    // Register GPU MR and bind to this EP (EP must be enabled first on CXI).
    if (have_hmem) {
      ctx.data_gpu_mrs[i] = fabric_reg_gpu_mr(
          ctx.domain, ctx.data_eps[i], gpu_buf, gpu_buf_size, gpu_idx,
          mr_access);
    } else {
      ctx.data_gpu_mrs[i] = fabric_reg_host_mr(
          ctx.domain, ctx.data_eps[i], gpu_buf, gpu_buf_size, mr_access);
    }
  }

  fprintf(stderr, "[FABRIC] rank=%d thread=%d: %d data EPs created\n",
          rank, thread_idx, kChannelPerProxy);

  // ---- 6. Control EP ----
  ret = fi_endpoint(ctx.domain, ctx.info, &ctx.ctrl_ep, nullptr);
  check_fi(ret, "fi_endpoint (ctrl)");

  ret = fi_ep_bind(ctx.ctrl_ep, &ctx.tx_cq->fid, FI_TRANSMIT);
  check_fi(ret, "fi_ep_bind TX CQ (ctrl)");

  ret = fi_ep_bind(ctx.ctrl_ep, &ctx.rx_cq->fid, FI_RECV);
  check_fi(ret, "fi_ep_bind RX CQ (ctrl)");

  ret = fi_ep_bind(ctx.ctrl_ep, &ctx.av->fid, 0);
  check_fi(ret, "fi_ep_bind AV (ctrl)");

  ret = fi_enable(ctx.ctrl_ep);
  check_fi(ret, "fi_enable (ctrl)");

  // MR registration after EP enable (CXI requirement).
  if (have_hmem) {
    ctx.ctrl_gpu_mr = fabric_reg_gpu_mr(
        ctx.domain, ctx.ctrl_ep, gpu_buf, gpu_buf_size, gpu_idx, mr_access);
  } else {
    ctx.ctrl_gpu_mr = fabric_reg_host_mr(
        ctx.domain, ctx.ctrl_ep, gpu_buf, gpu_buf_size, mr_access);
  }

  if (atomic_buf && atomic_buf_size > 0) {
    ctx.ctrl_atomic_mr = fabric_reg_host_mr(
        ctx.domain, ctx.ctrl_ep, atomic_buf, atomic_buf_size, mr_access);
  }

  // ---- 7. Atomic scratch buffer ----
  size_t scratch_sz = FabricCtx::kMaxAtomicOps * sizeof(uint64_t);
  ret = posix_memalign((void**)&ctx.atomic_old_values_buf, 64, scratch_sz);
  if (ret) {
    fprintf(stderr, "[FABRIC] posix_memalign failed: %s\n", strerror(ret));
    std::abort();
  }
  memset(ctx.atomic_old_values_buf, 0, scratch_sz);

  // The scratch buffer must also be registered against the ctrl EP so that
  // fi_fetch_atomic can write results into it.  We register it as host memory
  // because it is a regular malloc'd buffer.
  ctx.atomic_old_values_mr = fabric_reg_host_mr(
      ctx.domain, ctx.ctrl_ep, ctx.atomic_old_values_buf, scratch_sz,
      FI_WRITE | FI_READ);
  // Note: ctrl_ep is already enabled, but CXI allows MR registration after
  // enable as long as the MR is bound before first use.  If this fails on
  // some providers, move this block before fi_enable(ctrl_ep) above.

  fi_freeinfo(hints);

  fprintf(stderr,
          "[FABRIC] rank=%d thread=%d: init complete (gpu_idx=%d)\n",
          rank, thread_idx, gpu_idx);
}

// ---------------------------------------------------------------------------
// Cleanup
// ---------------------------------------------------------------------------

static void close_mr(FabricMR& m) {
  if (m.mr) {
    fi_close(&m.mr->fid);
    m.mr = nullptr;
    m.desc = nullptr;
    m.key = 0;
  }
}

void fabric_destroy(FabricCtx& ctx) {
  // Close MRs BEFORE EPs (MRs hold references to EPs in CXI).
  for (int i = 0; i < kChannelPerProxy; i++) {
    close_mr(ctx.data_gpu_mrs[i]);
  }
  close_mr(ctx.ctrl_gpu_mr);
  close_mr(ctx.ctrl_atomic_mr);
  close_mr(ctx.atomic_old_values_mr);

  // Now close EPs.
  for (int i = 0; i < kChannelPerProxy; i++) {
    if (ctx.data_eps[i]) {
      fi_close(&ctx.data_eps[i]->fid);
      ctx.data_eps[i] = nullptr;
    }
  }
  if (ctx.ctrl_ep) {
    fi_close(&ctx.ctrl_ep->fid);
    ctx.ctrl_ep = nullptr;
  }

  // CQs
  if (ctx.tx_cq) { fi_close(&ctx.tx_cq->fid); ctx.tx_cq = nullptr; }
  if (ctx.rx_cq) { fi_close(&ctx.rx_cq->fid); ctx.rx_cq = nullptr; }

  // AV
  if (ctx.av) { fi_close(&ctx.av->fid); ctx.av = nullptr; }

  // Domain and fabric
  if (ctx.domain) { fi_close(&ctx.domain->fid); ctx.domain = nullptr; }
  if (ctx.fabric) { fi_close(&ctx.fabric->fid); ctx.fabric = nullptr; }

  // fi_info
  if (ctx.info) { fi_freeinfo(ctx.info); ctx.info = nullptr; }

  // Scratch buffer
  if (ctx.atomic_old_values_buf) {
    free(ctx.atomic_old_values_buf);
    ctx.atomic_old_values_buf = nullptr;
  }

  fprintf(stderr, "[FABRIC] destroy complete\n");
}

// ---------------------------------------------------------------------------
// Connection / address exchange
// ---------------------------------------------------------------------------

void fabric_get_local_info(FabricCtx& ctx, void* gpu_buf, size_t gpu_buf_size,
                           void* atomic_buf, size_t atomic_buf_size,
                           FabricConnectionInfo& info) {
  memset(&info, 0, sizeof(info));

  // Control EP address
  info.ctrl_ep_addr_len = sizeof(info.ctrl_ep_addr);
  int ret = fi_getname(&ctx.ctrl_ep->fid, info.ctrl_ep_addr,
                       &info.ctrl_ep_addr_len);
  check_fi(ret, "fi_getname (ctrl)");

  // GPU buffer
  info.gpu_mr_key = ctx.ctrl_gpu_mr.key;
  info.gpu_buf_addr = (uintptr_t)gpu_buf;
  info.gpu_buf_len = gpu_buf_size;

  // Atomic buffer
  info.atomic_mr_key = ctx.ctrl_atomic_mr.key;
  info.atomic_buf_addr = (uintptr_t)atomic_buf;
  info.atomic_buf_len = atomic_buf_size;

  // Data EPs
  info.num_data_eps = kChannelPerProxy;
  for (int i = 0; i < kChannelPerProxy; i++) {
    info.data_ep_addr_lens[i] = sizeof(info.data_ep_addrs[i]);
    ret = fi_getname(&ctx.data_eps[i]->fid,
                     info.data_ep_addrs[i], &info.data_ep_addr_lens[i]);
    check_fi(ret, "fi_getname (data EP)");

    info.data_gpu_keys[i] = ctx.data_gpu_mrs[i].key;
  }
}

void fabric_insert_peer(FabricCtx& ctx, int peer_rank,
                        FabricConnectionInfo const& remote) {
  // Ensure vectors are large enough
  if ((int)ctx.ctrl_peer_addrs.size() <= peer_rank) {
    ctx.ctrl_peer_addrs.resize(peer_rank + 1, FI_ADDR_UNSPEC);
  }
  if ((int)ctx.remote_info.size() <= peer_rank) {
    ctx.remote_info.resize(peer_rank + 1);
  }

  // Insert control EP address into AV
  fi_addr_t ctrl_addr;
  int ret = fi_av_insert(ctx.av, remote.ctrl_ep_addr, 1, &ctrl_addr, 0,
                         nullptr);
  if (ret != 1) {
    fprintf(stderr, "[FABRIC] fi_av_insert (ctrl) failed: %d\n", ret);
    std::abort();
  }
  ctx.ctrl_peer_addrs[peer_rank] = ctrl_addr;

  // Insert data EP addresses
  for (int i = 0; i < kChannelPerProxy; i++) {
    fi_addr_t data_addr;
    ret = fi_av_insert(ctx.av, remote.data_ep_addrs[i], 1, &data_addr, 0,
                       nullptr);
    if (ret != 1) {
      fprintf(stderr, "[FABRIC] fi_av_insert (data %d) failed: %d\n", i, ret);
      std::abort();
    }
    ctx.data_peer_addrs[i][peer_rank] = data_addr;
  }

  // Store remote MR info
  FabricRemoteMRInfo& ri = ctx.remote_info[peer_rank];
  ri.gpu_key = remote.gpu_mr_key;
  ri.gpu_addr = remote.gpu_buf_addr;
  ri.gpu_len = remote.gpu_buf_len;
  ri.atomic_key = remote.atomic_mr_key;
  ri.atomic_addr = remote.atomic_buf_addr;
  ri.atomic_len = remote.atomic_buf_len;
  for (int i = 0; i < kChannelPerProxy; i++) {
    ri.data_gpu_keys[i] = remote.data_gpu_keys[i];
  }
}

// ---------------------------------------------------------------------------
// Data-path operations
// ---------------------------------------------------------------------------

int fabric_write_with_data(FabricCtx& ctx, int channel_idx,
                           void* local_buf, size_t len,
                           int dst_rank, uintptr_t remote_offset,
                           uint64_t cq_data, void* context, bool signaled) {
  fid_ep* ep = ctx.data_eps[channel_idx];
  fi_addr_t dest = ctx.data_peer_addrs[channel_idx][dst_rank];
  FabricRemoteMRInfo const& ri = ctx.remote_info[dst_rank];
  // CXI mr_mode does NOT include FI_MR_VIRT_ADDR, so the remote address
  // in fi_write is an OFFSET from the MR base, not an absolute VA.
  uint64_t remote_addr = remote_offset;
  uint64_t rkey = ri.data_gpu_keys[channel_idx];
  void* desc = ctx.data_gpu_mrs[channel_idx].desc;

  (void)cq_data;

  // Use fi_write for maximum throughput. TX completion means the NIC
  // accepted the command (not that data arrived at remote).
  // Callers that need delivery guarantee should use fi_writemsg with
  // FI_DELIVERY_COMPLETE on the last write, or call fabric_fence().
  ssize_t ret;
  do {
    ret = fi_write(ep, local_buf, len, desc,
                   dest, remote_addr, rkey, context);
    if (ret == -FI_EAGAIN) {
      fi_cq_data_entry drain[16];
      fi_cq_read(ctx.tx_cq, drain, 16);
    }
  } while (ret == -FI_EAGAIN);

  return (int)ret;
}

int fabric_write_batch(FabricCtx& ctx, int channel_idx,
                       void* local_buf, size_t len,
                       int dst_rank, uintptr_t remote_offset,
                       void* context, bool is_last) {
  fid_ep* ep = ctx.data_eps[channel_idx];
  fi_addr_t dest = ctx.data_peer_addrs[channel_idx][dst_rank];
  uint64_t remote_addr = remote_offset;
  uint64_t rkey = ctx.remote_info[dst_rank].data_gpu_keys[channel_idx];
  void* desc = ctx.data_gpu_mrs[channel_idx].desc;

  struct iovec iov = {local_buf, len};
  struct fi_rma_iov rma_iov = {remote_addr, len, rkey};
  struct fi_msg_rma msg = {};
  msg.msg_iov = &iov;
  msg.desc = &desc;
  msg.iov_count = 1;
  msg.addr = dest;
  msg.rma_iov = &rma_iov;
  msg.rma_iov_count = 1;
  msg.context = context;

  // FI_MORE tells CXI to batch without ringing the doorbell.
  // On the last write, omit FI_MORE to trigger transmission of
  // the entire batch.
  uint64_t flags = FI_COMPLETION;
  if (!is_last) flags |= FI_MORE;

  ssize_t ret;
  do {
    ret = fi_writemsg(ep, &msg, flags);
    if (ret == -FI_EAGAIN) {
      fi_cq_data_entry drain[16];
      fi_cq_read(ctx.tx_cq, drain, 16);
    }
  } while (ret == -FI_EAGAIN);

  return (int)ret;
}

int fabric_fetch_add(FabricCtx& ctx, int dst_rank,
                     uintptr_t remote_offset, uint64_t add_value,
                     uint64_t* local_result, void* context) {
  fi_addr_t dest = ctx.ctrl_peer_addrs[dst_rank];
  FabricRemoteMRInfo const& ri = ctx.remote_info[dst_rank];
  uint64_t remote_addr = ri.atomic_addr + remote_offset;
  uint64_t rkey = ri.atomic_key;

  ssize_t ret;
  do {
    ret = fi_fetch_atomic(
        ctx.ctrl_ep,
        &add_value, 1, ctx.ctrl_atomic_mr.desc,
        local_result, ctx.atomic_old_values_mr.desc,
        dest, remote_addr, rkey,
        FI_UINT64, FI_SUM, context);
    if (ret == -FI_EAGAIN) {
      fi_cq_data_entry drain[4];
      fi_cq_read(ctx.tx_cq, drain, 4);
    }
  } while (ret == -FI_EAGAIN);

  return (int)ret;
}

int fabric_send_data(FabricCtx& ctx, int dst_rank,
                     uint64_t cq_data, void* context) {
  ssize_t ret;
  do {
    ret = fi_senddata(ctx.ctrl_ep, nullptr, 0, nullptr,
                      cq_data, ctx.ctrl_peer_addrs[dst_rank], context);
    if (ret == -FI_EAGAIN) {
      fi_cq_data_entry drain[4];
      fi_cq_read(ctx.tx_cq, drain, 4);
    }
  } while (ret == -FI_EAGAIN);
  return (int)ret;
}

int fabric_post_recv(FabricCtx& ctx) {
  return (int)fi_recv(ctx.ctrl_ep, nullptr, 0, nullptr,
                      FI_ADDR_UNSPEC, nullptr);
}

int fabric_flush(FabricCtx& ctx, int dst_rank, int channel,
                 void* local_buf, void* local_desc) {
  // Flush all prior writes to |dst_rank| by issuing a small fi_read from
  // the remote buffer.  RMA ordering guarantees that when the read
  // completes, all prior writes to the same destination are visible.
  // This is the same pattern used by aws-ofi-nccl (post_flush_req).
  fid_ep* ep = ctx.data_eps[channel];
  fi_addr_t dest = ctx.data_peer_addrs[channel][dst_rank];
  FabricRemoteMRInfo const& ri = ctx.remote_info[dst_rank];
  uint64_t rkey = ri.data_gpu_keys[channel];

  ssize_t ret;
  do {
    ret = fi_read(ep, local_buf, sizeof(uint64_t), local_desc,
                  dest, 0, rkey, (void*)0xF005ULL);
    if (ret == -FI_EAGAIN) {
      fi_cq_data_entry drain[16];
      fi_cq_read(ctx.tx_cq, drain, 16);
    }
  } while (ret == -FI_EAGAIN);

  return (int)ret;
}

// ---------------------------------------------------------------------------
// Completion polling
// ---------------------------------------------------------------------------

int fabric_poll_tx(FabricCtx& ctx,
                   std::unordered_set<uint64_t>& acked_wrs) {
  fi_cq_data_entry entries[64];
  ssize_t ret = fi_cq_read(ctx.tx_cq, entries, 64);
  if (ret == -FI_EAGAIN) return 0;
  if (ret < 0) {
    fi_cq_err_entry err = {};
    fi_cq_readerr(ctx.tx_cq, &err, 0);
    fprintf(stderr, "[FABRIC] TX CQ error: %s (prov_errno=%d)\n",
            fi_strerror(err.err), err.prov_errno);
    return -1;
  }
  for (ssize_t i = 0; i < ret; i++) {
    uint64_t wr_id = (uint64_t)entries[i].op_context;
    acked_wrs.insert(wr_id);
  }
  return (int)ret;
}

int fabric_poll_rx(FabricCtx& ctx, fi_cq_data_entry* entries,
                   int max_entries) {
  ssize_t ret = fi_cq_read(ctx.rx_cq, entries, max_entries);
  if (ret == -FI_EAGAIN) return 0;
  if (ret < 0) {
    fi_cq_err_entry err = {};
    fi_cq_readerr(ctx.rx_cq, &err, 0);
    fprintf(stderr, "[FABRIC] RX CQ error: %s (prov_errno=%d)\n",
            fi_strerror(err.err), err.prov_errno);
    return -1;
  }
  return (int)ret;
}
