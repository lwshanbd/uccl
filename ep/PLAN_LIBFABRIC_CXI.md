# UCCL-EP LibFabric/CXI Rewrite Plan

## Goal

Replace the ibverbs RDMA backend in UCCL-EP with a LibFabric backend targeting the HPE Slingshot CXI provider. The MVP deliverable is efficient **point-to-point RMA writes** and **all-to-all communication** over CXI, sufficient to run the EP dispatch/combine kernels on a Slingshot cluster.

---

## 1. System Environment

Confirmed via on-machine probing:

| Item | Value |
|------|-------|
| LibFabric | `/opt/cray/libfabric/2.2.0rc1` (API 2.2, lib in `lib64/`) |
| Provider | `cxi`, protocol `FI_PROTO_CXI` |
| **Endpoint type** | **`FI_EP_RDM` only** (connectionless reliable datagram) |
| cq_data_size | **8 bytes** (ibverbs only has 4-byte immediate) |
| FI_HMEM | **Supported** (GPU memory direct registration) |
| FI_RMA | Supported (`fi_write`, `fi_writedata`, `fi_writemsg`) |
| FI_RMA_EVENT | Supported in rx_attr (remote write triggers CQ completion) |
| FI_ATOMIC | Supported (FI_SUM, FI_MIN, FI_MAX, etc.) |
| FI_REMOTE_CQ_DATA | Supported (carry data with RMA write to remote CQ) |
| mr_mode | `FI_MR_ALLOCATED \| FI_MR_PROV_KEY \| FI_MR_ENDPOINT` |
| iov_limit | 1 (no scatter-gather) |
| inject_size | 192 bytes |
| max_msg_size | 4 GiB |
| tx/rx_ctx_cnt | 1/1 per EP (no scalable EP; multi-rail = multiple EPs) |
| Address format | `FI_ADDR_CXI`, 8 bytes |
| CXI device | `/dev/cxi0`, driver `cxi_ss1`, 200 Gbps |
| GPU | NVIDIA A100 80GB PCIe, sm_80 |
| CXI extensions | `fi_cxi_ext.h`: HRP, PCIe AMO, counter writeback, traffic class |

### Key architectural difference: CXI `FI_EP_RDM` vs ibverbs RC QP

```
ibverbs (current):
  Per-peer QP set: 1 main + 1 ack + 8 data QPs per peer
  Each QP requires INIT -> RTR -> RTS state transitions
  GID query, AH creation for RoCE
  Total QPs: num_peers * 10

CXI/libfabric (target):
  Single FI_EP_RDM endpoint talks to all peers via Address Vector
  fi_enable() in one step, no state machine
  Multi-rail: multiple independent EPs (kChannelPerProxy = 8)
  Total EPs: kChannelPerProxy + 1 (control)
```

This eliminates ~60% of the connection management code.

---

## 2. API Mapping

### 2.1 Data path (performance critical)

| Current ibverbs | CXI/libfabric replacement | Notes |
|---|---|---|
| `ibv_post_send(IBV_WR_RDMA_WRITE_WITH_IMM)` | `fi_writemsg(ep, &msg, FI_REMOTE_CQ_DATA)` | Core path. `msg.data` carries 64-bit CQ data (we use low 32 bits for WriteImm/AtomicsImm) |
| `ibv_wr_start()` + `ibv_wr_rdma_write_imm()` + `ibv_wr_set_sge()` + `ibv_wr_complete()` | Loop of `fi_writemsg()` calls | CXI iov_limit=1, no batched SGE, but can submit consecutively |
| `ibv_post_send(IBV_WR_SEND_WITH_IMM)` | `fi_senddata(ep, buf, len, desc, data, dest, ctx)` | For barrier send-with-imm |
| `ibv_post_send(IBV_WR_ATOMIC_FETCH_AND_ADD)` | `fi_fetch_atomic(ep, &val, 1, desc, result, rdesc, dest, addr, key, FI_UINT64, FI_SUM, ctx)` | Identical semantics |
| `ibv_post_recv()` | `fi_recv(ep, buf, len, desc, src, ctx)` | Pre-post receive buffers |
| `ibv_poll_cq()` / `ibv_start_poll()` | `fi_cq_read(cq, entries, count)` | Returns `fi_cq_data_entry` with `.data` field |
| `ibv_wc_read_imm_data()` | `entries[i].data` | Direct field access |
| `ibv_wc_read_opcode()` | `entries[i].flags & (FI_RMA \| FI_RECV \| FI_REMOTE_CQ_DATA)` | Flag-based, not opcode enum |

### 2.2 Initialization path

| Current ibverbs | CXI/libfabric replacement | Notes |
|---|---|---|
| `ibv_get_device_list()` + `ibv_open_device()` | `fi_getinfo()` + `fi_fabric()` + `fi_domain()` | Single call discovers and opens device |
| `ibv_alloc_pd()` | (not needed) | PD absorbed into `fid_domain` |
| `ibv_reg_mr()` / `ibv_reg_mr_iova2()` | `fi_mr_regattr(domain, &attr, 0, &mr)` with `attr.iface = FI_HMEM_CUDA` | GPU memory direct registration |
| `ibv_reg_dmabuf_mr()` | `fi_mr_regattr()` + `FI_MR_DMABUF` + `attr.dmabuf = {fd, offset, len}` | DMA-BUF path also supported |
| `ibv_create_cq()` | `fi_cq_open(domain, &cq_attr, &cq, NULL)` with `FI_CQ_FORMAT_DATA` | Must use DATA format for `.data` field |
| `ibv_create_qp()` + `ibv_modify_qp(INIT->RTR->RTS)` | `fi_endpoint()` + `fi_ep_bind()` + `fi_enable()` | No state machine |
| `ibv_query_port()` + `ibv_query_gid()` + `ibv_create_ah()` | `fi_getname()` + `fi_av_insert()` | Address resolution greatly simplified |
| `ibv_dereg_mr()` | `fi_close(&mr->fid)` | |
| `ibv_destroy_qp()` | `fi_close(&ep->fid)` | |
| `ibv_destroy_cq()` | `fi_close(&cq->fid)` | |
| `ibv_dealloc_pd()` | (not needed) | |
| `ibv_close_device()` | `fi_close(&domain->fid)` + `fi_close(&fabric->fid)` | |

### 2.3 MR endpoint binding (CXI-specific constraint)

CXI's `FI_MR_ENDPOINT` requires MR to be bound to an EP before the EP is enabled:

```
fi_mr_regattr()  ->  fi_mr_bind(mr, &ep->fid, access)  ->  fi_enable(ep)
                                                             |
                                                    fi_mr_key(mr) now valid
```

For multi-rail (N data EPs), the same GPU buffer needs N separate MR registrations, one bound to each EP.

---

## 3. Data Structures

### 3.1 `FabricCtx` (replaces ibverbs fields in `ProxyCtx`)

```cpp
// ep/include/fabric_ctx.hpp

struct FabricMR {
    fid_mr* mr = nullptr;
    void* desc = nullptr;    // fi_mr_desc(mr)
    uint64_t key = 0;        // fi_mr_key(mr), provider-assigned
};

struct FabricCtx {
    // Global resources (per proxy thread)
    fi_info* info = nullptr;
    fid_fabric* fabric = nullptr;
    fid_domain* domain = nullptr;
    fid_av* av = nullptr;
    fid_cq* tx_cq = nullptr;
    fid_cq* rx_cq = nullptr;

    // Control EP (for barrier, ack messages)
    fid_ep* ctrl_ep = nullptr;
    FabricMR ctrl_gpu_mr;
    FabricMR ctrl_atomic_mr;

    // Data EPs (multi-rail, kChannelPerProxy endpoints)
    std::vector<fid_ep*> data_eps;                 // [kChannelPerProxy]
    std::vector<FabricMR> data_gpu_mrs;            // per-EP GPU MR
    std::vector<FabricMR> data_atomic_mrs;         // per-EP atomic MR

    // Local scratch for fetch-atomic results
    uint64_t* atomic_old_values_buf = nullptr;
    FabricMR atomic_old_values_mr;

    // Per-peer addressing
    std::vector<fi_addr_t> ctrl_peer_addrs;        // [num_ranks]
    std::vector<std::vector<fi_addr_t>> data_peer_addrs;  // [num_ranks][kChannelPerProxy]

    // Per-peer remote MR info (exchanged during connection setup)
    struct RemoteMRInfo {
        uint64_t gpu_key;
        uintptr_t gpu_addr;
        uint64_t gpu_len;
        uint64_t atomic_key;
        uintptr_t atomic_addr;
        uint64_t atomic_len;
        // Per data-EP keys (each remote EP has its own MR key)
        uint64_t data_gpu_keys[kChannelPerProxy];
    };
    std::vector<RemoteMRInfo> remote_info;         // [num_ranks]

    int numa_node = -1;
    int gpu_idx = -1;
};
```

### 3.2 `FabricConnectionInfo` (replaces `RDMAConnectionInfo`)

```cpp
struct FabricConnectionInfo {
    // Control EP address (FI_ADDR_CXI = 8 bytes)
    uint8_t ctrl_ep_addr[64];
    size_t ctrl_ep_addr_len;

    // GPU buffer info
    uint64_t gpu_mr_key;
    uintptr_t gpu_buf_addr;
    uint64_t gpu_buf_len;

    // Atomic buffer info
    uint64_t atomic_mr_key;
    uintptr_t atomic_buf_addr;
    uint64_t atomic_buf_len;

    // Multi-rail data EP addresses and keys
    uint32_t num_data_eps;
    uint8_t data_ep_addrs[kChannelPerProxy][64];
    size_t data_ep_addr_lens[kChannelPerProxy];
    uint64_t data_gpu_keys[kChannelPerProxy];
};
```

---

## 4. File Change Map

### 4.1 New files

| File | ~Lines | Content |
|------|--------|---------|
| `ep/include/fabric_ctx.hpp` | 120 | `FabricCtx`, `FabricConnectionInfo`, `FabricMR` structs |
| `ep/include/fabric.hpp` | 100 | Function declarations for all fabric operations |
| `ep/src/fabric.cpp` | 1800 | All libfabric operations: init, MR reg, write, atomic, poll, destroy |

### 4.2 Modified files

| File | Change scope | What changes |
|------|-------------|--------------|
| `ep/include/proxy_ctx.hpp` | Major | `#ifdef USE_LIBFABRIC`: replace `ibv_*` types with `FabricCtx`; keep token counters, barrier state, wr_id tracking unchanged |
| `ep/include/rdma.hpp` | Minor | `#ifdef USE_LIBFABRIC` to skip ibverbs includes; keep `AtomicsImm`, `WriteImm`, `BarrierImm`, `ImmType`, `PendingUpdate` (transport-independent) |
| `ep/src/proxy.cpp` | Major (~400 lines) | Replace `ibv_*` calls with `fabric_*` calls in `post_gpu_command()`, `local_poll_completions()`, `remote_poll_completions()`, `run_sender()`, `run_remote()`, `run_dual()` |
| `ep/src/uccl_proxy.cpp` | Minor (~50 lines) | Init/destroy path: call `fabric_init`/`fabric_destroy` |
| `ep/src/uccl_ep.cc` | Minor (~30 lines) | Conditional compilation for connection setup |
| `ep/Makefile` | Moderate | Link `-lfabric` instead of `-libverbs -lnl-3 -lnl-route-3`; add `FABRIC_HOME`; swap source files |
| `ep/setup.py` | Moderate | Detect libfabric path; update libraries list |

### 4.3 Unchanged files (no modifications needed)

| File | Reason |
|------|--------|
| `ep/src/internode.cu` | GPU kernels write to D2H queue only, no RDMA calls |
| `ep/src/internode_ll.cu` | Same |
| `ep/src/intranode.cu` | NVLink path, no network |
| `ep/include/ring_buffer.cuh` | `TransferCmd` is GPU->CPU command format, transport-independent |
| `ep/include/d2h_queue_device.cuh` | Device-to-host queue, no network |
| `ep/include/d2h_queue_host.hpp` | Host side of D2H queue |
| `ep/include/uccl_ibgda.cuh` | GPU-side command emission, does not call ibverbs |
| `ep/src/layout.cu` | Buffer layout kernels |
| `ep/src/ep_runtime.cu` | GPU runtime init |
| `ep/include/ep_config.hpp` | Buffer sizing config |
| `ep/include/common.hpp` | Constants (keep `kChannelPerProxy` etc.) |
| `ep/src/fifo.cpp` | FIFO queue implementation |
| `ep/bench/` | Python test scripts (unchanged, they call the Python API) |

---

## 5. Core Implementation Details

### 5.1 Initialization (`fabric_init`)

```cpp
void fabric_init(FabricCtx& ctx, void* gpu_buf, size_t gpu_buf_size,
                 void* atomic_buf, size_t atomic_buf_size,
                 int gpu_idx, int rank, int thread_idx, int local_rank) {
    cudaSetDevice(gpu_idx);
    ctx.gpu_idx = gpu_idx;

    // --- 1. fi_getinfo: discover CXI device ---
    fi_info* hints = fi_allocinfo();
    hints->caps = FI_MSG | FI_RMA | FI_ATOMIC | FI_RMA_EVENT |
                  FI_REMOTE_CQ_DATA | FI_HMEM;
    hints->ep_attr->type = FI_EP_RDM;
    hints->fabric_attr->prov_name = strdup("cxi");
    hints->domain_attr->mr_mode = FI_MR_ALLOCATED | FI_MR_PROV_KEY |
                                   FI_MR_ENDPOINT;
    hints->domain_attr->threading = FI_THREAD_SAFE;

    int ret = fi_getinfo(FI_VERSION(2, 2), nullptr, nullptr, 0,
                         hints, &ctx.info);
    check(ret, "fi_getinfo");

    // --- 2. Open fabric and domain ---
    fi_fabric(ctx.info->fabric_attr, &ctx.fabric, nullptr);
    fi_domain(ctx.fabric, ctx.info, &ctx.domain, nullptr);

    // --- 3. Address Vector ---
    fi_av_attr av_attr = {};
    av_attr.type = FI_AV_MAP;
    av_attr.count = 2048;
    fi_av_open(ctx.domain, &av_attr, &ctx.av, nullptr);

    // --- 4. Completion Queues ---
    fi_cq_attr cq_attr = {};
    cq_attr.format = FI_CQ_FORMAT_DATA;   // MUST use DATA for .data field
    cq_attr.size = 8192;
    cq_attr.wait_obj = FI_WAIT_NONE;      // busy-poll mode
    fi_cq_open(ctx.domain, &cq_attr, &ctx.tx_cq, nullptr);
    fi_cq_open(ctx.domain, &cq_attr, &ctx.rx_cq, nullptr);

    // --- 5. Create data EPs (multi-rail) ---
    ctx.data_eps.resize(kChannelPerProxy);
    ctx.data_gpu_mrs.resize(kChannelPerProxy);
    for (int i = 0; i < kChannelPerProxy; i++) {
        fi_endpoint(ctx.domain, ctx.info, &ctx.data_eps[i], nullptr);
        fi_ep_bind(ctx.data_eps[i], &ctx.tx_cq->fid, FI_TRANSMIT);
        fi_ep_bind(ctx.data_eps[i], &ctx.rx_cq->fid, FI_RECV);
        fi_ep_bind(ctx.data_eps[i], &ctx.av->fid, 0);

        // Register GPU MR and bind to this EP
        ctx.data_gpu_mrs[i] = fabric_reg_gpu_mr(
            ctx.domain, ctx.data_eps[i], gpu_buf, gpu_buf_size, gpu_idx,
            FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ);

        fi_enable(ctx.data_eps[i]);
    }

    // --- 6. Control EP ---
    fi_endpoint(ctx.domain, ctx.info, &ctx.ctrl_ep, nullptr);
    fi_ep_bind(ctx.ctrl_ep, &ctx.tx_cq->fid, FI_TRANSMIT);
    fi_ep_bind(ctx.ctrl_ep, &ctx.rx_cq->fid, FI_RECV);
    fi_ep_bind(ctx.ctrl_ep, &ctx.av->fid, 0);

    ctx.ctrl_gpu_mr = fabric_reg_gpu_mr(
        ctx.domain, ctx.ctrl_ep, gpu_buf, gpu_buf_size, gpu_idx,
        FI_REMOTE_WRITE | FI_REMOTE_READ | FI_WRITE | FI_READ);

    ctx.ctrl_atomic_mr = fabric_reg_host_mr(
        ctx.domain, ctx.ctrl_ep, atomic_buf, atomic_buf_size,
        FI_REMOTE_WRITE | FI_REMOTE_READ | FI_READ | FI_WRITE);

    fi_enable(ctx.ctrl_ep);

    // --- 7. Atomic scratch buffer ---
    size_t scratch_sz = FabricCtx::kMaxAtomicOps * sizeof(uint64_t);
    posix_memalign((void**)&ctx.atomic_old_values_buf, 64, scratch_sz);
    memset(ctx.atomic_old_values_buf, 0, scratch_sz);
    ctx.atomic_old_values_mr = fabric_reg_host_mr(
        ctx.domain, ctx.ctrl_ep, ctx.atomic_old_values_buf, scratch_sz,
        FI_WRITE | FI_READ);

    fi_freeinfo(hints);
}
```

### 5.2 GPU memory registration

```cpp
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
        fprintf(stderr, "[FABRIC] fi_mr_regattr GPU failed: %s\n",
                fi_strerror(-ret));
        exit(1);
    }

    // FI_MR_ENDPOINT: must bind to EP before enable
    ret = fi_mr_bind(result.mr, &ep->fid, access);
    if (ret) {
        fprintf(stderr, "[FABRIC] fi_mr_bind failed: %s\n",
                fi_strerror(-ret));
        exit(1);
    }

    result.desc = fi_mr_desc(result.mr);
    result.key = fi_mr_key(result.mr);
    return result;
}
```

### 5.3 Connection setup (address exchange)

```cpp
void fabric_get_local_info(FabricCtx& ctx, void* gpu_buf, size_t gpu_buf_size,
                            void* atomic_buf, size_t atomic_buf_size,
                            FabricConnectionInfo& info) {
    // Control EP address
    info.ctrl_ep_addr_len = sizeof(info.ctrl_ep_addr);
    fi_getname(&ctx.ctrl_ep->fid, info.ctrl_ep_addr, &info.ctrl_ep_addr_len);

    // GPU buffer info
    info.gpu_mr_key = ctx.ctrl_gpu_mr.key;
    info.gpu_buf_addr = (uintptr_t)gpu_buf;
    info.gpu_buf_len = gpu_buf_size;

    // Atomic buffer info
    info.atomic_mr_key = ctx.ctrl_atomic_mr.key;
    info.atomic_buf_addr = (uintptr_t)atomic_buf;
    info.atomic_buf_len = atomic_buf_size;

    // Data EP addresses and keys
    info.num_data_eps = kChannelPerProxy;
    for (int i = 0; i < kChannelPerProxy; i++) {
        info.data_ep_addr_lens[i] = sizeof(info.data_ep_addrs[i]);
        fi_getname(&ctx.data_eps[i]->fid,
                   info.data_ep_addrs[i], &info.data_ep_addr_lens[i]);
        info.data_gpu_keys[i] = ctx.data_gpu_mrs[i].key;
    }
}

void fabric_insert_peer(FabricCtx& ctx, int peer_rank,
                         const FabricConnectionInfo& remote) {
    // Insert control EP address
    fi_addr_t ctrl_addr;
    fi_av_insert(ctx.av, remote.ctrl_ep_addr, 1, &ctrl_addr, 0, nullptr);
    ctx.ctrl_peer_addrs[peer_rank] = ctrl_addr;

    // Insert data EP addresses
    for (int i = 0; i < kChannelPerProxy; i++) {
        fi_addr_t data_addr;
        fi_av_insert(ctx.av, remote.data_ep_addrs[i], 1, &data_addr, 0, nullptr);
        ctx.data_peer_addrs[peer_rank][i] = data_addr;
    }

    // Record remote MR info
    ctx.remote_info[peer_rank] = {
        .gpu_key = remote.gpu_mr_key,
        .gpu_addr = remote.gpu_buf_addr,
        .gpu_len = remote.gpu_buf_len,
        .atomic_key = remote.atomic_mr_key,
        .atomic_addr = remote.atomic_buf_addr,
        .atomic_len = remote.atomic_buf_len,
    };
    for (int i = 0; i < kChannelPerProxy; i++) {
        ctx.remote_info[peer_rank].data_gpu_keys[i] = remote.data_gpu_keys[i];
    }
}
```

The existing TCP OOB channel (`recv_connection_info_as_server` / `send_connection_info_as_client`) is reused for exchanging `FabricConnectionInfo` instead of `RDMAConnectionInfo`.

### 5.4 RMA write with CQ data (core data path)

This is the hot path that replaces `ibv_post_send(IBV_WR_RDMA_WRITE_WITH_IMM)`:

```cpp
int fabric_write_with_data(FabricCtx& ctx, int channel_idx,
                            void* local_buf, size_t len,
                            int dst_rank, uintptr_t remote_offset,
                            uint64_t cq_data,   // WriteImm or AtomicsImm encoded
                            void* context,       // wr_id as pointer
                            bool signaled) {
    fid_ep* ep = ctx.data_eps[channel_idx];
    fi_addr_t dest = ctx.data_peer_addrs[dst_rank][channel_idx];
    uint64_t remote_addr = ctx.remote_info[dst_rank].gpu_addr + remote_offset;
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
    msg.data = cq_data;

    uint64_t flags = FI_REMOTE_CQ_DATA;
    if (signaled) flags |= FI_COMPLETION;

    ssize_t ret;
    do {
        ret = fi_writemsg(ep, &msg, flags);
        if (ret == -FI_EAGAIN) {
            // TX queue full: drain completions to make room
            fi_cq_data_entry drain[16];
            fi_cq_read(ctx.tx_cq, drain, 16);
        }
    } while (ret == -FI_EAGAIN);

    return (int)ret;
}
```

For the MVP **all-to-all** pattern, the proxy posts one `fi_writemsg` per (destination rank, expert) pair. The `FI_REMOTE_CQ_DATA` flag causes the remote rx_cq to generate a completion with the CQ data containing dispatch metadata (rank, expert_idx, num_tokens).

### 5.5 Batch write (replacing `post_rdma_async_batched`)

The existing code groups commands by destination rank and posts a linked list of work requests. With CXI (iov_limit=1), each command is an independent `fi_writemsg`:

```cpp
void fabric_post_batch(FabricCtx& ctx,
                        std::vector<TransferCmd> const& cmds,
                        std::vector<uint64_t> const& wr_ids,
                        int my_rank, int thread_idx,
                        bool use_normal_mode) {
    // Group by destination rank
    std::unordered_map<int, std::vector<size_t>> by_dst;
    for (size_t i = 0; i < cmds.size(); i++) {
        by_dst[cmds[i].dst_rank].push_back(i);
    }

    for (auto& [dst_rank, indices] : by_dst) {
        for (size_t j = 0; j < indices.size(); j++) {
            size_t idx = indices[j];
            auto& cmd = cmds[idx];

            int channel = select_channel(cmd, thread_idx);
            uintptr_t laddr = decode_write_offset(cmd.req_lptr, use_normal_mode);
            uintptr_t raddr = decode_write_offset(cmd.req_rptr, use_normal_mode);

            // Encode immediate data (same logic as ibverbs path)
            uint32_t imm = encode_imm(cmd, my_rank);

            bool is_last = (j == indices.size() - 1);
            fabric_write_with_data(
                ctx, channel,
                (void*)(gpu_buf_base + laddr), cmd.bytes,
                dst_rank, raddr,
                (uint64_t)imm,
                (void*)wr_ids[idx],
                /*signaled=*/is_last);
        }
    }
}
```

### 5.6 Fetch-and-add atomic

```cpp
int fabric_fetch_add(FabricCtx& ctx, int dst_rank,
                      uintptr_t remote_offset, uint64_t add_value,
                      uint64_t* local_result, void* context) {
    fi_addr_t dest = ctx.ctrl_peer_addrs[dst_rank];
    uint64_t remote_addr = ctx.remote_info[dst_rank].atomic_addr + remote_offset;
    uint64_t rkey = ctx.remote_info[dst_rank].atomic_key;

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
```

### 5.7 Completion polling

```cpp
int fabric_poll_tx(FabricCtx& ctx,
                    std::unordered_set<uint64_t>& acked_wrs) {
    fi_cq_data_entry entries[64];
    ssize_t ret = fi_cq_read(ctx.tx_cq, entries, 64);
    if (ret == -FI_EAGAIN) return 0;
    if (ret < 0) {
        fi_cq_err_entry err = {};
        fi_cq_readerr(ctx.tx_cq, &err, 0);
        fprintf(stderr, "[FABRIC] TX CQ error: %s (prov=%d)\n",
                fi_strerror(err.err), err.prov_errno);
        return -1;
    }
    for (int i = 0; i < ret; i++) {
        uint64_t wr_id = (uint64_t)entries[i].op_context;
        acked_wrs.insert(wr_id);
    }
    return (int)ret;
}

int fabric_poll_rx(FabricCtx& ctx, CopyRingBuffer& ring,
                    void* atomic_buffer_ptr, int num_ranks,
                    int num_experts,
                    std::set<PendingUpdate>& pending_atomic_updates,
                    int my_rank) {
    fi_cq_data_entry entries[64];
    ssize_t ret = fi_cq_read(ctx.rx_cq, entries, 64);
    if (ret == -FI_EAGAIN) return 0;
    if (ret < 0) {
        fi_cq_err_entry err = {};
        fi_cq_readerr(ctx.rx_cq, &err, 0);
        fprintf(stderr, "[FABRIC] RX CQ error: %s (prov=%d)\n",
                fi_strerror(err.err), err.prov_errno);
        return -1;
    }
    for (int i = 0; i < ret; i++) {
        uint64_t flags = entries[i].flags;
        uint32_t imm = (uint32_t)entries[i].data;

        if (flags & FI_REMOTE_CQ_DATA) {
            if (ImmType::IsAtomics(imm)) {
                AtomicsImm a(imm);
                // Process atomic update (same as ibverbs path)
                // ...
            } else if (ImmType::IsBarrier(imm)) {
                BarrierImm b(imm);
                // Process barrier (same as ibverbs path)
                // ...
            } else {
                WriteImm w(imm);
                // Process dispatch/combine data arrival
                // ...
            }
        }
    }
    return (int)ret;
}
```

### 5.8 Barrier (send with data)

```cpp
void fabric_send_barrier(FabricCtx& ctx, int dst_rank,
                          uint32_t barrier_imm, void* context) {
    fi_senddata(ctx.ctrl_ep, nullptr, 0, nullptr,
                (uint64_t)barrier_imm,
                ctx.ctrl_peer_addrs[dst_rank],
                context);
}

void fabric_post_recv(FabricCtx& ctx) {
    fi_recv(ctx.ctrl_ep, nullptr, 0, nullptr, FI_ADDR_UNSPEC, nullptr);
}
```

### 5.9 Cleanup

```cpp
void fabric_destroy(FabricCtx& ctx) {
    // Close EPs
    for (auto* ep : ctx.data_eps) fi_close(&ep->fid);
    if (ctx.ctrl_ep) fi_close(&ctx.ctrl_ep->fid);

    // Close MRs
    auto close_mr = [](FabricMR& m) { if (m.mr) fi_close(&m.mr->fid); };
    for (auto& m : ctx.data_gpu_mrs) close_mr(m);
    close_mr(ctx.ctrl_gpu_mr);
    close_mr(ctx.ctrl_atomic_mr);
    close_mr(ctx.atomic_old_values_mr);

    // Close CQs, AV, domain, fabric
    if (ctx.tx_cq) fi_close(&ctx.tx_cq->fid);
    if (ctx.rx_cq) fi_close(&ctx.rx_cq->fid);
    if (ctx.av) fi_close(&ctx.av->fid);
    if (ctx.domain) fi_close(&ctx.domain->fid);
    if (ctx.fabric) fi_close(&ctx.fabric->fid);
    if (ctx.info) fi_freeinfo(ctx.info);

    free(ctx.atomic_old_values_buf);
}
```

---

## 6. Build System Changes

### 6.1 Makefile

```makefile
# New variables
FABRIC_HOME ?= /opt/cray/libfabric/2.2.0rc1
FABRIC_CFLAGS := -I$(FABRIC_HOME)/include
FABRIC_LDFLAGS := -L$(FABRIC_HOME)/lib64 -lfabric -Wl,-rpath,$(FABRIC_HOME)/lib64

# Replace:
#   LDFLAGS := ... -libverbs -lnl-3 -lnl-route-3 -lnuma
# With:
#   LDFLAGS := -lpthread $(FABRIC_LDFLAGS) -lnuma

# Replace:
#   SRCS += src/rdma.cpp
# With:
#   SRCS += src/fabric.cpp

# Add to CXXFLAGS:
#   -DUSE_LIBFABRIC $(FABRIC_CFLAGS)

# Remove:
#   EFA_HOME, -lefa, all EFA-specific flags
#   -lnl-3 -lnl-route-3 (RoCE address resolution deps)
```

### 6.2 setup.py

```python
fabric_home = os.getenv("FABRIC_HOME", "/opt/cray/libfabric/2.2.0rc1")
has_fabric = os.path.exists(os.path.join(fabric_home, "include", "rdma", "fabric.h"))

if has_fabric:
    include_dirs.append(os.path.join(fabric_home, "include"))
    library_dirs.append(os.path.join(fabric_home, "lib64"))
    libraries = ["fabric", "numa"]
    define_macros.append(("USE_LIBFABRIC", "1"))
```

---

## 7. proxy.cpp Modification Map

Specific functions that need changes and what changes:

| Function | Current ibverbs calls | Replacement |
|----------|----------------------|-------------|
| `Proxy::init_sender()` | `per_thread_rdma_init()`, `create_per_thread_qp()`, `create_per_thread_cq()`, `modify_qp_to_init/rtr/rts()` | `fabric_init()`, `fabric_get_local_info()` |
| `Proxy::init_remote()` | Same + `post_receive_buffer_for_imm()` | `fabric_init()` + `fabric_post_recv()` |
| `Proxy::init_common()` | Connection exchange via TCP, QP setup | `fabric_get_local_info()` → TCP exchange → `fabric_insert_peer()` |
| `Proxy::post_gpu_command()` | `post_rdma_async_batched()` → `ibv_post_send()` | `fabric_post_batch()` → `fi_writemsg()` |
| `Proxy::local_poll_completions()` | `poll_cq_once()` → `ibv_poll_cq()` | `fabric_poll_tx()` → `fi_cq_read()` |
| `Proxy::remote_poll_completions()` | `ibv_start_poll()` etc. | `fabric_poll_rx()` → `fi_cq_read()` |
| `Proxy::notify_gpu_completion()` | Unchanged (uses acked_wrs set, touches D2H ring buffer only) | Unchanged |
| `post_atomic_operations()` | `ibv_post_send(ATOMIC_FETCH_AND_ADD)` | `fabric_fetch_add()` |
| barrier logic | `ibv_post_send(SEND_WITH_IMM)` + `ibv_post_recv()` | `fabric_send_barrier()` + `fabric_post_recv()` |
| `Proxy::destroy()` | `ibv_dereg_mr()`, `ibv_destroy_qp()`, etc. | `fabric_destroy()` |

---

## 8. MVP Definition

The MVP must demonstrate:

### 8.1 Point-to-point RMA write with CQ data

- Rank A writes to Rank B's GPU buffer via `fi_writemsg` + `FI_REMOTE_CQ_DATA`
- Rank B receives completion with metadata in `fi_cq_data_entry.data`
- GPU memory directly registered via `FI_HMEM_CUDA`
- Verified correct data transfer with `cudaMemcpy` + comparison

### 8.2 All-to-all communication (dispatch pattern)

- N ranks, each dispatching tokens to destination experts across all ranks
- Each rank posts `fi_writemsg` to every other rank (multi-destination)
- Remote completions carry `WriteImm` metadata (rank, expert_idx, num_tokens)
- Counter tracking via atomic updates or CQ data counting
- Verified with `bench/test_internode.py` dispatch-only test

### 8.3 Performance targets for MVP

| Metric | Target | How to verify |
|--------|--------|---------------|
| Single-write latency | < 5 us (same as ibverbs baseline) | `fi_pingpong` style benchmark |
| Write bandwidth | > 20 GB/s per EP (200 Gbps link) | Sustained writes with large buffers |
| All-to-all throughput | Within 20% of ibverbs baseline | `bench/test_internode.py` |
| GPU memory registration | Works without staging buffer | Direct `FI_HMEM_CUDA` registration |

---

## 9. Implementation Phases

### Phase A: Scaffolding (3 days)

**Files:** `fabric_ctx.hpp`, `fabric.hpp`, `fabric.cpp` (skeleton), Makefile changes

1. Create `ep/include/fabric_ctx.hpp` with `FabricCtx`, `FabricConnectionInfo`, `FabricMR` structs
2. Create `ep/include/fabric.hpp` with function declarations
3. Create `ep/src/fabric.cpp` with `fabric_init()` and `fabric_destroy()` only
4. Modify `ep/Makefile`: add `FABRIC_HOME`, link `-lfabric`, add source file
5. Add `#ifdef USE_LIBFABRIC` guards to `rdma.hpp` and `proxy_ctx.hpp`
6. **Verify:** compiles and links; `fabric_init` succeeds at runtime (fi_getinfo + fi_fabric + fi_domain)

### Phase B: Memory registration + address exchange (3 days)

**Files:** `fabric.cpp` (MR functions), connection exchange

1. Implement `fabric_reg_gpu_mr()` with `fi_mr_regattr` + `FI_HMEM_CUDA` + `fi_mr_bind`
2. Implement `fabric_reg_host_mr()` for atomic buffers
3. Implement `fabric_get_local_info()` and `fabric_insert_peer()`
4. Wire up TCP OOB exchange to use `FabricConnectionInfo`
5. **Verify:** two ranks exchange addresses and complete AV insert; MR keys are valid

### Phase C: Point-to-point write + completion (4 days)

**Files:** `fabric.cpp` (write + poll), simple test

1. Implement `fabric_write_with_data()` using `fi_writemsg` + `FI_REMOTE_CQ_DATA`
2. Implement `fabric_poll_tx()` and `fabric_poll_rx()`
3. Implement `-FI_EAGAIN` retry logic
4. Write a minimal 2-rank P2P test: write GPU buffer A -> B, verify data + CQ data
5. **Verify:** data correctness; completion with correct immediate data

### Phase D: Integrate with proxy (5 days)

**Files:** `proxy_ctx.hpp`, `proxy.cpp`, `uccl_proxy.cpp`

1. Add `#ifdef USE_LIBFABRIC` to `proxy_ctx.hpp`: replace ibv types with FabricCtx
2. Modify `Proxy::init_sender/init_remote/init_common`: call fabric init + connect
3. Modify `Proxy::post_gpu_command()`: call `fabric_post_batch()`
4. Modify `Proxy::local_poll_completions()`: call `fabric_poll_tx()`
5. Modify `Proxy::remote_poll_completions()`: call `fabric_poll_rx()`
6. Modify completion processing to use `fi_cq_data_entry.data` instead of `ibv_wc`
7. **Verify:** `bench/test_internode.py` dispatch-only mode passes

### Phase E: Atomics + barrier + combine (4 days)

**Files:** `fabric.cpp` (atomic + barrier), `proxy.cpp` (barrier logic)

1. Implement `fabric_fetch_add()` using `fi_fetch_atomic`
2. Implement `fabric_send_barrier()` and `fabric_post_recv()`
3. Wire up `post_atomic_operations()` in proxy
4. Wire up barrier logic in proxy
5. **Verify:** `bench/test_internode.py` full test (dispatch + combine + barrier)

### Phase F: All-to-all performance + tuning (3 days)

1. Run `bench/test_internode.py` with all-to-all pattern (multiple ranks)
2. Profile with CXI counters / `fi_mon_sampler`
3. Tune CQ sizes, number of inflight operations
4. Try CXI-specific optimizations:
   - `FI_CXI_HRP` for high rate puts (small messages)
   - Traffic class via `FI_OPT_CXI_SET_TCLASS`
5. **Verify:** performance within 20% of ibverbs baseline (or better due to 200G Slingshot)

### Timeline summary

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| A: Scaffolding | 3 days | Compiles and inits |
| B: MR + address exchange | 3 days | Two ranks connected |
| C: P2P write + completion | 4 days | Verified data transfer |
| D: Proxy integration | 5 days | dispatch-only test passes |
| E: Atomics + barrier | 4 days | Full internode test passes |
| F: All-to-all tuning | 3 days | Performance-validated MVP |
| **Total** | **~22 working days** | |

---

## 10. CXI-Specific Optimizations (post-MVP)

| Optimization | API | Effect |
|---|---|---|
| **High Rate Puts** | `fi_writemsg()` + `FI_CXI_HRP` flag | Higher message rate for small writes |
| **Inject** | `fi_inject_writedata()` for messages <= 192B | No local MR needed, lower latency |
| **PCIe Atomic** | `fi_atomicmsg()` + `FI_CXI_PCIE_AMO` | Atomic ops via PCIe instead of NIC |
| **Counter writeback** | `fi_cxi_cntr_ops::set_wb_buffer()` | Counter values written to host/GPU memory, skip CQ poll |
| **64-bit CQ data** | `cq_data_size = 8` | Expand immediate data beyond 32 bits; pack more metadata per write |
| **Hardware collectives** | `FI_COLLECTIVE` + `fi_join_collective()` | Hardware-accelerated AllReduce/Barrier |
| **Traffic class** | `FI_OPT_CXI_SET_TCLASS` | QoS control, prioritize latency-sensitive traffic |
| **Triggered operations** | `FI_TRIGGER` | Chain dependent operations in hardware |

---

## 11. Risks and Mitigations

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| `FI_MR_ENDPOINT` forces N MR registrations for N EPs on same buffer | Certain | Extra init-time overhead | CXI mr_cnt=100, kChannelPerProxy=8+1 ctrl = 9 MRs, well within limit |
| `fi_writemsg` returns `-FI_EAGAIN` frequently under load | Medium | Stalls in hot path | Retry loop with CQ drain; tune CQ size and max inflight |
| CXI iov_limit=1 prevents scatter-gather | Certain | Cannot batch multiple SGEs in one WR | Current code already uses 1 SGE per WR; no impact |
| HMEM CUDA registration fails on this node | Low | Must fall back to staging | Test in Phase B; `FI_MR_DMABUF` as backup path |
| `fi_recv` pre-post depth insufficient for barrier | Low | Missed barrier messages | Pre-post sufficient recv buffers (same pattern as ibverbs) |
| VNI/Service ID configuration issues in multi-node | Low | Connection failures | Use `SLINGSHOT_SVC_ID` env var or default CXI auth key |
| RMA_EVENT completions not generated for `fi_writedata` | Very low | Remote side doesn't see writes | fi_info confirms `FI_RMA_EVENT` in rx_attr caps |

---

## 12. Testing Plan

### Unit tests (Phase C)

- 2-rank GPU-to-GPU write with CQ data verification
- Host-to-host atomic fetch-add correctness
- MR registration with FI_HMEM_CUDA on A100

### Integration tests (Phase D-E)

- `bench/test_internode.py` dispatch-only (2 nodes, 8 GPUs each)
- `bench/test_internode.py` full (dispatch + combine)
- `bench/test_low_latency.py` (if time permits)

### Performance tests (Phase F)

- Single-write latency: `fi_pingpong` style
- Sustained bandwidth: large buffer streaming
- All-to-all throughput: `bench/test_internode.py` with varying token counts
- Comparison with ibverbs baseline numbers (if available on similar hardware)

### Correctness invariants

- Every `fi_writedata` with `FI_REMOTE_CQ_DATA` must generate exactly one remote CQ entry with matching data
- `fi_fetch_atomic(FI_SUM)` must return the old value before addition
- Token counters must match after dispatch-combine round-trip
- GPU buffer contents must be bitwise identical after transfer
