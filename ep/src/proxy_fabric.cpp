// Libfabric/CXI implementations of Proxy methods.
// This file is #included from proxy.cpp when USE_LIBFABRIC is defined.
// It replaces the ibverbs-specific implementations of init_common(),
// init_sender(), init_remote(), run_sender(), run_remote(), run_dual(),
// post_gpu_commands_mixed(), quiet_cq(), post_barrier_msg(), and destroy().

#include "fabric.hpp"

// Helper: decode the local buffer offset from a TransferCmd.
static uintptr_t decode_lptr(TransferCmd const& cmd, bool use_normal_mode) {
  if (use_normal_mode) {
    return static_cast<uintptr_t>(cmd.req_lptr) << 2;
  }
  return static_cast<uintptr_t>(cmd.req_lptr) << 4;
}

// Helper: decode the remote buffer offset from a TransferCmd.
static uintptr_t decode_rptr(TransferCmd const& cmd, bool use_normal_mode) {
  if (use_normal_mode) {
    return static_cast<uintptr_t>(cmd.req_rptr) << 2;
  }
  return static_cast<uintptr_t>(cmd.req_rptr) << 4;
}

// ---------------------------------------------------------------------------
// Proxy::init_common — libfabric path
// ---------------------------------------------------------------------------
void Proxy::init_common() {
  int const my_rank = cfg_.rank;
  int const num_ranks = static_cast<int>(peers_.size());

  // Initialize libfabric resources: fabric, domain, AV, CQs, EPs, MRs.
  fabric_init(fabric_ctx_, cfg_.gpu_buffer, cfg_.total_size,
              atomic_buffer_ptr_, atomic_buffer_ptr_ ? kAtomicBufferSize : 0,
              cfg_.local_rank, my_rank, cfg_.thread_idx, cfg_.local_rank);

  ctx_.numa_node = fabric_ctx_.numa_node;
  pin_thread_to_numa_wrapper();

  // Resize peer address tables.
  fabric_ctx_.ctrl_peer_addrs.resize(num_ranks, FI_ADDR_UNSPEC);
  fabric_ctx_.remote_info.resize(num_ranks);

  // Get local connection info.
  local_fabric_infos_.resize(num_ranks);
  remote_fabric_infos_.resize(num_ranks);

  // Fill one canonical local info (same for all peers since CXI is RDM).
  FabricConnectionInfo canonical_local{};
  fabric_get_local_info(fabric_ctx_, cfg_.gpu_buffer, cfg_.total_size,
                        atomic_buffer_ptr_,
                        atomic_buffer_ptr_ ? kAtomicBufferSize : 0,
                        canonical_local);

  for (int peer = 0; peer < num_ranks; ++peer) {
    local_fabric_infos_[peer] = canonical_local;
  }

  usleep(50 * 1000);

  // OOB exchange: TCP-based, reusing the existing listen socket.
  // Exchange FabricConnectionInfo instead of RDMAConnectionInfo.
  auto send_fabric_info = [&](int peer) {
    int fd = socket(AF_INET, SOCK_STREAM, 0);
    if (fd < 0) { perror("socket"); std::abort(); }
    struct sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_port = htons(peers_[peer].listen_ports[cfg_.thread_idx]);
    inet_pton(AF_INET, peers_[peer].ip.c_str(), &addr.sin_addr);
    int retries = 0;
    while (connect(fd, (struct sockaddr*)&addr, sizeof(addr)) < 0) {
      if (++retries > MAX_RETRIES) {
        fprintf(stderr, "connect to peer %d failed after %d retries\n",
                peer, retries);
        std::abort();
      }
      usleep(RETRY_DELAY_MS * 1000);
    }
    // Send: my_rank (4 bytes) + FabricConnectionInfo
    int32_t my_r = my_rank;
    ssize_t n = write(fd, &my_r, sizeof(my_r));
    (void)n;
    n = write(fd, &local_fabric_infos_[peer], sizeof(FabricConnectionInfo));
    (void)n;
    close(fd);
  };

  auto recv_fabric_info = [&]() {
    struct sockaddr_in cli{};
    socklen_t len = sizeof(cli);
    int fd = accept(listen_fd_, (struct sockaddr*)&cli, &len);
    if (fd < 0) { perror("accept"); std::abort(); }
    int32_t peer_rank = -1;
    ssize_t n = read(fd, &peer_rank, sizeof(peer_rank));
    (void)n;
    FabricConnectionInfo info{};
    n = read(fd, &info, sizeof(info));
    (void)n;
    close(fd);
    if (peer_rank >= 0 && peer_rank < num_ranks) {
      remote_fabric_infos_[peer_rank] = info;
    }
  };

  // Count how many peers need connection exchange.
  int num_remote_peers = 0;
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - my_rank) % MAX_NUM_GPUS != 0)
      continue;
    num_remote_peers++;
  }

  // Receiver thread.
  std::thread receiver_thread([&]() {
    for (int i = 0; i < num_remote_peers; i++) {
      recv_fabric_info();
    }
  });

  // Send to all remote peers.
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - my_rank) % MAX_NUM_GPUS != 0)
      continue;
    send_fabric_info(peer);
  }

  receiver_thread.join();

  // Insert all remote peers into AV and record their MR keys.
  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    if (cfg_.use_normal_mode && std::abs(peer - my_rank) % MAX_NUM_GPUS != 0)
      continue;
    fabric_insert_peer(fabric_ctx_, peer, remote_fabric_infos_[peer]);

    fprintf(stderr,
            "[Proxy-Fabric] rank=%d thread=%d: connected to peer %d "
            "(gpu_key=0x%lx, addr=0x%lx)\n",
            my_rank, cfg_.thread_idx, peer,
            (unsigned long)fabric_ctx_.remote_info[peer].gpu_key,
            (unsigned long)fabric_ctx_.remote_info[peer].gpu_addr);
  }

  usleep(50 * 1000);

  // Normal-mode local barrier setup (transport-independent).
  if (cfg_.use_normal_mode) {
    std::string const my_ip = peers_[cfg_.rank].ip;
    std::vector<int> local_ranks;
    local_ranks.reserve(num_ranks);
    int leader_rank = cfg_.rank;
    for (int r = 0; r < num_ranks; ++r) {
      if (peers_[r].ip == my_ip) {
        local_ranks.push_back(r);
        if (r < leader_rank) leader_rank = r;
      }
    }
    ctx_.num_local_ranks = (int)local_ranks.size();
    ctx_.node_leader_rank = leader_rank;
    ctx_.local_rank = cfg_.local_rank;
    ctx_.thread_idx = cfg_.thread_idx;

#ifndef USE_SUBSET_BARRIER
    std::string const shm_name = shm_name_for_barrier(my_ip, cfg_.thread_idx);
    ctx_.lb = map_local_barrier_shm(shm_name, &ctx_.lb_owner);
    if (!ctx_.lb) {
      fprintf(stderr, "Failed to map local barrier shm: %s\n",
              shm_name.c_str());
      std::abort();
    }
    if (ctx_.lb_owner) {
      ctx_.lb->full_mask = (ctx_.num_local_ranks >= 64)
                               ? ~0ULL
                               : ((1ULL << ctx_.num_local_ranks) - 1ULL);
      for (int i = 0; i < ctx_.num_local_ranks; ++i) {
        ctx_.lb->arrive_seq[i].store(0, std::memory_order_relaxed);
        ctx_.lb->release_seq[i].store(0, std::memory_order_relaxed);
      }
    } else {
      while (ctx_.lb->full_mask == 0ULL) cpu_relax();
    }
#endif
  }

#ifdef USE_MSCCLPP_FIFO_BACKEND
  fifo_seq_.assign(cfg_.d2h_queues.size(), 0);
  fifo_pending_.assign(cfg_.d2h_queues.size(),
                       std::deque<std::pair<uint64_t, size_t>>{});
#endif

  // Pre-post receive buffers for barrier messages.
  for (int i = 0; i < kMaxOutstandingRecvs; i++) {
    fabric_post_recv(fabric_ctx_);
  }

  // Close the listen socket — TCP OOB exchange is done.  Leaving it open
  // can cause spurious connection attempts from other subsystems (MPI, CXI)
  // to hit this socket and trigger retries / aborts.
  if (listen_fd_ >= 0) {
    close(listen_fd_);
    listen_fd_ = -1;
  }

  fprintf(stderr, "[Proxy-Fabric] rank=%d thread=%d: init_common complete\n",
          my_rank, cfg_.thread_idx);
}

// ---------------------------------------------------------------------------
void Proxy::init_sender() {
  init_common();
}

void Proxy::init_remote() {
  init_common();
}

// ---------------------------------------------------------------------------
// MPI-based init: split init_common into phases
// ---------------------------------------------------------------------------

void Proxy::init_fabric_local(FabricConnectionInfo& out_local_info) {
  int const my_rank = cfg_.rank;

  // Initialize libfabric resources (same as first part of init_common).
  fabric_init(fabric_ctx_, cfg_.gpu_buffer, cfg_.total_size,
              atomic_buffer_ptr_, atomic_buffer_ptr_ ? kAtomicBufferSize : 0,
              cfg_.local_rank, my_rank, cfg_.thread_idx, cfg_.local_rank);

  ctx_.numa_node = fabric_ctx_.numa_node;
  pin_thread_to_numa_wrapper();

  // Get this thread's local connection info.
  memset(&out_local_info, 0, sizeof(out_local_info));
  fabric_get_local_info(fabric_ctx_, cfg_.gpu_buffer, cfg_.total_size,
                        atomic_buffer_ptr_,
                        atomic_buffer_ptr_ ? kAtomicBufferSize : 0,
                        out_local_info);

  fprintf(stderr,
          "[Proxy-Fabric] rank=%d thread=%d: init_fabric_local complete\n",
          my_rank, cfg_.thread_idx);
}

void Proxy::init_fabric_insert_peers(
    int num_ranks,
    std::vector<FabricConnectionInfo> const& remote_infos) {
  int const my_rank = cfg_.rank;

  fabric_ctx_.ctrl_peer_addrs.resize(num_ranks, FI_ADDR_UNSPEC);
  fabric_ctx_.remote_info.resize(num_ranks);

  for (int peer = 0; peer < num_ranks; ++peer) {
    if (peer == my_rank) continue;
    if (peers_[peer].ip == peers_[my_rank].ip) continue;
    fabric_insert_peer(fabric_ctx_, peer, remote_infos[peer]);

    fprintf(stderr,
            "[Proxy-Fabric] rank=%d thread=%d: peer %d inserted "
            "(gpu_key=0x%lx)\n",
            my_rank, cfg_.thread_idx, peer,
            (unsigned long)fabric_ctx_.remote_info[peer].gpu_key);
  }
}

void Proxy::run_dual_after_init() {
#ifdef USE_MSCCLPP_FIFO_BACKEND
  // Initialize FIFO state (normally done in init_common).
  fifo_seq_.assign(cfg_.d2h_queues.size(), 0);
  fifo_pending_.assign(cfg_.d2h_queues.size(),
                       std::deque<std::pair<uint64_t, size_t>>{});
#endif

  fprintf(stderr,
          "[Proxy-Fabric] rank=%d thread=%d: entering run_dual_after_init, "
          "posting %d recvs\n",
          cfg_.rank, cfg_.thread_idx, kMaxOutstandingRecvs);
  fflush(stderr);

  // Pre-post receive buffers for barrier/atomic messages.
  for (int i = 0; i < kMaxOutstandingRecvs; i++) {
    int ret = fabric_post_recv(fabric_ctx_);
    if (ret && i < 3) {
      fprintf(stderr, "[Proxy-Fabric] rank=%d thread=%d: fabric_post_recv[%d] "
              "failed: %d\n", cfg_.rank, cfg_.thread_idx, i, ret);
    }
  }

  int num_ranks = static_cast<int>(peers_.size());
  fprintf(stderr,
          "[Proxy-Fabric] rank=%d thread=%d: run_dual_after_init started "
          "(num_ranks=%d, d2h_ready=%d, d2h_queues=%zu, "
          "data_eps[0]=%p, ctrl_ep=%p)\n",
          cfg_.rank, cfg_.thread_idx, num_ranks,
          d2h_ready_.load(std::memory_order_relaxed),
          cfg_.d2h_queues.size(),
          (void*)fabric_ctx_.data_eps[0],
          (void*)fabric_ctx_.ctrl_ep);

  // Enter the polling loop (same as run_dual but without init_common).
  uint64_t my_tail = 0;
  size_t seen = 0;
  fi_cq_data_entry rx_entries[64];
  int loop_count = 0;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    if (loop_count < 3 && cfg_.thread_idx == 0) {
      fprintf(stderr, "[Proxy-Fabric] rank=%d thread=%d: poll loop iter %d\n",
              cfg_.rank, cfg_.thread_idx, loop_count);
      fflush(stderr);
    }
    loop_count++;

    fabric_poll_tx(fabric_ctx_, acked_wrs_);

    int ne = fabric_poll_rx(fabric_ctx_, rx_entries, 64);
    for (int i = 0; i < ne; i++) {
      uint32_t imm = (uint32_t)rx_entries[i].data;
      if (rx_entries[i].flags & FI_REMOTE_CQ_DATA) {
        if (ImmType::IsBarrier(imm)) {
          BarrierImm b(imm);
          if (b.GetIsAck()) {
            ctx_.barrier_released = true;
            ctx_.barrier_release_seq = b.GetSeq();
          } else {
            int src_node = b.GetRank();
            if (ctx_.barrier_arrived.size() <=
                static_cast<size_t>(src_node)) {
              ctx_.barrier_arrived.resize(src_node + 1, 0);
            }
            if (!ctx_.barrier_arrived[src_node]) {
              ctx_.barrier_arrived[src_node] = 1;
              ++ctx_.barrier_arrival_count;
            }
          }
        } else if (ImmType::IsAtomics(imm)) {
          AtomicsImm a(imm);
          if (atomic_buffer_ptr_) {
            int off = a.GetOff();
            int val = a.GetValue();
            auto* p = reinterpret_cast<std::atomic<int64_t>*>(
                static_cast<char*>(atomic_buffer_ptr_) + off * sizeof(int64_t));
            p->fetch_add(val, std::memory_order_relaxed);
          }
        }
        fabric_post_recv(fabric_ctx_);
      }
    }

    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);

    if (cfg_.use_normal_mode) {
      barrier_check();
    }
  }
}

// ---------------------------------------------------------------------------
// Main loops
// ---------------------------------------------------------------------------
void Proxy::run_sender() {
  printf("CPU sender thread %d started (libfabric)\n", cfg_.thread_idx);
  init_sender();
  size_t seen = 0;
  uint64_t my_tail = 0;
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    fabric_poll_tx(fabric_ctx_, acked_wrs_);
    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
  }
}

void Proxy::run_remote() {
  printf("Remote CPU thread %d started (libfabric)\n", cfg_.thread_idx);
  init_remote();
  std::set<PendingUpdate> pending_atomic_updates;
  fi_cq_data_entry rx_entries[64];
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    int ne = fabric_poll_rx(fabric_ctx_, rx_entries, 64);
    for (int i = 0; i < ne; i++) {
      uint32_t imm = (uint32_t)rx_entries[i].data;
      if (rx_entries[i].flags & FI_REMOTE_CQ_DATA) {
        if (ImmType::IsBarrier(imm)) {
          BarrierImm b(imm);
          if (b.GetIsAck()) {
            ctx_.barrier_released = true;
            ctx_.barrier_release_seq = b.GetSeq();
          } else {
            // Arrival from a peer node
            int src_node = b.GetRank();
            if (ctx_.barrier_arrived.size() <=
                static_cast<size_t>(src_node)) {
              ctx_.barrier_arrived.resize(src_node + 1, 0);
            }
            if (!ctx_.barrier_arrived[src_node]) {
              ctx_.barrier_arrived[src_node] = 1;
              ++ctx_.barrier_arrival_count;
            }
          }
        } else if (ImmType::IsAtomics(imm)) {
          AtomicsImm a(imm);
          // Apply atomic update to local buffer
          if (atomic_buffer_ptr_) {
            int off = a.GetOff();
            int val = a.GetValue();
            auto* p = reinterpret_cast<std::atomic<int64_t>*>(
                static_cast<char*>(atomic_buffer_ptr_) + off * sizeof(int64_t));
            p->fetch_add(val, std::memory_order_relaxed);
          }
        } else {
          // WriteImm — data arrived (dispatch/combine notification)
          // The actual data was already written by the remote RMA write.
          // Nothing to do here for the MVP; counter tracking can be added.
        }
        // Re-post a receive buffer.
        fabric_post_recv(fabric_ctx_);
      }
    }
  }
}

void Proxy::run_dual() {
  init_common();
  uint64_t my_tail = 0;
  size_t seen = 0;
  std::set<PendingUpdate> pending_atomic_updates;
  fi_cq_data_entry rx_entries[64];
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    // Poll TX completions
    fabric_poll_tx(fabric_ctx_, acked_wrs_);

    // Poll RX completions
    int ne = fabric_poll_rx(fabric_ctx_, rx_entries, 64);
    for (int i = 0; i < ne; i++) {
      uint32_t imm = (uint32_t)rx_entries[i].data;
      if (rx_entries[i].flags & FI_REMOTE_CQ_DATA) {
        if (ImmType::IsBarrier(imm)) {
          BarrierImm b(imm);
          if (b.GetIsAck()) {
            ctx_.barrier_released = true;
            ctx_.barrier_release_seq = b.GetSeq();
          } else {
            int src_node = b.GetRank();
            if (ctx_.barrier_arrived.size() <=
                static_cast<size_t>(src_node)) {
              ctx_.barrier_arrived.resize(src_node + 1, 0);
            }
            if (!ctx_.barrier_arrived[src_node]) {
              ctx_.barrier_arrived[src_node] = 1;
              ++ctx_.barrier_arrival_count;
            }
          }
        } else if (ImmType::IsAtomics(imm)) {
          AtomicsImm a(imm);
          if (atomic_buffer_ptr_) {
            int off = a.GetOff();
            int val = a.GetValue();
            auto* p = reinterpret_cast<std::atomic<int64_t>*>(
                static_cast<char*>(atomic_buffer_ptr_) + off * sizeof(int64_t));
            p->fetch_add(val, std::memory_order_relaxed);
          }
        }
        fabric_post_recv(fabric_ctx_);
      }
    }

    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);

    if (cfg_.use_normal_mode) {
      barrier_check();
    }
  }
}

// ---------------------------------------------------------------------------
// run_dual_poll_only — resume after pause (skip init_common)
// ---------------------------------------------------------------------------
void Proxy::run_dual_poll_only() {
  uint64_t my_tail = 0;
  size_t seen = 0;
  fi_cq_data_entry rx_entries[64];
  while (ctx_.progress_run.load(std::memory_order_acquire)) {
    fabric_poll_tx(fabric_ctx_, acked_wrs_);

    int ne = fabric_poll_rx(fabric_ctx_, rx_entries, 64);
    for (int i = 0; i < ne; i++) {
      uint32_t imm = (uint32_t)rx_entries[i].data;
      if (rx_entries[i].flags & FI_REMOTE_CQ_DATA) {
        if (ImmType::IsAtomics(imm)) {
          AtomicsImm a(imm);
          if (atomic_buffer_ptr_) {
            int off = a.GetOff();
            int val = a.GetValue();
            auto* p = reinterpret_cast<std::atomic<int64_t>*>(
                static_cast<char*>(atomic_buffer_ptr_) + off * sizeof(int64_t));
            p->fetch_add(val, std::memory_order_relaxed);
          }
        }
        fabric_post_recv(fabric_ctx_);
      }
    }

    notify_gpu_completion(my_tail);
    post_gpu_command(my_tail, seen);
  }
}

// ---------------------------------------------------------------------------
// post_gpu_commands_mixed — libfabric path
// ---------------------------------------------------------------------------
void Proxy::post_gpu_commands_mixed(
    std::vector<uint64_t> const& wrs_to_post,
    std::vector<TransferCmd> const& cmds_to_post) {

  for (size_t i = 0; i < cmds_to_post.size(); ++i) {
    auto const& cmd = cmds_to_post[i];
    uint64_t wr_id = wrs_to_post[i];

    switch (get_base_cmd(cmd.cmd_type)) {
      case CmdType::WRITE: {
        int dst = cmd.dst_rank;
        int channel = dst % kChannelPerProxy;
        uintptr_t loffset = decode_lptr(cmd, cfg_.use_normal_mode);
        uintptr_t roffset = decode_rptr(cmd, cfg_.use_normal_mode);

        // Encode WriteImm metadata in CQ data (same encoding as ibverbs).
        uint32_t imm = WriteImm::Pack(
            get_is_combine(cmd.cmd_type),
            get_low_latency(cmd.cmd_type),
            cmd.expert_idx,
            cmd.bytes > 0 ? 1 : 0,  // num_tokens placeholder
            cfg_.rank).GetImmData();

        bool signaled = true;  // Always signal for now (tune later)
        void* local_buf =
            reinterpret_cast<void*>(fabric_ctx_.gpu_buf_base + loffset);

        fabric_write_with_data(fabric_ctx_, channel, local_buf, cmd.bytes,
                               dst, roffset, (uint64_t)imm,
                               reinterpret_cast<void*>(wr_id), signaled);
        break;
      }

      case CmdType::ATOMIC: {
        int dst = cmd.dst_rank;
        uintptr_t roffset = static_cast<uintptr_t>(cmd.req_rptr);
        uint64_t add_value = static_cast<uint64_t>(cmd.value);

        // Encode AtomicsImm in CQ data.
        uint32_t imm = AtomicsImm::PackAtomic(
            cmd.value,
            static_cast<uint16_t>(roffset & AtomicsImm::kOFF_MASK))
            .GetImmData();

        // Use fi_writedata with zero-length to deliver the immediate.
        // The actual atomic is done via CQ data notification.
        fabric_write_with_data(fabric_ctx_, dst % kChannelPerProxy,
                               reinterpret_cast<void*>(fabric_ctx_.gpu_buf_base),
                               0, dst, 0, (uint64_t)imm,
                               reinterpret_cast<void*>(wr_id), true);
        (void)add_value;
        break;
      }

      case CmdType::BARRIER: {
        send_barrier(wr_id);
        break;
      }

      case CmdType::QUIET: {
        ctx_.quiet_wr = wr_id;
        quiet_cq();
        acked_wrs_.insert(wr_id);
        break;
      }

      default:
        fprintf(stderr, "Error: Unknown command type %d\n",
                static_cast<int>(cmd.cmd_type));
        std::abort();
    }
  }
}

// ---------------------------------------------------------------------------
// quiet_cq — libfabric path
// ---------------------------------------------------------------------------
void Proxy::quiet_cq() {
  constexpr int kConsecutiveEmptyToExit = 3;
  int empty_iters = 0;
  fi_cq_data_entry rx_entries[64];

  for (;;) {
    int ne_tx = fabric_poll_tx(fabric_ctx_, acked_wrs_);
    int ne_rx = fabric_poll_rx(fabric_ctx_, rx_entries, 64);

    // Process any rx completions (barrier/atomic notifications).
    for (int i = 0; i < ne_rx; i++) {
      if (rx_entries[i].flags & FI_REMOTE_CQ_DATA) {
        fabric_post_recv(fabric_ctx_);
      }
    }

    if (ne_tx > 0 || ne_rx > 0) {
      empty_iters = 0;
    } else {
      ++empty_iters;
    }
    if (empty_iters >= kConsecutiveEmptyToExit) break;
  }
}

// ---------------------------------------------------------------------------
// post_barrier_msg — libfabric path
// ---------------------------------------------------------------------------
void Proxy::post_barrier_msg(int dst_rank, bool ack, uint64_t seq) {
  uint32_t imm = BarrierImm::Pack(ack, (uint32_t)seq, (uint8_t)cfg_.rank);
  uint64_t wr_id = kBarrierWrTag | (0 & kBarrierMask);
  int ret = fabric_send_data(fabric_ctx_, dst_rank, (uint64_t)imm,
                             reinterpret_cast<void*>(wr_id));
  if (ret) {
    fprintf(stderr, "post_barrier_msg (fabric) failed: %s (%d)\n",
            fi_strerror(-ret), ret);
    std::abort();
  }
}

// ---------------------------------------------------------------------------
// destroy — libfabric path
// ---------------------------------------------------------------------------
void Proxy::destroy(bool free_gpu_buffer) {
  fabric_destroy(fabric_ctx_);

  if (free_gpu_buffer && cfg_.gpu_buffer) {
    cudaError_t e;
    if (cfg_.free_buffer_with_cuda_free_host) {
      e = cudaFreeHost(cfg_.gpu_buffer);
    } else {
      e = cudaFree(cfg_.gpu_buffer);
    }
    if (e != cudaSuccess) {
      fprintf(stderr, "[destroy] cudaFree failed: %s\n",
              cudaGetErrorString(e));
    } else {
      cfg_.gpu_buffer = nullptr;
    }
  }

#ifndef USE_SUBSET_BARRIER
  std::string const my_ip =
      (cfg_.rank < (int)peers_.size()) ? peers_[cfg_.rank].ip : "";
  std::string const shm_name = shm_name_for_barrier(my_ip, cfg_.thread_idx);
  unmap_local_barrier_shm(shm_name, ctx_.lb, ctx_.lb_owner);
  ctx_.lb = nullptr;
  ctx_.lb_owner = false;
#endif

  acked_wrs_.clear();
  wr_id_to_start_time_.clear();
  local_fabric_infos_.clear();
  remote_fabric_infos_.clear();
}
