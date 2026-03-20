#pragma once

#include "common.hpp"
#include "barrier_local.hpp"
#include "util/gpu_rt.h"

#include <rdma/fabric.h>
#include <rdma/fi_domain.h>
#include <rdma/fi_endpoint.h>
#include <rdma/fi_eq.h>
#include <rdma/fi_cm.h>
#include <rdma/fi_rma.h>
#include <rdma/fi_atomic.h>
#include <rdma/fi_errno.h>
#include <rdma/fi_cxi_ext.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <unordered_map>
#include <vector>

// Registered memory region handle for libfabric.
struct FabricMR {
  fid_mr* mr = nullptr;
  void* desc = nullptr;   // fi_mr_desc(mr)
  uint64_t key = 0;       // fi_mr_key(mr), provider-assigned
};

// Connection metadata exchanged between peers via TCP OOB channel.
// Replaces RDMAConnectionInfo for the libfabric backend.
struct FabricConnectionInfo {
  // Control EP address (FI_ADDR_CXI = 8 bytes, but reserve space)
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

  // Multi-rail data EP addresses and MR keys
  uint32_t num_data_eps;
  uint8_t data_ep_addrs[kChannelPerProxy][64];
  size_t data_ep_addr_lens[kChannelPerProxy];
  uint64_t data_gpu_keys[kChannelPerProxy];
};

// Remote peer MR metadata (stored locally after connection exchange).
struct FabricRemoteMRInfo {
  uint64_t gpu_key;
  uintptr_t gpu_addr;
  uint64_t gpu_len;
  uint64_t atomic_key;
  uintptr_t atomic_addr;
  uint64_t atomic_len;
  uint64_t data_gpu_keys[kChannelPerProxy];
};

// All libfabric resources owned by one proxy thread.
// This replaces the ibverbs fields in ProxyCtx.
struct FabricCtx {
  // Global resources (per proxy thread)
  fi_info* info = nullptr;
  fid_fabric* fabric = nullptr;
  fid_domain* domain = nullptr;
  fid_av* av = nullptr;
  fid_cq* tx_cq = nullptr;
  fid_cq* rx_cq = nullptr;

  // Control EP (barrier, ack messages)
  fid_ep* ctrl_ep = nullptr;
  FabricMR ctrl_gpu_mr;
  FabricMR ctrl_atomic_mr;

  // Data EPs (multi-rail)
  fid_ep* data_eps[kChannelPerProxy] = {};
  FabricMR data_gpu_mrs[kChannelPerProxy];

  // Local scratch for fetch-atomic results
  uint64_t* atomic_old_values_buf = nullptr;
  FabricMR atomic_old_values_mr;
  static constexpr size_t kMaxAtomicOps = 1024;

  // Per-peer addressing
  std::vector<fi_addr_t> ctrl_peer_addrs;
  fi_addr_t data_peer_addrs[kChannelPerProxy][1024];  // [channel][rank]

  // Per-peer remote MR info
  std::vector<FabricRemoteMRInfo> remote_info;

  int numa_node = -1;
  int gpu_idx = -1;

  // Base address of the local GPU buffer (set during init)
  uintptr_t gpu_buf_base = 0;
};

// Per-peer proxy context.  Under libfabric the transport resources live in
// FabricCtx (shared across all peers on this thread); ProxyCtx holds only
// per-peer bookkeeping that is transport-independent.
struct ProxyCtx {
  // Transport-independent per-peer state
  uintptr_t remote_addr = 0;
  uint64_t remote_len = 0;
  int numa_node = -1;

  // Remote atomic buffer
  uintptr_t remote_atomic_buffer_addr = 0;
  uint64_t remote_atomic_buffer_len = 0;

  // Buffer offset within rdma_buffer for address translation
  uintptr_t dispatch_recv_data_offset = 0;

  // Progress/accounting
  std::atomic<uint64_t> completed{0};
  std::atomic<bool> progress_run{true};

  // GPU copy helpers
  gpuStream_t copy_stream = nullptr;
  bool peer_enabled[MAX_NUM_GPUS][MAX_NUM_GPUS] = {};
  size_t pool_index = 0;
  void* per_gpu_device_buf[MAX_NUM_GPUS] = {nullptr};

  uint32_t tag = 0;

  using DispatchTokenKey = std::tuple<int, int, int>;
  using CombineTokenKey = std::pair<int, int>;
  using NormalTokenKey = std::pair<int, int>;

  template <typename Key>
  class TokenCounter {
   public:
    using MapType = std::map<Key, size_t>;
    void Add(Key const& key, size_t k) { counter_[key] += k; }
    size_t Get(Key const& key) const {
      auto it = counter_.find(key);
      return (it == counter_.end()) ? 0 : it->second;
    }
    void Reset(Key const& key) { counter_[key] = 0; }
    void Clear() { counter_.clear(); }

   private:
    MapType counter_;
  };

  struct WriteStruct {
    int expert_idx;
    int dst_rank;
    bool is_combine;
    int low_latency_buffer_idx;
  };

  TokenCounter<DispatchTokenKey> dispatch_token_counter;
  TokenCounter<CombineTokenKey> combine_token_counter;
  TokenCounter<NormalTokenKey> normal_token_counter;

  std::unordered_map<uint64_t, WriteStruct> wr_id_to_write_struct;
  TokenCounter<DispatchTokenKey> dispatch_sent_counter;
  TokenCounter<DispatchTokenKey> combine_sent_counter;
  TokenCounter<NormalTokenKey> normal_sent_counter;

  // Async-barrier state
  bool barrier_inflight = false;
  uint64_t barrier_seq = 0;
  int barrier_wr = -1;

  bool quiet_inflight = false;
  int quiet_wr = -1;

  // Rank-0 bookkeeping
  std::vector<uint8_t> barrier_arrived;
  int barrier_arrival_count = 0;

  // Followers: release flag from rank-0
  bool barrier_released = false;
  uint64_t barrier_release_seq = 0;

  // Intra-node (shared-memory) barrier state
  LocalBarrier* lb = nullptr;
  bool lb_owner = false;
  int num_local_ranks = 0;
  int node_leader_rank = -1;
  int local_rank = -1;
  int thread_idx = -1;

  std::unordered_map<uint64_t, uint8_t> next_seq_per_index;
  inline uint64_t seq_key(int dst_rank, size_t index) {
    return (static_cast<uint64_t>(static_cast<uint32_t>(dst_rank)) << 32) ^
           static_cast<uint64_t>(static_cast<uint32_t>(index));
  }
};
