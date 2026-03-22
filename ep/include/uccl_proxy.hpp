#pragma once
#include "bench_utils.hpp"
#include "fifo.hpp"
#include "proxy.hpp"
#include "ring_buffer.cuh"
#include <algorithm>
#include <atomic>
#include <memory>
#include <string>
#include <thread>
#include <vector>

class UcclProxy {
 public:
  UcclProxy(int thread_idx, uintptr_t gpu_buffer_addr, size_t total_size,
            int rank, int node_idx, int local_rank, int num_experts = 0,
            int num_ranks = 0, int num_nodes = 0, bool use_normal_mode = false,
            bool is_intranode = false,
            bool gpu_buffer_is_host_allocated = false,
            bool defer_ring_alloc = false);
  ~UcclProxy();

  void start_sender();
  void start_remote();
  void start_local();
  void start_dual();
  void stop();
  int get_listen_port() const { return proxy_->get_listen_port(); }

  // Set the offset of dispatch_rdma_recv_data_buffer within rdma_buffer
  void set_dispatch_recv_data_offset(uintptr_t offset) {
    proxy_->set_dispatch_recv_data_offset(offset);
  }

  void* get_atomic_buffer_ptr() {
    if (!atomic_buffer_ptr_) {
      fprintf(stderr, "Error: atomic_buffer_ptr_ is not set yet\n");
      std::abort();
    }
    return atomic_buffer_ptr_;
  }

  void set_atomic_buffer_ptr(void* ptr) {
    // printf("Set atomic_buffer_ptr_ to %p\n", ptr);
    atomic_buffer_ptr_ = ptr;
    proxy_->set_atomic_buffer_ptr(atomic_buffer_ptr_);
  }

  // Calculate and set dispatch_recv_data_offset automatically based on layout
  // parameters
  void calculate_and_set_dispatch_recv_data_offset(int num_tokens, int hidden,
                                                   int num_experts) {
    // Calculate layout parameters (same logic as ep_config.hpp and test)
    int num_scales = hidden / 128;
    size_t num_bytes_per_dispatch_msg =
        4 + std::max(hidden * 2, hidden + num_scales * 4);
    size_t dispatch_send_buffer_bytes = num_tokens * num_bytes_per_dispatch_msg;
    size_t combine_send_buffer_bytes =
        num_experts * num_tokens * hidden * 2;  // sizeof(bfloat16)
    size_t send_buffer_bytes =
        std::max(dispatch_send_buffer_bytes, combine_send_buffer_bytes);
    size_t dispatch_recv_count_buffer_bytes = num_experts * 4;
    size_t signaling_buffer_bytes_aligned =
        ((dispatch_recv_count_buffer_bytes + 127) / 128) * 128;
    uintptr_t dispatch_recv_data_offset =
        signaling_buffer_bytes_aligned * 2 + send_buffer_bytes * 2;
    proxy_->set_dispatch_recv_data_offset(dispatch_recv_data_offset);
    proxy_->cfg_.num_experts = num_experts;
  }

  std::vector<uint64_t> get_d2h_channel_addrs() const;
  int thread_idx() const noexcept { return thread_idx_; }
  void* gpu_buffer_addr() const noexcept { return gpu_buffer_addr_; }

  // Prevent stop()/destroy() from freeing the GPU buffer.  Use when the
  // caller (e.g. uccl_comm) owns the buffer lifetime.
  void release_gpu_buffer() { proxy_->cfg_.gpu_buffer = nullptr; }

  // Access the underlying Proxy for direct method calls (e.g. MPI-based
  // init_fabric_insert_peers).
  Proxy* proxy_access() { return proxy_.get(); }

#ifdef USE_LIBFABRIC
  // MPI-based init (no TCP).  All fabric operations (init, insert_peers,
  // polling) run on a SINGLE thread to satisfy CXI HMEM's requirement
  // that CUDA context and DMA ops share the same thread.
  //
  // The proxy thread does:
  //   1. fabric_init → write local_info → set init_done
  //   2. spin-wait for peers_done
  //   3. post recvs → enter polling loop
  //
  // The main thread does:
  //   1. spin-wait for init_done
  //   2. MPI_Allgather → insert_peers → set peers_done

  // Shared state for handshake (set by caller before start).
  FabricConnectionInfo* mpi_local_info_out = nullptr;
  std::vector<FabricConnectionInfo> const* mpi_remote_infos = nullptr;
  int mpi_num_ranks = 0;
  std::atomic<bool> init_done{false};
  std::atomic<bool> peers_done{false};

  // Start the proxy thread.  It will do fabric_init, then wait for
  // the main thread to signal peers_done before entering the poll loop.
  void start_mpi_init() {
    proxy_->set_progress_run(true);
    running_.store(true, std::memory_order_release);
    thread_ = std::thread([this]() {
      // Phase 1: fabric_init on this thread (NUMA + CUDA context).
      proxy_->init_fabric_local(*mpi_local_info_out);
      init_done.store(true, std::memory_order_release);

      // Phase 2: wait for main thread to exchange info and insert peers.
      while (!peers_done.load(std::memory_order_acquire)) {
        cpu_relax();
      }

      // Phase 3: main thread already called insert_peers — post recvs
      // and enter polling loop (all on THIS thread).
      proxy_->run_dual_after_init();
    });
  }

  void wait_fabric_init() {
    while (!init_done.load(std::memory_order_acquire)) {
      usleep(100);
    }
  }

  void signal_peers_ready() {
    peers_done.store(true, std::memory_order_release);
  }
#endif

  // Re-allocate ring buffers and update the running proxy's D2H queues.
  // Call AFTER all fabric_init() calls have completed (CXI fi_mr_regattr
  // with FI_HMEM_CUDA invalidates pre-existing cudaHostAlloc/cudaMallocManaged
  // registrations, so ring buffers must be allocated after fabric init).
  void reinit_ring_buffers() {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    // FIFO backend: free old FIFOs and create new ones.
    fifos.clear();
    d2h_channel_addrs_.clear();
    d2h_queues.clear();
    d2h_queues.resize(kChannelPerProxy);

    std::vector<d2hq::HostD2HHandle> new_handles;
    new_handles.reserve(kChannelPerProxy);
    for (size_t i = 0; i < kChannelPerProxy; ++i) {
      auto fifo = std::make_unique<mscclpp::Fifo>(kQueueSize);
      uintptr_t addr = reinterpret_cast<uintptr_t>(fifo.get());
      d2hq::init_from_addr(d2h_queues[i], addr);
      new_handles.push_back(d2h_queues[i]);
      d2h_channel_addrs_.push_back(addr);
      fifos.push_back(std::move(fifo));
    }
#else
    // Ring buffer backend.
    for (auto addr : d2h_channel_addrs_) {
      free_cmd_ring(addr);
    }
    d2h_channel_addrs_.clear();
    d2h_queues.clear();
    d2h_queues.resize(kChannelPerProxy);

    std::vector<d2hq::HostD2HHandle> new_handles;
    new_handles.reserve(kChannelPerProxy);
    for (size_t i = 0; i < kChannelPerProxy; ++i) {
      uintptr_t addr = alloc_cmd_ring();
      d2hq::init_from_addr(d2h_queues[i], addr);
      new_handles.push_back(d2h_queues[i]);
      d2h_channel_addrs_.push_back(addr);
    }
#endif

    fprintf(stderr, "[UcclProxy] reinit_ring_buffers: allocated %zu queues\n",
            d2h_channel_addrs_.size());

    // Update the running proxy (thread-safe via atomic flag).
    proxy_->update_d2h_queues(new_handles);
  }
  double avg_rdma_write_us() const { return proxy_->avg_rdma_write_us(); }
  double avg_wr_latency_us() const { return proxy_->avg_wr_latency_us(); }
  void set_peers_meta(std::vector<PeerMeta> const& peers);
  void set_bench_d2h_channel_addrs(std::vector<uintptr_t> const& addrs) {
    proxy_->set_bench_d2h_channel_addrs(addrs);
  }

 private:
  enum class Mode { None, Sender, Remote, Local, Dual };
  void start(Mode m);

  std::unique_ptr<Proxy> proxy_;
  std::thread thread_;
  Mode mode_;
  std::atomic<bool> running_;
  std::vector<uintptr_t> d2h_channel_addrs_;
  int thread_idx_;
  void* gpu_buffer_addr_;
  std::vector<PeerMeta> peers_;
  int local_rank_;
  void* atomic_buffer_ptr_;
  bool atomic_buffer_is_host_allocated_ =
      false;  // true => cudaFreeHost, false => cudaFree
  int node_idx_;
  bool is_intranode_;
  std::vector<d2hq::HostD2HHandle> d2h_queues;
  std::vector<std::unique_ptr<mscclpp::Fifo>> fifos;
};

// ============================================================================
// FIFO-based Proxy Wrapper
// ============================================================================

// Python-facing FIFO proxy wrapper that wraps the real Proxy class
class FifoProxy {
 public:
  FifoProxy(int thread_idx, uintptr_t gpu_buffer_addr, size_t total_size,
            int rank, int node_idx, int local_rank, bool is_intranode = false);
  ~FifoProxy();

  void set_fifo(mscclpp::Fifo* fifo);
  void set_peers_meta(std::vector<PeerMeta> const& meta);

  void start_sender();
  void start_remote();
  void stop();
  int get_listen_port() const { return proxy_->get_listen_port(); }

  double avg_wr_latency_us() const;
  uint64_t processed_count() const;

  int thread_idx;

 private:
  void run_sender();
  void run_remote();

  mscclpp::Fifo* fifo_;
  std::unique_ptr<Proxy> proxy_;  // Underlying Proxy for RDMA operations
  std::unique_ptr<std::thread> thread_;
  std::atomic<bool> stop_flag_;

  uintptr_t gpu_buffer_addr_;
  size_t total_size_;
  int rank_;
  int node_idx_;
  int local_rank_;
  bool is_intranode_;
};
