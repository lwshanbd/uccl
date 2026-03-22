// UCCL Communication Library — Implementation
//
// Wraps the UCCL-EP proxy architecture (UcclProxy, Proxy, FabricCtx)
// into the clean C API defined in uccl_comm.h.

#include "uccl_comm.h"
#include "uccl_proxy.hpp"
#include "common.hpp"
#include "proxy.hpp"

#ifdef USE_LIBFABRIC
#include "fabric.hpp"
#endif

#include <mpi.h>
#include <cuda_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <string>
#include <vector>
#include <arpa/inet.h>
#include <unistd.h>

// --------------------------------------------------------------------------
// Internal state
// --------------------------------------------------------------------------

struct uccl_comm_impl {
  int rank;
  int nranks;
  int local_rank;
  int gpu_idx;
  int num_nodes;

  void* gpu_buf;
  size_t gpu_buf_size;

  // Per-rank info for intra/inter-node routing.
  struct RankInfo {
    char hostname[256];
    int gpu_idx;
    void* gpu_buf;       // remote GPU buffer (IPC-opened for same-node)
    bool is_same_node;
  };
  std::vector<RankInfo> rank_info;
  cudaStream_t copy_stream = nullptr;  // for intra-node cudaMemcpyPeerAsync

  // Proxy threads (kNumProxyThs per rank)
  std::vector<std::unique_ptr<UcclProxy>> proxies;

  // Aggregated D2H channel addresses (for GPU kernels)
  std::vector<uint64_t> d2h_addrs;

#ifdef USE_LIBFABRIC
  FabricCtx cpu_fabric_ctx;
  bool cpu_ctx_initialized = false;
#endif

  MPI_Comm mpi_comm;
  bool owns_mpi = false;
};

// --------------------------------------------------------------------------
// Helper: get hostname/IP
// --------------------------------------------------------------------------
static std::string get_local_ip() {
  char hostname[256];
  gethostname(hostname, sizeof(hostname));
  return std::string(hostname);
}

// --------------------------------------------------------------------------
// Init with MPI
// --------------------------------------------------------------------------

int uccl_comm_init_mpi(uccl_comm_t* out_comm, void* mpi_comm_ptr,
                       void* gpu_buf, size_t gpu_buf_size) {
  MPI_Comm mpi_comm = *(MPI_Comm*)mpi_comm_ptr;

  int rank, nranks;
  MPI_Comm_rank(mpi_comm, &rank);
  MPI_Comm_size(mpi_comm, &nranks);

  // Determine local_rank from environment (avoids MPI_Allgather before
  // fabric_init — cray-mpich's MPI_Allgather uses CXI internally and can
  // interfere with subsequent fi_mr_regattr for GPU HMEM).
  char hostname[256];
  gethostname(hostname, sizeof(hostname));

  int local_rank = 0;
  char const* lr_env = getenv("PALS_LOCAL_RANKID");
  if (!lr_env) lr_env = getenv("PMI_LOCAL_RANK");
  if (!lr_env) lr_env = getenv("OMPI_COMM_WORLD_LOCAL_RANK");
  if (lr_env) {
    local_rank = atoi(lr_env);
  }

  int num_gpus = 0;
  cudaGetDeviceCount(&num_gpus);
  int num_local = (num_gpus > 0) ? num_gpus : 1;
  int num_nodes = (nranks + num_local - 1) / num_local;

  int gpu_idx = local_rank % (num_gpus > 0 ? num_gpus : 1);
  cudaSetDevice(gpu_idx);

  fprintf(stderr,
          "[uccl_comm] rank=%d/%d local_rank=%d gpu=%d nodes=%d\n",
          rank, nranks, local_rank, gpu_idx, num_nodes);

  auto* comm = new uccl_comm_impl();
  comm->rank = rank;
  comm->nranks = nranks;
  comm->local_rank = local_rank;
  comm->gpu_idx = gpu_idx;
  comm->num_nodes = num_nodes;
  comm->gpu_buf = gpu_buf;
  comm->gpu_buf_size = gpu_buf_size;
  comm->mpi_comm = mpi_comm;

#ifdef USE_LIBFABRIC
  // Initialize fabric EARLY — before any proxy threads or complex MPI
  // collectives, to avoid CXI resource conflicts with cray-mpich.
  {
    fabric_init(comm->cpu_fabric_ctx, gpu_buf, gpu_buf_size,
                nullptr, 0,
                gpu_idx, rank, /*thread_idx=*/0, local_rank);

    FabricConnectionInfo local_info{};
    fabric_get_local_info(comm->cpu_fabric_ctx, gpu_buf, gpu_buf_size,
                          nullptr, 0, local_info);

    std::vector<FabricConnectionInfo> all_infos(nranks);
    MPI_Allgather(&local_info, sizeof(FabricConnectionInfo), MPI_BYTE,
                  all_infos.data(), sizeof(FabricConnectionInfo), MPI_BYTE,
                  mpi_comm);

    for (int r = 0; r < nranks; r++) {
      if (r == rank) continue;
      fabric_insert_peer(comm->cpu_fabric_ctx, r, all_infos[r]);
    }
    comm->cpu_ctx_initialized = true;
    fprintf(stderr, "[uccl_comm] rank=%d: fabric context initialized\n", rank);
  }
#endif

  // Exchange hostnames and GPU IPC handles for intra-node routing.
  {
    struct RankExchange {
      char hostname[256];
      int gpu_idx;
      cudaIpcMemHandle_t ipc_handle;
    };

    RankExchange my_info{};
    gethostname(my_info.hostname, sizeof(my_info.hostname));
    my_info.gpu_idx = gpu_idx;
    cudaIpcGetMemHandle(&my_info.ipc_handle, gpu_buf);

    std::vector<RankExchange> all_info(nranks);
    MPI_Allgather(&my_info, sizeof(RankExchange), MPI_BYTE,
                  all_info.data(), sizeof(RankExchange), MPI_BYTE, mpi_comm);

    comm->rank_info.resize(nranks);
    cudaStreamCreate(&comm->copy_stream);

    for (int r = 0; r < nranks; r++) {
      auto& ri = comm->rank_info[r];
      strncpy(ri.hostname, all_info[r].hostname, sizeof(ri.hostname));
      ri.gpu_idx = all_info[r].gpu_idx;
      ri.is_same_node = (strncmp(hostname, all_info[r].hostname, 256) == 0);

      if (r == rank) {
        ri.gpu_buf = gpu_buf;  // self
      } else if (ri.is_same_node) {
        // Open IPC handle for same-node peer — enables NVLink access.
        cudaIpcOpenMemHandle(&ri.gpu_buf, all_info[r].ipc_handle,
                             cudaIpcMemLazyEnablePeerAccess);
        fprintf(stderr, "[uccl_comm] rank=%d: IPC opened to rank %d (gpu %d)\n",
                rank, r, ri.gpu_idx);
      } else {
        ri.gpu_buf = nullptr;  // inter-node, use fabric
      }
    }
  }

  int node_idx = rank / num_local;

  int num_proxies = kNumProxyThs;
  for (int t = 0; t < num_proxies; t++) {
    auto proxy = std::make_unique<UcclProxy>(
        t,
        reinterpret_cast<uintptr_t>(gpu_buf),
        gpu_buf_size,
        rank, node_idx, local_rank,
        /*num_experts=*/0,
        /*num_ranks=*/nranks,
        /*num_nodes=*/num_nodes,
        /*use_normal_mode=*/false,
        /*is_intranode=*/false,
        /*gpu_buffer_is_host_allocated=*/false,
        /*defer_ring_alloc=*/true);
    comm->proxies.push_back(std::move(proxy));
  }

  // Build PeerMeta (IP + GPU ptr) for each rank — no listen_ports needed
  // since we use MPI for connection exchange instead of TCP.
  struct RankMeta {
    char ip[256];
    uintptr_t gpu_ptr;
    size_t gpu_size;
  };

  RankMeta my_meta{};
  strncpy(my_meta.ip, hostname, sizeof(my_meta.ip) - 1);
  my_meta.gpu_ptr = reinterpret_cast<uintptr_t>(gpu_buf);
  my_meta.gpu_size = gpu_buf_size;

  std::vector<RankMeta> all_meta(nranks);
  MPI_Allgather(&my_meta, sizeof(RankMeta), MPI_BYTE,
                all_meta.data(), sizeof(RankMeta), MPI_BYTE, mpi_comm);

  std::vector<PeerMeta> peers(nranks);
  for (int r = 0; r < nranks; r++) {
    peers[r].rank = r;
    peers[r].ptr = all_meta[r].gpu_ptr;
    peers[r].nbytes = all_meta[r].gpu_size;
    peers[r].ip = std::string(all_meta[r].ip);
    // listen_ports unused (MPI exchange, no TCP).
    memset(peers[r].listen_ports, 0, sizeof(peers[r].listen_ports));
  }

  // Share atomic buffer pointer across proxies.
  void* atomic_ptr = nullptr;
  if (!comm->proxies.empty()) {
    atomic_ptr = comm->proxies[0]->get_atomic_buffer_ptr();
    if (atomic_ptr) {
      for (auto& proxy : comm->proxies) {
        proxy->set_atomic_buffer_ptr(atomic_ptr);
      }
    }
    for (auto& proxy : comm->proxies) {
      proxy->set_peers_meta(peers);
    }
  }

  // ---- MPI-based proxy init (no TCP) ----
  //
  // All fabric operations (init, insert_peers, polling) run on a SINGLE
  // proxy thread to satisfy CXI HMEM's requirement that the CUDA context
  // and DMA ops share the same thread.
  //
  // Step 1: launch proxy threads — each does fabric_init then waits.
  std::vector<FabricConnectionInfo> local_infos(num_proxies);
  for (int t = 0; t < num_proxies; t++) {
    auto& p = comm->proxies[t];
    p->mpi_local_info_out = &local_infos[t];
    p->mpi_num_ranks = nranks;
    p->start_mpi_init();
  }

  // Step 2: wait for all proxy threads to finish fabric_init.
  for (int t = 0; t < num_proxies; t++) {
    comm->proxies[t]->wait_fabric_init();
  }

  // Step 3: MPI_Allgather to exchange FabricConnectionInfo.
  int infos_per_rank = num_proxies;
  std::vector<FabricConnectionInfo> all_infos(nranks * infos_per_rank);
  MPI_Allgather(local_infos.data(),
                infos_per_rank * sizeof(FabricConnectionInfo), MPI_BYTE,
                all_infos.data(),
                infos_per_rank * sizeof(FabricConnectionInfo), MPI_BYTE,
                mpi_comm);

  // Step 4: insert peers into each proxy's fabric context.
  //         This is called on the main thread but only writes to data
  //         structures; the proxy thread is spin-waiting and not touching
  //         these fields.
  for (int t = 0; t < num_proxies; t++) {
    std::vector<FabricConnectionInfo> remote_for_t(nranks);
    for (int r = 0; r < nranks; r++) {
      remote_for_t[r] = all_infos[r * infos_per_rank + t];
    }
    comm->proxies[t]->proxy_access()->init_fabric_insert_peers(
        nranks, remote_for_t);
  }

  // Step 5: allocate ring buffers AFTER all fabric_init calls (CXI HMEM
  //         invalidates pre-existing cudaHostAlloc registrations).
  comm->d2h_addrs.clear();
  for (int t = 0; t < num_proxies; t++) {
    comm->proxies[t]->reinit_ring_buffers();
    auto addrs = comm->proxies[t]->get_d2h_channel_addrs();
    for (auto a : addrs) {
      comm->d2h_addrs.push_back(a);
    }
  }

  // Step 6: signal proxy threads to enter polling loop.
  for (int t = 0; t < num_proxies; t++) {
    comm->proxies[t]->signal_peers_ready();
  }

  fprintf(stderr,
          "[uccl_comm] rank=%d: initialized %d proxy threads, "
          "%zu D2H channels (MPI exchange, no TCP)\n",
          rank, num_proxies, comm->d2h_addrs.size());

  MPI_Barrier(mpi_comm);
  *out_comm = comm;
  return 0;
}

// --------------------------------------------------------------------------
// Destroy
// --------------------------------------------------------------------------

int uccl_comm_destroy(uccl_comm_t comm) {
  if (!comm) return -1;

  // Stop proxy threads.  Caller owns the GPU buffer, so release it
  // from the proxies first to prevent double-free.
  for (auto& proxy : comm->proxies) {
    proxy->release_gpu_buffer();
    proxy->stop();
  }
  comm->proxies.clear();

#ifdef USE_LIBFABRIC
  if (comm->cpu_ctx_initialized) {
    fabric_destroy(comm->cpu_fabric_ctx);
  }
#endif

  delete comm;
  return 0;
}

// --------------------------------------------------------------------------
// Query
// --------------------------------------------------------------------------

int uccl_comm_get_info(uccl_comm_t comm, uccl_comm_info_t* info) {
  if (!comm || !info) return -1;
  info->rank = comm->rank;
  info->nranks = comm->nranks;
  info->local_rank = comm->local_rank;
  info->num_nodes = comm->num_nodes;
  return 0;
}

int uccl_get_num_proxy_threads(void) { return kNumProxyThs; }
int uccl_get_channels_per_proxy(void) { return kChannelPerProxy; }

// --------------------------------------------------------------------------
// D2H channels (for GPU kernels)
// --------------------------------------------------------------------------

int uccl_get_d2h_channels(uccl_comm_t comm, uccl_d2h_channels_t* channels) {
  if (!comm || !channels) return -1;
  channels->addrs = comm->d2h_addrs.data();
  channels->count = static_cast<int>(comm->d2h_addrs.size());
  channels->num_proxies = kNumProxyThs;
  channels->channels_per_proxy = kChannelPerProxy;
  return 0;
}

// --------------------------------------------------------------------------
// CPU-initiated put
// --------------------------------------------------------------------------

int uccl_put(uccl_comm_t comm, int dst_rank,
             const void* local_buf, size_t remote_offset, size_t size) {
  if (!comm || dst_rank < 0 || dst_rank >= comm->nranks) return -1;
  if (dst_rank == comm->rank) return -1;

#ifdef USE_LIBFABRIC
  if (!comm->cpu_ctx_initialized) return -1;
  int channel = dst_rank % kChannelPerProxy;
  return fabric_write_with_data(
      comm->cpu_fabric_ctx, channel,
      const_cast<void*>(local_buf), size,
      dst_rank, remote_offset,
      /*cq_data=*/0,
      /*context=*/nullptr,
      /*signaled=*/true);
#else
  // ibverbs path: not implemented in this file
  fprintf(stderr, "[uccl_comm] uccl_put: ibverbs path not implemented\n");
  return -1;
#endif
}

// --------------------------------------------------------------------------
// Quiet
// --------------------------------------------------------------------------

int uccl_quiet(uccl_comm_t comm) {
  if (!comm) return -1;

#ifdef USE_LIBFABRIC
  if (!comm->cpu_ctx_initialized) return -1;
  std::unordered_set<uint64_t> acked;
  int polls = 0;
  // Poll until no more pending completions.
  int empty = 0;
  while (empty < 3 && polls < 10000000) {
    int ne = fabric_poll_tx(comm->cpu_fabric_ctx, acked);
    if (ne == 0) empty++;
    else empty = 0;
    polls++;
  }
  return 0;
#else
  return -1;
#endif
}

// --------------------------------------------------------------------------
// All-to-all
// --------------------------------------------------------------------------

int uccl_alltoall(uccl_comm_t comm,
                  const void* sendbuf, void* recvbuf, size_t sendcount) {
  if (!comm) return -1;

  int rank = comm->rank;
  int nranks = comm->nranks;

  // Phase 1: Intra-node via NVLink (cudaMemcpyAsync through IPC pointers).
  for (int dst = 0; dst < nranks; dst++) {
    if (dst == rank) continue;
    if (!comm->rank_info[dst].is_same_node) continue;
    void* dst_ptr = (char*)comm->rank_info[dst].gpu_buf + rank * sendcount;
    cudaMemcpyAsync(dst_ptr, sendbuf, sendcount, cudaMemcpyDeviceToDevice,
                    comm->copy_stream);
  }

#ifdef USE_LIBFABRIC
  // Phase 2: Inter-node via CXI (fi_write, same as before).
  if (comm->cpu_ctx_initialized) {
    int remote_count = 0;
    for (int dst = 0; dst < nranks; dst++) {
      if (dst == rank) continue;
      if (comm->rank_info[dst].is_same_node) continue;
      int channel = dst % kChannelPerProxy;
      int ret = fabric_write_with_data(
          comm->cpu_fabric_ctx, channel,
          const_cast<void*>(sendbuf), sendcount,
          dst, rank * sendcount,
          0, (void*)(uint64_t)(dst + 1), true);
      if (ret) return ret;
      remote_count++;
    }

    if (remote_count > 0) {
      std::unordered_set<uint64_t> acked;
      while ((int)acked.size() < remote_count) {
        fabric_poll_tx(comm->cpu_fabric_ctx, acked);
      }
    }
  }
#endif

  cudaStreamSynchronize(comm->copy_stream);
  MPI_Barrier(comm->mpi_comm);
  return 0;
}

// --------------------------------------------------------------------------
// Barrier
// --------------------------------------------------------------------------

int uccl_barrier(uccl_comm_t comm) {
  if (!comm) return -1;
  MPI_Barrier(comm->mpi_comm);
  return 0;
}

// --------------------------------------------------------------------------
// Manual init (without MPI)
// --------------------------------------------------------------------------

int uccl_comm_init(uccl_comm_t* comm, int rank, int nranks,
                   int local_rank, int gpu_idx,
                   void* gpu_buf, size_t gpu_buf_size,
                   const char** peer_ips, const int* peer_base_ports) {
  // TODO: implement TCP-only bootstrap without MPI
  (void)comm; (void)rank; (void)nranks; (void)local_rank; (void)gpu_idx;
  (void)gpu_buf; (void)gpu_buf_size; (void)peer_ips; (void)peer_base_ports;
  fprintf(stderr, "[uccl_comm] Manual init not yet implemented, use MPI\n");
  return -1;
}
