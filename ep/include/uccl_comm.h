// UCCL Communication Library — C/C++ API
//
// A high-level communication interface built on top of UCCL-EP's optimized
// proxy architecture and libfabric/CXI transport.  Designed as a drop-in
// replacement for MPI one-sided / NVSHMEM in HPC programs.
//
// Key internal optimizations exposed through this API:
//   - 4 CPU proxy threads per rank, each with 8 FIFO channels (32 total)
//   - GPU-initiated RDMA: GPU kernels push commands to a lock-free FIFO,
//     proxy threads pick them up and issue fi_write over CXI
//   - Multi-rail: 8 libfabric endpoints per proxy thread for bandwidth
//   - Batching & backpressure: commands grouped by destination, inflight
//     limit prevents queue overflow
//
// Usage:
//   #include "uccl_comm.h"
//
//   // Bootstrap with MPI (or manual rank/IP assignment)
//   uccl_comm_t comm;
//   uccl_comm_init_mpi(&comm, MPI_COMM_WORLD, gpu_buf, gpu_buf_size);
//
//   // One-sided GPU RDMA put (CPU-initiated, through proxy)
//   uccl_put(comm, dst_rank, local_gpu_buf, remote_offset, size);
//   uccl_quiet(comm);  // wait for completion
//
//   // Get FIFO handles for GPU-initiated communication
//   uccl_d2h_channels_t channels;
//   uccl_get_d2h_channels(comm, &channels);
//   // Pass channels.addrs and channels.count to GPU kernel, then call
//   // nvshmemi_ibgda_put_nbi_warp() from device code (see uccl_ibgda.cuh).
//
//   // All-to-all: every rank writes to every other rank
//   uccl_alltoall(comm, sendbuf, recvbuf, msg_size);
//
//   // Barrier
//   uccl_barrier(comm);
//
//   uccl_comm_destroy(comm);

#ifndef UCCL_COMM_H
#define UCCL_COMM_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

// Opaque communicator handle.
typedef struct uccl_comm_impl* uccl_comm_t;

// D2H channel handles for GPU kernels.
typedef struct {
  uint64_t* addrs;   // Array of D2H FIFO addresses (device-accessible)
  int count;         // Number of channels (= kNumProxyThs * kChannelPerProxy)
  int num_proxies;   // Number of proxy threads (= kNumProxyThs)
  int channels_per_proxy;  // Channels per proxy (= kChannelPerProxy)
} uccl_d2h_channels_t;

// Communicator info.
typedef struct {
  int rank;
  int nranks;
  int local_rank;
  int num_nodes;
} uccl_comm_info_t;

// --------------------------------------------------------------------------
// Initialization / Finalize
// --------------------------------------------------------------------------

// Initialize using MPI for bootstrap.  Allocates proxy threads, registers
// GPU buffer, exchanges addresses with all peers.
// |gpu_buf|: cudaMalloc'd buffer for RDMA (send + receive region).
// |gpu_buf_size|: size in bytes.
int uccl_comm_init_mpi(uccl_comm_t* comm, void* mpi_comm,
                       void* gpu_buf, size_t gpu_buf_size);

// Initialize manually (without MPI).  The caller provides rank info and
// peer IP/port arrays.
int uccl_comm_init(uccl_comm_t* comm, int rank, int nranks,
                   int local_rank, int gpu_idx,
                   void* gpu_buf, size_t gpu_buf_size,
                   const char** peer_ips, const int* peer_base_ports);

// Destroy communicator: stops proxy threads, deregisters MRs, closes EPs.
int uccl_comm_destroy(uccl_comm_t comm);

// Query communicator info.
int uccl_comm_get_info(uccl_comm_t comm, uccl_comm_info_t* info);

// --------------------------------------------------------------------------
// GPU-initiated communication (FIFO path)
// --------------------------------------------------------------------------

// Get D2H channel handles.  Pass these to GPU kernels so they can call
// nvshmemi_ibgda_put_nbi_warp() from device code.
// The returned addresses are valid until uccl_comm_destroy().
int uccl_get_d2h_channels(uccl_comm_t comm, uccl_d2h_channels_t* channels);

// --------------------------------------------------------------------------
// CPU-initiated one-sided operations
// --------------------------------------------------------------------------

// Put |size| bytes from |local_buf| (GPU memory) to rank |dst_rank|'s
// GPU buffer at |remote_offset|.  Non-blocking: returns immediately
// after queuing the operation.
int uccl_put(uccl_comm_t comm, int dst_rank,
             const void* local_buf, size_t remote_offset, size_t size);

// Wait for all outstanding put operations from this rank to complete.
int uccl_quiet(uccl_comm_t comm);

// --------------------------------------------------------------------------
// Collective operations
// --------------------------------------------------------------------------

// All-to-all: each rank sends |sendcount| bytes from |sendbuf| to every
// other rank.  Rank r's data is written to offset r*sendcount in |recvbuf|
// on the destination.  |sendbuf| and |recvbuf| must be GPU memory.
// Blocking: returns after all data is transferred and visible.
int uccl_alltoall(uccl_comm_t comm,
                  const void* sendbuf, void* recvbuf, size_t sendcount);

// Barrier: blocks until all ranks have entered the barrier.
int uccl_barrier(uccl_comm_t comm);

// --------------------------------------------------------------------------
// Utility
// --------------------------------------------------------------------------

// Return the number of proxy threads per rank.
int uccl_get_num_proxy_threads(void);

// Return the number of FIFO channels per proxy thread.
int uccl_get_channels_per_proxy(void);

#ifdef __cplusplus
}
#endif

#endif  // UCCL_COMM_H
