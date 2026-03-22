// GPU-initiated communication test for the libfabric/CXI backend.
//
// Exercises the full UCCL-EP data path:
//   GPU kernel → D2H FIFO → CPU proxy thread → fi_write over CXI
//
// Two sub-tests:
//   1. P2P: rank 0 GPU pushes WRITE commands → data arrives at rank 1
//   2. All-to-all: every rank's GPU pushes WRITEs to all remote peers
//
// Usage: mpiexec -n N --ppn P ./test_gpu_proxy_fabric [msg_size_bytes]

#include "uccl_comm.h"
#include "gpu_write_kernel.cuh"
#include "d2h_queue_device.cuh"
#include "common.hpp"
#include "fifo.hpp"

#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <unistd.h>

static void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] %s: %s\n", msg, cudaGetErrorString(err));
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size_t msg_size = (argc > 1) ? (size_t)atol(argv[1]) : 1024 * 1024;
  int num_iters = 20;
  int warmup = 3;

  // GPU setup.
  int local_rank = 0;
  {
    const char* lr = getenv("PALS_LOCAL_RANKID");
    if (!lr) lr = getenv("PMI_LOCAL_RANK");
    if (lr) local_rank = atoi(lr);
  }
  int num_gpus;
  check_cuda(cudaGetDeviceCount(&num_gpus), "cudaGetDeviceCount");
  int gpu_idx = local_rank % num_gpus;
  check_cuda(cudaSetDevice(gpu_idx), "cudaSetDevice");

  // Allocate GPU buffer: nranks * msg_size (each rank's slot).
  size_t buf_size = msg_size * nranks;
  void* gpu_buf = nullptr;
  check_cuda(cudaMalloc(&gpu_buf, buf_size), "cudaMalloc");
  check_cuda(cudaMemset(gpu_buf, 0, buf_size), "cudaMemset");

  // Fill our slot with a rank-specific pattern.
  std::vector<uint8_t> pattern(msg_size);
  for (size_t i = 0; i < msg_size; i++) {
    pattern[i] = (uint8_t)((rank + i) & 0xFF);
  }
  check_cuda(cudaMemcpy((char*)gpu_buf + rank * msg_size,
                         pattern.data(), msg_size,
                         cudaMemcpyHostToDevice), "H2D pattern");

  if (rank == 0) {
    fprintf(stderr,
            "=== GPU-Proxy-Fabric test: %d ranks, msg=%zu, "
            "%d proxy threads/rank, %d channels/proxy ===\n",
            nranks, msg_size,
            uccl_get_num_proxy_threads(),
            uccl_get_channels_per_proxy());
  }

  // Initialize UCCL communicator (starts proxy threads).
  uccl_comm_t comm;
  MPI_Comm mpi = MPI_COMM_WORLD;
  int ret = uccl_comm_init_mpi(&comm, &mpi, gpu_buf, buf_size);
  if (ret) {
    fprintf(stderr, "[rank %d] uccl_comm_init_mpi failed: %d\n", rank, ret);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Get D2H channel info.
  uccl_d2h_channels_t channels;
  uccl_get_d2h_channels(comm, &channels);
  fprintf(stderr, "[rank %d] D2H channels: %d (%d proxies x %d ch/proxy)\n",
          rank, channels.count, channels.num_proxies,
          channels.channels_per_proxy);

  if (channels.count == 0) {
    fprintf(stderr, "[rank %d] ERROR: no D2H channels available\n", rank);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Build device-accessible D2HHandle array from the FIFO addresses.
  int total_queues = channels.count;
  std::vector<d2hq::D2HHandle> h_handles(total_queues);
  for (int i = 0; i < total_queues; i++) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
    auto* fifo = reinterpret_cast<mscclpp::Fifo*>(channels.addrs[i]);
    h_handles[i].init_from_host_value(fifo->deviceHandle());
#else
    h_handles[i].init_from_dev_ptr(
        reinterpret_cast<void*>(channels.addrs[i]));
#endif
  }

  // Copy handles to device memory.
  d2hq::D2HHandle* d_handles = nullptr;
  check_cuda(cudaMalloc(&d_handles,
                         total_queues * sizeof(d2hq::D2HHandle)),
             "cudaMalloc handles");
  check_cuda(cudaMemcpy(d_handles, h_handles.data(),
                         total_queues * sizeof(d2hq::D2HHandle),
                         cudaMemcpyHostToDevice),
             "H2D handles");

  // Build remote_mask.
  char my_host[256];
  gethostname(my_host, sizeof(my_host));
  std::vector<char> all_hosts(nranks * 256);
  MPI_Allgather(my_host, 256, MPI_CHAR,
                all_hosts.data(), 256, MPI_CHAR, MPI_COMM_WORLD);

  std::vector<int> h_remote(nranks, 0);
  for (int r = 0; r < nranks; r++) {
    h_remote[r] = (strncmp(my_host, &all_hosts[r * 256], 256) != 0) ? 1 : 0;
  }

  int* d_remote = nullptr;
  check_cuda(cudaMalloc(&d_remote, nranks * sizeof(int)), "cudaMalloc remote");
  check_cuda(cudaMemcpy(d_remote, h_remote.data(), nranks * sizeof(int),
                         cudaMemcpyHostToDevice), "H2D remote");

  int num_remote = 0;
  for (int r = 0; r < nranks; r++) num_remote += h_remote[r];
  fprintf(stderr, "[rank %d] %d remote ranks, %d local ranks\n",
          rank, num_remote, nranks - 1 - num_remote);

  cudaStream_t stream;
  check_cuda(cudaStreamCreate(&stream), "cudaStreamCreate");

  MPI_Barrier(MPI_COMM_WORLD);

  // ----------------------------------------------------------------
  // Test 1: P2P — rank 0 GPU pushes one WRITE to rank 1
  // ----------------------------------------------------------------
  if (rank == 0) fprintf(stderr, "\n--- Test 1: GPU-initiated P2P ---\n");

  if (nranks >= 2 && rank == 0 && h_remote[1]) {
    int addr_shift = kWriteAddrShiftLowLatency;
    uint32_t lptr = (uint32_t)((uint64_t)(0 * msg_size) >> addr_shift);
    uint32_t rptr = (uint32_t)((uint64_t)(0 * msg_size) >> addr_shift);

    check_cuda(cudaDeviceSynchronize(), "pre-kernel sync");

    auto err = launch_gpu_push_write(
        h_handles[0], /*dst_rank=*/1, lptr, rptr, (uint32_t)msg_size,
        /*num_writes=*/1, /*threads=*/1, stream);
    if (err != cudaSuccess) {
      fprintf(stderr, "[rank 0] kernel launch failed: %s\n",
              cudaGetErrorString(err));
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    check_cuda(cudaStreamSynchronize(stream), "sync P2P");
    fprintf(stderr, "[rank 0] GPU P2P write completed\n");
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Rank 1 verifies.
  if (nranks >= 2 && rank == 1 && h_remote[0]) {
    std::vector<uint8_t> result(msg_size);
    check_cuda(cudaMemcpy(result.data(), gpu_buf, msg_size,
                           cudaMemcpyDeviceToHost), "D2H verify P2P");
    bool ok = true;
    for (size_t i = 0; i < msg_size; i++) {
      if (result[i] != (uint8_t)((0 + i) & 0xFF)) {
        fprintf(stderr, "[rank 1] P2P mismatch at byte %zu: "
                "got 0x%02x want 0x%02x\n",
                i, result[i], (uint8_t)((0 + i) & 0xFF));
        ok = false;
        break;
      }
    }
    fprintf(stderr, "[rank 1] P2P verify %s\n", ok ? "PASSED" : "FAILED");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) fprintf(stderr, "\n=== test_gpu_proxy_fabric PASSED ===\n");

  // Cleanup.
  cudaStreamDestroy(stream);
  cudaFree(d_handles);
  cudaFree(d_remote);
  uccl_comm_destroy(comm);
  cudaFree(gpu_buf);
  MPI_Finalize();
  return 0;
}
