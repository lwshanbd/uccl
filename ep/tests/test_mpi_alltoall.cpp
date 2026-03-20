// MPI baseline all-to-all bandwidth test for comparison with uccl_comm.
// Usage: mpiexec -n N --ppn P ./test_mpi_alltoall [msg_size_bytes]

#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

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
  int num_iters = 50;
  int warmup = 5;

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

  // Allocate GPU buffers for MPI_Alltoall.
  // sendbuf: nranks * msg_size, recvbuf: nranks * msg_size
  size_t total = msg_size * nranks;
  void* sendbuf = nullptr;
  void* recvbuf = nullptr;
  check_cuda(cudaMalloc(&sendbuf, total), "cudaMalloc send");
  check_cuda(cudaMalloc(&recvbuf, total), "cudaMalloc recv");

  // Fill send buffer with rank-specific pattern.
  std::vector<uint8_t> pattern(total);
  for (size_t i = 0; i < total; i++) {
    pattern[i] = (uint8_t)((rank + i) & 0xFF);
  }
  check_cuda(cudaMemcpy(sendbuf, pattern.data(), total,
                        cudaMemcpyHostToDevice), "H2D");
  check_cuda(cudaMemset(recvbuf, 0, total), "memset recv");

  if (rank == 0) {
    fprintf(stderr,
            "=== MPI_Alltoall baseline: %d ranks, msg=%zu, GPU-aware ===\n",
            nranks, msg_size);
  }

  MPI_Barrier(MPI_COMM_WORLD);

  for (int iter = 0; iter < warmup + num_iters; iter++) {
    check_cuda(cudaMemset(recvbuf, 0, total), "reset recv");
    MPI_Barrier(MPI_COMM_WORLD);

    auto t0 = std::chrono::high_resolution_clock::now();

    MPI_Alltoall(sendbuf, msg_size, MPI_BYTE,
                 recvbuf, msg_size, MPI_BYTE,
                 MPI_COMM_WORLD);

    auto t1 = std::chrono::high_resolution_clock::now();

    if (iter >= warmup && rank == 0) {
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      double total_bytes = (double)nranks * (nranks - 1) * msg_size;
      double bw_gbps = total_bytes / us * 1e6 / 1e9;
      fprintf(stderr, "  iter %3d: %.1f us, %.2f GB/s aggregate\n",
              iter, us, bw_gbps);
    }
  }

  // Verify: check data from rank 0.
  if (rank != 0) {
    std::vector<uint8_t> result(msg_size);
    check_cuda(cudaMemcpy(result.data(), recvbuf, msg_size,
                          cudaMemcpyDeviceToHost), "D2H verify");
    bool ok = true;
    for (size_t i = 0; i < msg_size; i++) {
      if (result[i] != (uint8_t)((0 + i) & 0xFF)) {
        fprintf(stderr, "[rank %d] MPI A2A data mismatch at %zu\n", rank, i);
        ok = false;
        break;
      }
    }
    fprintf(stderr, "[rank %d] MPI A2A verify %s\n", rank, ok ? "PASSED" : "FAILED");
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) fprintf(stderr, "=== MPI baseline done ===\n");

  cudaFree(sendbuf);
  cudaFree(recvbuf);
  MPI_Finalize();
  return 0;
}
