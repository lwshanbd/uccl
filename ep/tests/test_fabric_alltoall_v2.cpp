// All-to-all benchmark using the uccl_comm API.
// Intra-node: NVLink via CUDA IPC (handled internally by uccl_alltoall)
// Inter-node: CXI fi_write with FI_MORE batching
//
// Usage: mpiexec -n N --ppn P ./test_fabric_alltoall_v2 [msg_size]

#include "uccl_comm.h"
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
  int num_iters = 100;
  int warmup = 10;

  // GPU setup via uccl_comm's env var detection.
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

  // Allocate GPU buffer: [rank0 slot | rank1 slot | ... | rankN-1 slot]
  // Each slot = msg_size bytes.  Our send data is in slot[rank].
  size_t buf_size = msg_size * nranks;
  void* gpu_buf = nullptr;
  check_cuda(cudaMalloc(&gpu_buf, buf_size), "cudaMalloc");
  check_cuda(cudaMemset(gpu_buf, 0, buf_size), "cudaMemset");

  // Fill our own slot with rank-specific pattern.
  std::vector<uint8_t> pattern(msg_size);
  for (size_t i = 0; i < msg_size; i++) {
    pattern[i] = (uint8_t)((rank + i) & 0xFF);
  }
  check_cuda(cudaMemcpy((char*)gpu_buf + rank * msg_size,
                        pattern.data(), msg_size,
                        cudaMemcpyHostToDevice), "H2D");

  if (rank == 0) {
    fprintf(stderr,
            "=== UCCL alltoall: %d ranks, msg=%zu, %d GPUs/node ===\n",
            nranks, msg_size, num_gpus);
  }

  // Initialize UCCL communicator.
  uccl_comm_t comm;
  MPI_Comm mpi = MPI_COMM_WORLD;
  int ret = uccl_comm_init_mpi(&comm, &mpi, gpu_buf, buf_size);
  if (ret) {
    fprintf(stderr, "[rank %d] uccl_comm_init_mpi failed: %d\n", rank, ret);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // Benchmark loop.
  // sendbuf = our slot in gpu_buf (offset rank * msg_size)
  void* sendbuf = (char*)gpu_buf + rank * msg_size;

  for (int iter = 0; iter < warmup + num_iters; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);
    auto t0 = std::chrono::high_resolution_clock::now();

    ret = uccl_alltoall(comm, sendbuf, gpu_buf, msg_size);
    if (ret) {
      fprintf(stderr, "[rank %d] uccl_alltoall failed: %d\n", rank, ret);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    auto t1 = std::chrono::high_resolution_clock::now();

    if (iter >= warmup && rank == 0) {
      double us = std::chrono::duration<double, std::micro>(t1 - t0).count();
      double total_bytes = (double)nranks * (nranks - 1) * msg_size;
      double bw_gbps = total_bytes / us * 1e6 / 1e9;
      fprintf(stderr, "  iter %3d: %.1f us, %.2f GB/s aggregate\n",
              iter, us, bw_gbps);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // Verify all pairs — log intra vs inter for each.
  uccl_comm_info_t info;
  uccl_comm_get_info(comm, &info);
  char my_host[256];
  gethostname(my_host, sizeof(my_host));

  // Get all hostnames to classify intra/inter.
  std::vector<char> all_hosts(nranks * 256);
  MPI_Allgather(my_host, 256, MPI_CHAR,
                all_hosts.data(), 256, MPI_CHAR, MPI_COMM_WORLD);

  int errors = 0;
  int intra_ok = 0, inter_ok = 0;
  for (int src = 0; src < nranks; src++) {
    if (src == rank) continue;
    bool same_node = (strncmp(my_host, &all_hosts[src * 256], 256) == 0);
    std::vector<uint8_t> result(msg_size);
    check_cuda(cudaMemcpy(result.data(), (char*)gpu_buf + src * msg_size,
                          msg_size, cudaMemcpyDeviceToHost), "D2H verify");
    bool ok = true;
    for (size_t i = 0; i < msg_size; i++) {
      uint8_t expected = (uint8_t)((src + i) & 0xFF);
      if (result[i] != expected) {
        fprintf(stderr, "[rank %d] from rank %d (%s) MISMATCH at byte %zu: "
                "got 0x%02x want 0x%02x\n",
                rank, src, same_node ? "NVLink" : "CXI", i,
                result[i], expected);
        ok = false;
        errors++;
        break;
      }
    }
    if (ok) {
      if (same_node) intra_ok++;
      else inter_ok++;
    }
  }
  fprintf(stderr, "[rank %d] verified: %d NVLink OK, %d CXI OK, %d errors\n",
          rank, intra_ok, inter_ok, errors);

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) fprintf(stderr, "=== test done ===\n");

  uccl_comm_destroy(comm);
  cudaFree(gpu_buf);
  MPI_Finalize();
  return 0;
}
