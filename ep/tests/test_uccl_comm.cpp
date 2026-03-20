// Test for the UCCL Communication Library API.
// Exercises the full stack: uccl_comm_init_mpi → proxy threads → fi_write.
//
// Usage: mpiexec -n N --ppn P ./test_uccl_comm [msg_size_bytes]

#include "uccl_comm.h"
#include <mpi.h>
#include <cuda_runtime.h>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <signal.h>
#include <execinfo.h>
#include <unistd.h>

static void crash_handler(int sig) {
  fprintf(stderr, "\n=== CRASH: signal %d ===\n", sig);
  void* bt[64];
  int n = backtrace(bt, 64);
  backtrace_symbols_fd(bt, n, STDERR_FILENO);
  fflush(stderr);
  _exit(128 + sig);
}

static void install_crash_handler() {
  signal(SIGSEGV, crash_handler);
  signal(SIGABRT, crash_handler);
  signal(SIGBUS, crash_handler);
}

static void check_cuda(cudaError_t err, const char* msg) {
  if (err != cudaSuccess) {
    fprintf(stderr, "[CUDA] %s: %s\n", msg, cudaGetErrorString(err));
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
}

int main(int argc, char** argv) {
  install_crash_handler();
  MPI_Init(&argc, &argv);
  int rank, nranks;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &nranks);

  size_t msg_size = (argc > 1) ? (size_t)atol(argv[1]) : 1024 * 1024;
  int num_iters = 50;
  int warmup = 5;

  // GPU setup: use PALS_LOCAL_RANKID (same as uccl_comm_init_mpi).
  int local_rank = 0;
  {
    char const* lr = getenv("PALS_LOCAL_RANKID");
    if (!lr) lr = getenv("PMI_LOCAL_RANK");
    if (lr) local_rank = atoi(lr);
  }
  int num_gpus;
  check_cuda(cudaGetDeviceCount(&num_gpus), "cudaGetDeviceCount");
  int gpu_idx = local_rank % num_gpus;
  check_cuda(cudaSetDevice(gpu_idx), "cudaSetDevice");

  // Allocate separate send and receive GPU buffers.
  // sendbuf: msg_size (this rank's data to send to everyone)
  // recvbuf: nranks * msg_size (receives from all ranks)
  size_t send_size = msg_size;
  size_t recv_size = msg_size * nranks;
  // Total registered buffer: send region at [0, msg_size), recv at [msg_size, msg_size + recv_size)
  size_t buf_size = send_size + recv_size;

  void* gpu_buf = nullptr;
  check_cuda(cudaMalloc(&gpu_buf, buf_size), "cudaMalloc");
  check_cuda(cudaMemset(gpu_buf, 0, buf_size), "cudaMemset");

  void* sendbuf = gpu_buf;
  void* recvbuf = (char*)gpu_buf + send_size;

  // Fill send region with a rank-specific pattern.
  std::vector<uint8_t> pattern(msg_size);
  for (size_t i = 0; i < msg_size; i++) {
    pattern[i] = (uint8_t)((rank + i) & 0xFF);
  }
  check_cuda(cudaMemcpy(gpu_buf, pattern.data(), msg_size,
                        cudaMemcpyHostToDevice),
             "cudaMemcpy H2D");

  if (rank == 0) {
    fprintf(stderr,
            "=== UCCL Comm Test: %d ranks, msg=%zu, "
            "%d proxy threads/rank, %d channels/proxy ===\n",
            nranks, msg_size,
            uccl_get_num_proxy_threads(),
            uccl_get_channels_per_proxy());
  }

  // --- 1. Initialize UCCL communicator ---
  uccl_comm_t comm;
  MPI_Comm mpi_comm = MPI_COMM_WORLD;
  int ret = uccl_comm_init_mpi(&comm, &mpi_comm, gpu_buf, buf_size);
  if (ret) {
    fprintf(stderr, "[rank %d] uccl_comm_init_mpi failed: %d\n", rank, ret);
    MPI_Abort(MPI_COMM_WORLD, 1);
  }

  // Query info.
  uccl_comm_info_t info;
  uccl_comm_get_info(comm, &info);
  fprintf(stderr, "[rank %d] Comm ready: nranks=%d, local_rank=%d, nodes=%d\n",
          info.rank, info.nranks, info.local_rank, info.num_nodes);

  // Check D2H channels are available (for GPU kernel path).
  uccl_d2h_channels_t channels;
  uccl_get_d2h_channels(comm, &channels);
  fprintf(stderr, "[rank %d] D2H channels: %d (%d proxies x %d channels)\n",
          rank, channels.count, channels.num_proxies,
          channels.channels_per_proxy);

  MPI_Barrier(MPI_COMM_WORLD);

  // --- 2. Point-to-point put test ---
  if (rank == 0) fprintf(stderr, "\n--- P2P Put Test ---\n");

  if (rank == 0 && nranks >= 2) {
    ret = uccl_put(comm, 1, gpu_buf, /*remote_offset=*/0, msg_size);
    if (ret) {
      fprintf(stderr, "[rank 0] uccl_put failed: %d\n", ret);
    }
    uccl_quiet(comm);
    fprintf(stderr, "[rank 0] P2P put to rank 1 completed\n");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  if (rank == 1) {
    std::vector<uint8_t> result(msg_size);
    check_cuda(cudaMemcpy(result.data(), gpu_buf, msg_size,
                          cudaMemcpyDeviceToHost),
               "D2H verify");
    bool ok = true;
    for (size_t i = 0; i < msg_size; i++) {
      if (result[i] != (uint8_t)((0 + i) & 0xFF)) {
        fprintf(stderr, "[rank 1] P2P data mismatch at byte %zu\n", i);
        ok = false;
        break;
      }
    }
    fprintf(stderr, "[rank 1] P2P data %s\n", ok ? "PASSED" : "FAILED");
  }
  MPI_Barrier(MPI_COMM_WORLD);

  // --- 3. All-to-all bandwidth test ---
  if (rank == 0) fprintf(stderr, "\n--- All-to-all Bandwidth ---\n");

  // Use gpu_buf[0..msg_size) as send, and gpu_buf as recv (offset by rank).
  for (int iter = 0; iter < warmup + num_iters; iter++) {
    // Reset receive region.
    check_cuda(cudaMemset((char*)gpu_buf + msg_size, 0,
                          msg_size * (nranks - 1)),
               "reset recv");
    MPI_Barrier(MPI_COMM_WORLD);

    auto t0 = std::chrono::high_resolution_clock::now();

    ret = uccl_alltoall(comm, gpu_buf, gpu_buf, msg_size);
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

  // --- 4. Verify all-to-all data: check EVERY source rank ---
  {
    int errors = 0;
    for (int src = 0; src < nranks; src++) {
      if (src == rank) continue;
      std::vector<uint8_t> result(msg_size);
      check_cuda(cudaMemcpy(result.data(),
                            (char*)gpu_buf + src * msg_size,
                            msg_size, cudaMemcpyDeviceToHost),
                 "D2H verify a2a");
      for (size_t i = 0; i < msg_size; i++) {
        uint8_t expected = (uint8_t)((src + i) & 0xFF);
        if (result[i] != expected) {
          fprintf(stderr, "[rank %d] A2A from rank %d MISMATCH at byte %zu: "
                  "got 0x%02x want 0x%02x\n",
                  rank, src, i, result[i], expected);
          errors++;
          break;
        }
      }
    }
    if (errors == 0) {
      fprintf(stderr, "[rank %d] A2A verify ALL %d sources PASSED\n",
              rank, nranks - 1);
    } else {
      fprintf(stderr, "[rank %d] A2A verify FAILED (%d sources bad)\n",
              rank, errors);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) fprintf(stderr, "\n=== test_uccl_comm PASSED ===\n");

  // --- 5. Cleanup ---
  uccl_comm_destroy(comm);
  cudaFree(gpu_buf);
  MPI_Finalize();
  return 0;
}
