// Two-rank point-to-point test for the libfabric/CXI backend.
// Each rank initializes fabric, exchanges addresses, then rank 0 writes
// to rank 1's GPU buffer and rank 1 verifies the data.
//
// Usage: mpiexec -n 2 --ppn 1 ./test_fabric_p2p

#include "fabric.hpp"
#include <arpa/inet.h>
#include <netdb.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mpi.h>
#include <cuda_runtime.h>

static void check_cuda(cudaError_t err, char const* msg) {
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

  if (nranks != 2) {
    if (rank == 0)
      fprintf(stderr, "This test requires exactly 2 ranks, got %d\n", nranks);
    MPI_Finalize();
    return 1;
  }

  // Each rank uses GPU 0 on its node.
  int gpu_idx = 0;
  check_cuda(cudaSetDevice(gpu_idx), "cudaSetDevice");

  // Allocate GPU buffer (1 MiB) and host atomic buffer.
  size_t gpu_buf_size = 1 * 1024 * 1024;
  void* gpu_buf = nullptr;
  check_cuda(cudaMalloc(&gpu_buf, gpu_buf_size), "cudaMalloc");
  check_cuda(cudaMemset(gpu_buf, 0, gpu_buf_size), "cudaMemset");

  size_t atomic_buf_size = kAtomicBufferSize;
  void* atomic_buf = nullptr;
  if (posix_memalign(&atomic_buf, 64, atomic_buf_size)) {
    fprintf(stderr, "posix_memalign failed\n");
    MPI_Abort(MPI_COMM_WORLD, 1);
  }
  memset(atomic_buf, 0, atomic_buf_size);

  fprintf(stderr, "[rank %d] GPU buf=%p, size=%zu\n", rank, gpu_buf,
          gpu_buf_size);

  // --- 1. Initialize fabric ---
  FabricCtx ctx{};
  fabric_init(ctx, gpu_buf, gpu_buf_size, atomic_buf, atomic_buf_size, gpu_idx,
              rank, /*thread_idx=*/0, /*local_rank=*/0);

  // --- 2. Exchange connection info via MPI ---
  FabricConnectionInfo local_info{};
  fabric_get_local_info(ctx, gpu_buf, gpu_buf_size, atomic_buf, atomic_buf_size,
                        local_info);

  FabricConnectionInfo remote_info{};
  MPI_Sendrecv(&local_info, sizeof(local_info), MPI_BYTE,
               1 - rank, 0,
               &remote_info, sizeof(remote_info), MPI_BYTE,
               1 - rank, 0,
               MPI_COMM_WORLD, MPI_STATUS_IGNORE);

  fabric_insert_peer(ctx, 1 - rank, remote_info);
  fprintf(stderr, "[rank %d] Peer %d inserted into AV\n", rank, 1 - rank);

  MPI_Barrier(MPI_COMM_WORLD);

  // --- 3. Rank 0 writes a pattern to rank 1's GPU buffer ---
  size_t test_size = 4096;
  if (rank == 0) {
    // Fill local GPU buffer with a known pattern.
    std::vector<uint8_t> pattern(test_size);
    for (size_t i = 0; i < test_size; i++) {
      pattern[i] = (uint8_t)(i & 0xFF);
    }
    check_cuda(cudaMemcpy(gpu_buf, pattern.data(), test_size,
                          cudaMemcpyHostToDevice),
               "cudaMemcpy H2D");

    // RMA write to rank 1's GPU buffer.
    int ret = fabric_write_with_data(ctx, /*channel=*/0, gpu_buf, test_size,
                                     /*dst_rank=*/1, /*remote_offset=*/0,
                                     /*cq_data=*/0, /*context=*/(void*)42,
                                     /*signaled=*/true);
    if (ret) {
      fprintf(stderr, "[rank 0] fi_write failed: %d (%s)\n", ret,
              fi_strerror(-ret));
      MPI_Abort(MPI_COMM_WORLD, 1);
    }

    // Poll TX CQ until local completion.
    std::unordered_set<uint64_t> acked;
    int polls = 0;
    while (acked.empty() && polls < 1000000) {
      fabric_poll_tx(ctx, acked);
      polls++;
    }
    if (acked.empty()) {
      fprintf(stderr, "[rank 0] TX completion not received after %d polls\n",
              polls);
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
    fprintf(stderr, "[rank 0] Write completed after %d polls\n", polls);
  }

  // Barrier ensures rank 0's write is visible before rank 1 reads.
  MPI_Barrier(MPI_COMM_WORLD);

  // --- 4. Rank 1 verifies data ---
  if (rank == 1) {
    std::vector<uint8_t> result(test_size, 0);
    check_cuda(cudaMemcpy(result.data(), gpu_buf, test_size,
                          cudaMemcpyDeviceToHost),
               "cudaMemcpy D2H");

    bool ok = true;
    for (size_t i = 0; i < test_size; i++) {
      if (result[i] != (uint8_t)(i & 0xFF)) {
        fprintf(stderr, "[rank 1] ERROR: data mismatch at byte %zu: "
                "expected 0x%02x, got 0x%02x\n",
                i, (uint8_t)(i & 0xFF), result[i]);
        ok = false;
        break;
      }
    }
    if (ok) {
      fprintf(stderr, "[rank 1] Data verification PASSED (%zu bytes)\n",
              test_size);
    } else {
      MPI_Abort(MPI_COMM_WORLD, 1);
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);

  // --- 5. Cleanup ---
  fabric_destroy(ctx);
  cudaFree(gpu_buf);
  free(atomic_buf);

  if (rank == 0) {
    fprintf(stderr, "=== test_fabric_p2p PASSED ===\n");
  }
  MPI_Finalize();
  return 0;
}
