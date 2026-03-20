// Multi-rank all-to-all test following the EXACT pattern of the working
// test_fabric_p2p: direct fabric_init, MPI for bootstrap, separate send/recv
// buffers, MPI_Barrier for synchronization, byte-level verification.
//
// Usage: mpiexec -n N --ppn P ./test_fabric_alltoall_v2 [msg_size]

#include "fabric.hpp"
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

  size_t msg_size = (argc > 1) ? (size_t)atol(argv[1]) : 4096;
  int num_iters = 50;
  int warmup = 5;

  // GPU setup: each rank on a node uses a different GPU.
  int local_rank = 0;
  {
    const char* lr = getenv("PALS_LOCAL_RANKID");
    if (!lr) lr = getenv("PMI_LOCAL_RANK");
    if (lr) local_rank = atoi(lr);
  }
  int num_gpus_avail;
  check_cuda(cudaGetDeviceCount(&num_gpus_avail), "cudaGetDeviceCount");
  int gpu_idx = local_rank % num_gpus_avail;
  check_cuda(cudaSetDevice(gpu_idx), "cudaSetDevice");

  // Separate send and receive buffers (never overlap).
  // sendbuf: msg_size bytes of this rank's data
  // recvbuf: nranks * msg_size bytes to receive from all ranks
  size_t sendbuf_size = msg_size;
  size_t recvbuf_size = msg_size * nranks;

  void* sendbuf = nullptr;
  void* recvbuf = nullptr;
  check_cuda(cudaMalloc(&sendbuf, sendbuf_size), "cudaMalloc send");
  check_cuda(cudaMalloc(&recvbuf, recvbuf_size), "cudaMalloc recv");
  check_cuda(cudaMemset(recvbuf, 0, recvbuf_size), "memset recv");

  // Fill sendbuf with rank-specific pattern (same as P2P test).
  std::vector<uint8_t> pattern(msg_size);
  for (size_t i = 0; i < msg_size; i++) {
    pattern[i] = (uint8_t)((rank + i) & 0xFF);
  }
  check_cuda(cudaMemcpy(sendbuf, pattern.data(), msg_size,
                        cudaMemcpyHostToDevice), "H2D send");

  // Atomic buffer (same as P2P test).
  void* atomic_buf = nullptr;
  posix_memalign(&atomic_buf, 64, kAtomicBufferSize);
  memset(atomic_buf, 0, kAtomicBufferSize);

  // Print hardware topology for every rank.
  {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, gpu_idx);
    char hostname[256];
    gethostname(hostname, sizeof(hostname));

    // Find which CXI NIC this GPU should use.
    // On Polaris: 4 GPUs, potentially 2 CXI NICs.
    int num_cxi = 0;
    for (int i = 0; i < 4; i++) {
      char path[256];
      snprintf(path, sizeof(path), "/sys/class/cxi/cxi%d", i);
      FILE* f = fopen(path, "r");
      if (!f) { f = fopen((std::string(path) + "/device/numa_node").c_str(), "r"); }
      if (f) { num_cxi++; fclose(f); }
    }

    fprintf(stderr,
            "[rank %d] host=%s gpu=%d (%s, PCI bus %02x:%02x) "
            "num_cxi_nics=%d\n",
            rank, hostname, gpu_idx, prop.name,
            prop.pciBusID, prop.pciDeviceID, num_cxi);
  }

  if (rank == 0) {
    fprintf(stderr,
            "=== UCCL alltoall v2: %d ranks, msg=%zu ===\n", nranks, msg_size);
  }

  // --- 1. fabric_init with recvbuf as the registered buffer ---
  FabricCtx ctx{};
  fabric_init(ctx, recvbuf, recvbuf_size, atomic_buf, kAtomicBufferSize,
              gpu_idx, rank, 0, 0);

  // --- 2. Exchange connection info via MPI (same as P2P test) ---
  FabricConnectionInfo local_info{};
  fabric_get_local_info(ctx, recvbuf, recvbuf_size, atomic_buf,
                        kAtomicBufferSize, local_info);

  std::vector<FabricConnectionInfo> all_infos(nranks);
  MPI_Allgather(&local_info, sizeof(FabricConnectionInfo), MPI_BYTE,
                all_infos.data(), sizeof(FabricConnectionInfo), MPI_BYTE,
                MPI_COMM_WORLD);

  for (int r = 0; r < nranks; r++) {
    if (r == rank) continue;
    fabric_insert_peer(ctx, r, all_infos[r]);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) fprintf(stderr, "All peers connected.\n");

  // --- 3. All-to-all with bandwidth measurement ---
  // sendbuf is NOT registered as the remote-writable MR. We need to use
  // recvbuf's desc for the local side too, OR register sendbuf separately.
  // Simplest: copy sendbuf content into recvbuf[rank*msg_size] (our own slot)
  // and use recvbuf as both local source and remote destination.
  check_cuda(cudaMemcpy((char*)recvbuf + rank * msg_size, sendbuf, msg_size,
                        cudaMemcpyDeviceToDevice), "copy send to own slot");

  for (int iter = 0; iter < warmup + num_iters; iter++) {
    MPI_Barrier(MPI_COMM_WORLD);

    auto t0 = std::chrono::high_resolution_clock::now();

    // Post all writes (fast fi_write, no delivery-complete).
    // TX completion = NIC accepted the write. CXI reliable transport
    // guarantees eventual delivery. MPI_Barrier afterward provides
    // the synchronization point.
    for (int dst = 0; dst < nranks; dst++) {
      if (dst == rank) continue;
      void* local_ptr = (char*)recvbuf + rank * msg_size;
      size_t remote_offset = rank * msg_size;
      int channel = dst % kChannelPerProxy;

      int ret = fabric_write_with_data(ctx, channel, local_ptr, msg_size,
                                       dst, remote_offset, 0,
                                       (void*)(uint64_t)(dst + 1), true);
      if (ret) {
        fprintf(stderr, "[rank %d] fi_write to %d failed: %d\n",
                rank, dst, ret);
        MPI_Abort(MPI_COMM_WORLD, 1);
      }
    }

    // Wait for TX completions then barrier. That's it.
    // aws-ofi-nccl sender: fi_write + TX completion = done.
    // fi_read flush is receiver-side, not needed here.
    std::unordered_set<uint64_t> acked;
    while ((int)acked.size() < nranks - 1) {
      fabric_poll_tx(ctx, acked);
    }
    MPI_Barrier(MPI_COMM_WORLD);

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

  // --- 4. Verify ALL pairs ---
  int errors = 0;
  for (int src = 0; src < nranks; src++) {
    if (src == rank) continue;
    std::vector<uint8_t> result(msg_size);
    check_cuda(cudaMemcpy(result.data(), (char*)recvbuf + src * msg_size,
                          msg_size, cudaMemcpyDeviceToHost), "D2H verify");
    for (size_t i = 0; i < msg_size; i++) {
      uint8_t expected = (uint8_t)((src + i) & 0xFF);
      if (result[i] != expected) {
        fprintf(stderr, "[rank %d] from rank %d MISMATCH at %zu: "
                "got 0x%02x want 0x%02x\n",
                rank, src, i, result[i], expected);
        errors++;
        break;
      }
    }
  }
  if (errors == 0) {
    fprintf(stderr, "[rank %d] ALL %d sources verified OK\n",
            rank, nranks - 1);
  } else {
    fprintf(stderr, "[rank %d] FAILED: %d sources bad\n", rank, errors);
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (rank == 0) fprintf(stderr, "=== test_fabric_alltoall_v2 done ===\n");

  fabric_destroy(ctx);
  cudaFree(sendbuf);
  cudaFree(recvbuf);
  free(atomic_buf);
  MPI_Finalize();
  return 0;
}
