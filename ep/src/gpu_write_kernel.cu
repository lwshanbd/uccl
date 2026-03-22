#include "gpu_write_kernel.cuh"
#include "common.hpp"
#include "d2h_queue_device.cuh"
#include <stdio.h>

#if defined(__HIP_DEVICE_COMPILE__)
#include "amd_nanosleep.cuh"
#endif

// ---------------------------------------------------------------------------
// Single-queue P2P write kernel
// ---------------------------------------------------------------------------

__global__ void gpu_push_write_kernel(d2hq::D2HHandle handle,
                                      uint8_t dst_rank,
                                      uint32_t lptr_shifted,
                                      uint32_t rptr_shifted,
                                      uint32_t bytes,
                                      int num_writes) {
  int tid = threadIdx.x;
  int nthreads = blockDim.x;

  for (int i = tid; i < num_writes; i += nthreads) {
    TransferCmd cmd{};
    cmd.cmd_type = CmdType::WRITE;
    cmd.dst_rank = dst_rank;
    cmd.bytes = bytes;
    cmd.req_lptr = lptr_shifted;
    cmd.req_rptr = rptr_shifted;
    cmd.expert_idx = 0;

    uint64_t slot = 0;
    handle.atomic_set_and_commit(cmd, &slot);
    if (tid == 0 && i == 0) {
      printf("[GPU] Pushed write %d to slot %llu\n",
             i, (unsigned long long)slot);
    }
  }

  __syncthreads();

  // Thread 0 waits for proxy to consume all pushed commands.
  // With FIFO backend, use fifo.sync() to wait for a given head value.
  if (tid == 0) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
#ifdef MSCCLPP_DEVICE_COMPILE
    // The last push returned a slot number. sync() waits for the proxy
    // to pop past that slot (tail >= slot).
    printf("[GPU] Waiting for FIFO sync...\n");
    handle.fifo.sync(num_writes - 1, /*maxSpinCount=*/-1);
    printf("[GPU] FIFO sync done — all %d writes consumed\n", num_writes);
#endif
#else
    uint64_t target = handle.ring->head;
    printf("[GPU] Waiting for ring tail to reach %llu...\n",
           (unsigned long long)target);
    while (handle.ring->volatile_tail() < target) {
      __nanosleep(64);
    }
    printf("[GPU] Ring buffer: all %d writes completed\n", num_writes);
#endif
  }
}

cudaError_t launch_gpu_push_write(d2hq::D2HHandle handle,
                                  uint8_t dst_rank,
                                  uint32_t lptr_shifted,
                                  uint32_t rptr_shifted,
                                  uint32_t bytes,
                                  int num_writes,
                                  int threads,
                                  cudaStream_t stream) {
  gpu_push_write_kernel<<<1, threads, 0, stream>>>(
      handle, dst_rank, lptr_shifted, rptr_shifted, bytes, num_writes);
  return cudaGetLastError();
}

// ---------------------------------------------------------------------------
// Multi-queue all-to-all kernel
// ---------------------------------------------------------------------------

__global__ void gpu_alltoall_write_kernel(d2hq::D2HHandle* handles,
                                          int num_handles,
                                          int my_rank,
                                          int nranks,
                                          uint32_t msg_size,
                                          int addr_shift,
                                          int* remote_mask) {
  int handle_idx = blockIdx.x;
  if (handle_idx >= num_handles) return;

  d2hq::D2HHandle& h = handles[handle_idx];
  int tid = threadIdx.x;

  // Thread 0 pushes all writes that map to this handle.
  if (tid == 0) {
    for (int dst = 0; dst < nranks; dst++) {
      if (dst == my_rank) continue;
      if (!remote_mask[dst]) continue;
      if (dst % num_handles != handle_idx) continue;

      TransferCmd cmd{};
      cmd.cmd_type = CmdType::WRITE;
      cmd.dst_rank = (uint8_t)dst;
      cmd.bytes = msg_size;
      cmd.req_lptr =
          (uint32_t)((uint64_t)(my_rank * msg_size) >> addr_shift);
      cmd.req_rptr =
          (uint32_t)((uint64_t)(my_rank * msg_size) >> addr_shift);
      cmd.expert_idx = 0;
      h.atomic_set_and_commit(cmd);
    }
  }

  __syncthreads();

  // Wait for all writes to be consumed.
  if (tid == 0) {
#ifdef USE_MSCCLPP_FIFO_BACKEND
#ifdef MSCCLPP_DEVICE_COMPILE
    // Sync on a generous count — the proxy will catch up.
    h.fifo.sync(0, /*maxSpinCount=*/-1);
#endif
#else
    uint64_t target = h.ring->head;
    while (h.ring->volatile_tail() < target) {
      __nanosleep(64);
    }
#endif
  }
}

cudaError_t launch_gpu_alltoall_write(d2hq::D2HHandle* handles,
                                      int num_handles,
                                      int my_rank,
                                      int nranks,
                                      uint32_t msg_size,
                                      int addr_shift,
                                      int* remote_mask,
                                      cudaStream_t stream) {
  gpu_alltoall_write_kernel<<<num_handles, 32, 0, stream>>>(
      handles, num_handles, my_rank, nranks, msg_size, addr_shift,
      remote_mask);
  return cudaGetLastError();
}
