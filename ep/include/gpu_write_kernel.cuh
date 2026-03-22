#ifndef GPU_WRITE_KERNEL_CUH
#define GPU_WRITE_KERNEL_CUH

#include "d2h_queue_device.cuh"
#include <cuda_runtime.h>
#include <stdint.h>

// Push |num_writes| WRITE commands into a single D2H queue and wait for
// the CPU proxy to consume all of them.
// Launch with <<<1, threads>>>.
cudaError_t launch_gpu_push_write(d2hq::D2HHandle handle,
                                  uint8_t dst_rank,
                                  uint32_t lptr_shifted,
                                  uint32_t rptr_shifted,
                                  uint32_t bytes,
                                  int num_writes,
                                  int threads,
                                  cudaStream_t stream);

// GPU-initiated all-to-all: push WRITE commands for all inter-node
// destinations, distributed across |num_handles| D2H queues.
// |handles| must be a device-accessible array of D2HHandle.
// |remote_mask| must be a device-accessible array: remote_mask[r] = 1
// if rank r is on a different node.
// Launch with <<<num_handles, threads>>>.
cudaError_t launch_gpu_alltoall_write(d2hq::D2HHandle* handles,
                                      int num_handles,
                                      int my_rank,
                                      int nranks,
                                      uint32_t msg_size,
                                      int addr_shift,
                                      int* remote_mask,
                                      cudaStream_t stream);

#endif  // GPU_WRITE_KERNEL_CUH
