#pragma once

#include "fabric_ctx.hpp"
#include "ring_buffer.cuh"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <set>
#include <unordered_set>
#include <vector>

// ---------------------------------------------------------------------------
// Initialization / teardown
// ---------------------------------------------------------------------------

// Initialize all libfabric resources for one proxy thread: fabric, domain,
// AV, CQs, control EP, and kChannelPerProxy data EPs.  Registers |gpu_buf|
// and |atomic_buf| as MRs and binds them to the EPs.
void fabric_init(FabricCtx& ctx, void* gpu_buf, size_t gpu_buf_size,
                 void* atomic_buf, size_t atomic_buf_size,
                 int gpu_idx, int rank, int thread_idx, int local_rank);

// Release all libfabric resources.
void fabric_destroy(FabricCtx& ctx);

// ---------------------------------------------------------------------------
// Memory registration helpers
// ---------------------------------------------------------------------------

// Register GPU memory via FI_HMEM_CUDA and bind the MR to |ep|.
// Must be called before fi_enable(ep).
FabricMR fabric_reg_gpu_mr(fid_domain* domain, fid_ep* ep,
                           void* gpu_buf, size_t bytes, int gpu_idx,
                           uint64_t access);

// Register ordinary host memory and bind the MR to |ep|.
FabricMR fabric_reg_host_mr(fid_domain* domain, fid_ep* ep,
                            void* buf, size_t bytes, uint64_t access);

// ---------------------------------------------------------------------------
// Connection / address exchange
// ---------------------------------------------------------------------------

// Fill |info| with this thread's EP addresses and MR keys.
void fabric_get_local_info(FabricCtx& ctx, void* gpu_buf, size_t gpu_buf_size,
                           void* atomic_buf, size_t atomic_buf_size,
                           FabricConnectionInfo& info);

// Insert a remote peer's addresses into the AV and record its MR keys.
void fabric_insert_peer(FabricCtx& ctx, int peer_rank,
                        FabricConnectionInfo const& remote);

// ---------------------------------------------------------------------------
// Data-path operations
// ---------------------------------------------------------------------------

// Post an RMA write with CQ data (remote completion).  Returns 0 on success.
int fabric_write_with_data(FabricCtx& ctx, int channel_idx,
                           void* local_buf, size_t len,
                           int dst_rank, uintptr_t remote_offset,
                           uint64_t cq_data, void* context, bool signaled);

// Post an RMA write with FI_MORE batching.  Set |is_last| on the final
// write in a batch to trigger transmission of all prior batched writes.
int fabric_write_batch(FabricCtx& ctx, int channel_idx,
                       void* local_buf, size_t len,
                       int dst_rank, uintptr_t remote_offset,
                       void* context, bool is_last);

// Post a fetch-and-add atomic.  Returns 0 on success.
int fabric_fetch_add(FabricCtx& ctx, int dst_rank,
                     uintptr_t remote_offset, uint64_t add_value,
                     uint64_t* local_result, void* context);

// Post a send with CQ data (used for barriers).
int fabric_send_data(FabricCtx& ctx, int dst_rank,
                     uint64_t cq_data, void* context);

// Pre-post a receive buffer on the control EP.
int fabric_post_recv(FabricCtx& ctx);

// Flush all prior writes to |dst_rank| on |channel| by issuing a small
// fi_read.  RMA ordering guarantees that when the read completes, all
// prior writes to the same destination are visible at the remote.
// |local_buf| must be a registered buffer to receive the read result.
int fabric_flush(FabricCtx& ctx, int dst_rank, int channel,
                 void* local_buf, void* local_desc);

// ---------------------------------------------------------------------------
// Completion polling
// ---------------------------------------------------------------------------

// Poll the TX CQ; insert completed wr_ids into |acked_wrs|.
// Returns number of completions (>=0) or -1 on error.
int fabric_poll_tx(FabricCtx& ctx,
                   std::unordered_set<uint64_t>& acked_wrs);

// Poll the RX CQ and process remote completions (writes, atomics, barriers).
// Returns number of completions (>=0) or -1 on error.
int fabric_poll_rx(FabricCtx& ctx, fi_cq_data_entry* entries, int max_entries);
