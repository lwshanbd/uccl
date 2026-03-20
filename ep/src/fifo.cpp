// Copyright (c) Microsoft Corporation.
// Licensed under the MIT license.

#include "fifo.hpp"
#include "fifo_util.hpp"
#include <cerrno>
#include <cstdio>
#include <cstring>
#if __has_include(<numaif.h>)
#include <numaif.h>
#define HAS_NUMAIF 1
#else
#define HAS_NUMAIF 0
#endif

namespace mscclpp {

struct Fifo::Impl {
  detail::UniqueGpuHostPtr<ProxyTrigger> triggers;
  detail::UniqueGpuPtr<uint64_t> head;
  detail::UniqueGpuHostPtr<uint64_t> tail;
  detail::UniqueGpuPtr<uint64_t> tailCache;
  int const size;

  Impl(int size)
      : triggers(detail::gpuCallocHostUnique<ProxyTrigger>(size)),
        head(detail::gpuCallocUnique<uint64_t>()),
        tail(detail::gpuCallocHostUnique<uint64_t>()),
        tailCache(detail::gpuCallocUnique<uint64_t>()),
        size(size) {}
};

Fifo::Fifo(int size) {
  int device;
  MSCCLPP_CUDATHROW(cudaGetDevice(&device));
  int numaNode = getDeviceNumaNode(device);
  if (numaNode >= 0) {
#if HAS_NUMAIF
    unsigned long nodemask = 1UL << numaNode;
    if (set_mempolicy(MPOL_PREFERRED, &nodemask, 8 * sizeof(nodemask)) != 0) {
      std::fprintf(stderr,
                   "Warning: set_mempolicy failed for device %d numaNode %d: "
                   "%s (errno=%d). Continuing without NUMA mempolicy.\n",
                   device, numaNode, std::strerror(errno), errno);
    }
#endif
  }
  pimpl_ = std::make_unique<Impl>(size);
}

Fifo::~Fifo() = default;

ProxyTrigger Fifo::poll() {
  ProxyTrigger trigger;
  ProxyTrigger* ptr = &pimpl_->triggers.get()[*(pimpl_->tail) % pimpl_->size];
  // we are loading fst first. if fst is non-zero then snd is also valid
  trigger.fst = atomicLoad(&(ptr->fst), memoryOrderAcquire);
  trigger.snd = ptr->snd;
  return trigger;
}

void Fifo::pop() {
  uint64_t curTail = *(pimpl_->tail);
  pimpl_->triggers.get()[curTail % pimpl_->size].fst = 0;
  atomicStore(pimpl_->tail.get(), curTail + 1, memoryOrderRelease);
}

int Fifo::size() const { return pimpl_->size; }

FifoDeviceHandle Fifo::deviceHandle() const {
  FifoDeviceHandle deviceHandle;
  deviceHandle.triggers = pimpl_->triggers.get();
  deviceHandle.head = pimpl_->head.get();
  deviceHandle.tail = pimpl_->tail.get();
  deviceHandle.tailCache = pimpl_->tailCache.get();
  deviceHandle.size = pimpl_->size;
  return deviceHandle;
}

}  // namespace mscclpp
