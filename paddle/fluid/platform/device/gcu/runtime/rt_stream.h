/* Copyright (c) 2023 Enflame. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <memory>
#include <vector>

#include "paddle/fluid/platform/device/gcu/runtime/rt_context.h"
#include "paddle/fluid/platform/device/gcu/runtime/rt_executable.h"

namespace hlir {
struct Tensor;
class HlirDispatch;
}  // namespace hlir

namespace paddle {
namespace platform {
namespace gcu {
namespace runtime {

struct Event;
struct Memory;

static std::vector<int64_t> unset_flag_dims = {-100};
struct GcuMemory {
  GcuMemory(void *in_ptr, const std::vector<int64_t> &in_dims, size_t in_size) {
    mem_ptr = in_ptr;
    dims = in_dims;
    size = in_size;
  }
  void *mem_ptr = nullptr;
  std::vector<int64_t> dims = unset_flag_dims;
  size_t size = 0;
};

// no thread safety stream, please use stream as thread.
struct Stream {
  Context *ctx;

  topsStream_t tops_stream;

  bool owning_stream_ = true;

  struct GcuLaunchParams {
    GcuLaunchParams() = default;
    GcuLaunchParams(const std::vector<void *> &ins,
                    const std::vector<void *> &outs)
        : inputs(ins), outputs(outs) {}
    std::vector<void *> inputs;
    std::vector<void *> outputs;
  };

  static std::shared_ptr<Stream> CreateStream(Context *ctx,
                                              topsStream_t stream_in = nullptr);

  void MemcpyH2DSync(std::shared_ptr<Memory> pmem,
                     const void *pdata,
                     uint64_t nbytes);

  void MemcpyD2HSync(void *pdata,
                     std::shared_ptr<Memory> pmem,
                     uint64_t nbytes);

  void MemcpyD2DSync(std::shared_ptr<Memory> pdst,
                     std::shared_ptr<Memory> psrc,
                     uint64_t nbytes);

  void Memset32Sync(std::shared_ptr<Memory> pmem, int value, uint64_t nbytes);

  void MemcpyH2DAsync(std::shared_ptr<Memory> pmem,
                      const void *pdata,
                      uint64_t nbytes);

  void MemcpyD2HAsync(void *pdata,
                      std::shared_ptr<Memory> pmem,
                      uint64_t nbytes);

  void MemcpyD2DAsync(std::shared_ptr<Memory> pdst,
                      std::shared_ptr<Memory> psrc,
                      uint64_t nbytes);

  void MemsetAsync(std::shared_ptr<Memory> pmem, int value, uint64_t nbytes);

  void MemcpySync(void *dst,
                  const void *src,
                  size_t size_bytes,
                  topsMemcpyKind kind);

  void MemcpyAsync(void *dst,
                   const void *src,
                   size_t size_bytes,
                   topsMemcpyKind kind);

  void MemsetAsync(void *dst, int value, size_t size_bytes);

  void EventRecord(std::shared_ptr<Event> pevent);

  void WaitEvent(std::shared_ptr<Event> pevent);

  void AddHostCallback(const std::function<void()> &fn);

  void RunExecutableSync(const ExecutablePtr &exe,
                         const std::vector<std::shared_ptr<Memory>> &ins,
                         const std::vector<std::shared_ptr<Memory>> &outs);

  void RunExecutableAsync(const ExecutablePtr &exe,
                          const std::vector<std::shared_ptr<Memory>> &ins,
                          const std::vector<std::shared_ptr<Memory>> &outs);

  void RunExecutableAsync(const ExecutablePtr &exe,
                          const std::vector<void *> &ins,
                          const std::vector<void *> &outs,
                          const std::vector<phi::DataType> &in_types,
                          const std::vector<phi::DataType> &out_types);

  void RunExecutableAsync(const ExecutablePtr &exe,
                          const std::vector<void *> &ins,
                          const std::vector<void *> &outs);

  void RunExecutable(const ExecutablePtr &exe,
                     GcuLaunchParams &params);  // NOLINT

  void Synchronize();

  Stream(Context *ctx, topsStream_t stream, bool owning_stream = true);

  ~Stream();

  Stream() = delete;
  RT_DISALLOW_COPY_AND_ASSIGN(Stream);
};

}  // namespace runtime
}  // namespace gcu
}  // namespace platform
}  // namespace paddle
