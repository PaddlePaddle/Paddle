/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <cuda.h>
#include <memory>
#include "paddle/fluid/framework/details/stream_executor_gpu.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/stream/gpu_event.h"

namespace paddle {
namespace framework {
namespace details {
class StreamExecutor;
}  // namespace framework
}
namespace platform {
namespace stream {

namespace internal {
class StreamInterface;
}  // namespace internal
   /*
    * Stream should be created and managed by executor, but now stream in
    * device_context, in order to keep so  leave this for extend.
    * Suppose we have a stream executor.
    */
class StreamExecutor;

namespace exc = paddle::framework::details;

class BaseStream {
 public:
  explicit BaseStream(exc::StreamExecutor* pe);
  virtual ~BaseStream();

  BaseStream& Init();

  // bool ok() const { return !ErrorState(); }
  // template <typename... Params, typename... Args>
  //    LaunchKernel(BlockDim block_dims, GridDim grid_dims,
  //                 const KernelParam<Params...> &kernel, Args... args);
  BaseStream& WaitForOtherStream(BaseStream* other);
  // should delete later
  // BaseStream& WaitForOtherStream(cudaStream_t other);

  BaseStream& InsertEvent(stream::Event* event);
  internal::StreamInterface* implementation() { return implementation_.get(); }

  BaseStream& Memcpy(void* host_dst, const void* gpu_src, uint64_t size);

 private:
  bool allocated_;
  bool ok_;
  std::unique_ptr<internal::StreamInterface> implementation_;
  exc::StreamExecutor* pe_;
};

}  // namespace stream
}  // namespace platform
}  // namespace paddle
