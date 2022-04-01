// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#if !defined(WITH_NV_JETSON) && !defined(PADDLE_WITH_HIP)

#include "paddle/fluid/operators/data/batch_decode_random_crop_op.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace data {

using LoDTensorBlockingQueueHolder = operators::reader::LoDTensorBlockingQueueHolder;

template <typename T>
class GPUBatchDecodeKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int num_threads = ctx.Attr<int>("num_threads");
    auto local_rank = ctx.Attr<int>("local_rank");
    auto program_id = ctx.Attr<int64_t>("program_id");
    auto host_memory_padding = ctx.Attr<int64_t>("host_memory_padding");
    auto device_memory_padding = ctx.Attr<int64_t>("device_memory_padding");

    // multi-phrase decode thread pool
    auto* decode_pool = 
      ImageDecoderThreadPoolManager::Instance()->GetDecoderThreadPool(
                          program_id, num_threads, local_rank,
                          static_cast<size_t>(host_memory_padding),
                          static_cast<size_t>(device_memory_padding));
    
    const framework::LoDTensorArray* inputs =
        ctx.Input<framework::LoDTensorArray>("X");

    auto* out = ctx.OutputVar("Out");
    auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    out_array.resize(inputs->size());

    for (size_t i = 0; i < inputs->size(); i++) {
      const framework::LoDTensor x = inputs->at(i);
      auto* x_data = x.data<T>();
      size_t x_numel = static_cast<size_t>(x.numel());

      ImageDecodeTask task = {
        .bit_stream = x_data,
        .bit_len = x_numel,
        .tensor = &out_array[i],
        .roi_generator = nullptr,
        .place = ctx.GetPlace()
      };
      decode_pool->AddTask(std::make_shared<ImageDecodeTask>(task));
    }

    decode_pool->RunAll(true);
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(batch_decode, ops::data::GPUBatchDecodeKernel<uint8_t>)

#endif
