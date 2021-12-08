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

#include <ThreadPool.h>
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/operators/data/nvjpeg_decoder.h"
#include "paddle/fluid/operators/reader/lod_tensor_blocking_queue.h"

namespace paddle {
namespace operators {
namespace data {

using LoDTensorBlockingQueueHolder = operators::reader::LoDTensorBlockingQueueHolder;

static NvjpegDecoderThreadPool* decode_pool = nullptr;

template <typename T>
class GPUBatchDecodeJpegKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int num_threads = ctx.Attr<int>("num_threads");
    LOG(ERROR) << "GPUBatchDecodeJpegKernel Compute start, num_threads: " << num_threads;
    auto mode = ctx.Attr<std::string>("mode");
    
    // multi-phrase decode thread pool
    if (!decode_pool) {
      LOG(ERROR) << "GPUBatchDecodeJpegKernel decode_pool init";
      decode_pool = new NvjpegDecoderThreadPool(num_threads, mode);
    }

    const framework::LoDTensorArray* inputs =
        ctx.Input<framework::LoDTensorArray>("X");

    auto* out = ctx.OutputVar("Out");
    auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    out_array.resize(inputs->size());

    // auto* in_var = ctx.InputVar("X");
    // auto in_queue = in_var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
    //
    // auto* out_var = ctx.OutputVar("Out");
    // auto out_queue = out_var->Get<LoDTensorBlockingQueueHolder>().GetQueue();
    // if (out_queue == nullptr) {
    //   LOG(ERROR) << "decode init output queue";
    //   auto* holder = out_var->template GetMutable<LoDTensorBlockingQueueHolder>();
    //   holder->InitOnce(2);
    //   out_queue = holder->GetQueue();
    // }
    //
    // bool success = true;
    // auto inputs = in_queue->Pop(&success);
    // PADDLE_ENFORCE_EQ(success, true, 
    //     platform::errors::PreconditionNotMet("Read from input queue failed"));
    // framework::LoDTensorArray out_array;
    // out_array.resize(inputs.size());

    for (size_t i = 0; i < inputs->size(); i++) {
      const framework::LoDTensor x = inputs->at(i);
      auto* x_data = x.data<T>();
      size_t x_numel = static_cast<size_t>(x.numel());

      NvjpegDecodeTask task = {
        .bit_stream = x_data,
        .bit_len = x_numel,
        .tensor = &out_array[i],
        .place = ctx.GetPlace()
      };
      decode_pool->AddTask(std::make_shared<NvjpegDecodeTask>(task));
    }

    decode_pool->RunAll(true);
    // out_queue->Push(out_array);

    // // multi-phrase decode single thread
    // if (!nvjpeg_decoder) {
    //   nvjpeg_decoder = new NvjpegDecoder(mode);
    // }
    //
    // const framework::LoDTensorArray* inputs =
    //     ctx.Input<framework::LoDTensorArray>("X");
    //
    // auto* out = ctx.OutputVar("Out");
    // auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    // out_array.resize(inputs->size());
    //
    // for (size_t i = 0; i < inputs->size(); i++) {
    //   const framework::LoDTensor x = inputs->at(i);
    //   auto* x_data = x.data<T>();
    //
    //   nvjpeg_decoder->Run(x_data, static_cast<size_t>(x.numel()),
    //                       &out_array[i], &ctx);
    // }

    LOG(ERROR) << "GPUBatchDecodeJpegKernel Compute finish";
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(batch_decode, ops::data::GPUBatchDecodeJpegKernel<uint8_t>)

#endif
