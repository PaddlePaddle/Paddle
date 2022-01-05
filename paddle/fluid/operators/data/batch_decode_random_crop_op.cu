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

static NvjpegDecoderThreadPool* decode_pool = nullptr;
// std::seed_seq* rand_seq = nullptr;

template <typename T>
class GPUBatchDecodeRandomCropKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    int num_threads = ctx.Attr<int>("num_threads");
    LOG(ERROR) << "GPUBatchDecodeJpegKernel Compute start, num_threads: " << num_threads;
    auto mode = ctx.Attr<std::string>("mode");
    auto local_rank = ctx.Attr<int>("local_rank");
    // multi-phrase decode thread pool
    if (!decode_pool) {
      LOG(ERROR) << "GPUBatchDecodeJpegKernel decode_pool init";
      decode_pool = new NvjpegDecoderThreadPool(num_threads, mode, local_rank);
      // rand_seq = new std::seed_seq(static_cast<int>(time(0)));
    }

    const framework::LoDTensorArray* inputs =
        ctx.Input<framework::LoDTensorArray>("X");

    auto* out = ctx.OutputVar("Out");
    auto dev = platform::CUDAPlace(local_rank);
    
    auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    out_array.resize(inputs->size());

    auto aspect_ratio_min = ctx.Attr<float>("aspect_ratio_min");
    auto aspect_ratio_max = ctx.Attr<float>("aspect_ratio_max");
    AspectRatioRange aspect_ratio_range{aspect_ratio_min, aspect_ratio_max};

    auto area_min = ctx.Attr<float>("area_min");
    auto area_max = ctx.Attr<float>("area_max");
    AreaRange area_range{area_min, area_max};

    std::seed_seq rand_seq{static_cast<int64_t>(time(0))};
    std::vector<int> rands(inputs->size());
    rand_seq.generate(rands.begin(), rands.end());

    for (size_t i = 0; i < inputs->size(); i++) {
      const framework::LoDTensor x = inputs->at(i);
      auto* x_data = x.data<T>();
      size_t x_numel = static_cast<size_t>(x.numel());

      NvjpegDecodeTask task = {
        .bit_stream = x_data,
        .bit_len = x_numel,
        .tensor = &out_array[i],
        .roi_generator = new RandomROIGenerator(
                                aspect_ratio_range, area_range, rands[i]),
        .place = dev
        // .place = ctx.GetPlace()
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
REGISTER_OP_CUDA_KERNEL(batch_decode_random_crop, ops::data::GPUBatchDecodeRandomCropKernel<uint8_t>)

#endif
