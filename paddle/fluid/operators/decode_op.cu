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
#include "paddle/fluid/platform/dynload/nvjpeg.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"
#include "paddle/fluid/operators/nvjpeg_decoder.h"

namespace paddle {
namespace operators {

static std::vector<cudaStream_t> nvjpeg_streams;
static nvjpegHandle_t batch_nvjpeg_handle = nullptr;
static std::unique_ptr<::ThreadPool> pool_;

static NvjpegDecoder* nvjpeg_decoder = nullptr;

void batch_InitNvjpegImage(nvjpegImage_t* img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img->channel[c] = nullptr;
    img->pitch[c] = 0;
  }
}

template <typename T>
class GPUBatchDecodeJpegKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    LOG(ERROR) << "GPUBatchDecodeJpegKernel Compute start";
    int num_threads_ = ctx.Attr<int>("num_threads");
    auto mode = ctx.Attr<std::string>("mode");
    
    // multi-phrase decode
    if (!nvjpeg_decoder) {
      nvjpeg_decoder = new NvjpegDecoder(mode);
    }

    const framework::LoDTensorArray* inputs =
        ctx.Input<framework::LoDTensorArray>("X");

    auto* out = ctx.OutputVar("Out");
    auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    out_array.resize(inputs->size());

    for (size_t i = 0; i < inputs->size(); i++) {
      const framework::LoDTensor x = inputs->at(i);
      auto* x_data = x.data<T>();

      nvjpeg_decoder->Run(x_data, static_cast<size_t>(x.numel()),
                          &out_array[i], ctx);
    }
    // // Create nvJPEG handle
    // if (batch_nvjpeg_handle == nullptr) {
    //   nvjpegStatus_t create_status =
    //       platform::dynload::nvjpegCreateSimple(&batch_nvjpeg_handle);
    //
    //   PADDLE_ENFORCE_EQ(create_status, NVJPEG_STATUS_SUCCESS,
    //                     platform::errors::Fatal("nvjpegCreateSimple failed: ",
    //                                             create_status));
    //
    //   nvjpeg_streams.reserve(num_threads_);
    //
    //   for (int i = 0; i < num_threads_; i++) {
    //     cudaStreamCreateWithFlags(&nvjpeg_streams[i], cudaStreamNonBlocking);
    //   }
    //   pool_.reset(new ::ThreadPool(num_threads_));
    // }
    //
    // const framework::LoDTensorArray* ins =
    //     ctx.Input<framework::LoDTensorArray>("X");
    //
    // auto* out = ctx.OutputVar("Out");
    // auto& out_array = *out->GetMutable<framework::LoDTensorArray>();
    // out_array.resize(ins->size());
    //
    // std::vector<std::future<int>> tasks(ins->size());
    //
    // auto dev = ctx.GetPlace();
    // for (int i = 0; i < ins->size(); i++) {
    //   auto nvjpeg_stream = nvjpeg_streams[i % num_threads_];
    //   auto nvjpeg_handle = batch_nvjpeg_handle;
    //   tasks[i] = pool_->enqueue([this, i, ins, &out_array, mode, nvjpeg_handle,
    //                              nvjpeg_stream, dev]() -> int {
    //     nvjpegJpegState_t nvjpeg_state;
    //     nvjpegStatus_t state_status = platform::dynload::nvjpegJpegStateCreate(
    //         batch_nvjpeg_handle, &nvjpeg_state);
    //
    //     PADDLE_ENFORCE_EQ(state_status, NVJPEG_STATUS_SUCCESS,
    //                       platform::errors::Fatal(
    //                           "nvjpegJpegStateCreate failed: ", state_status));
    //     const framework::LoDTensor x = ins->at(i);
    //     // framework::LoDTensor out = out_array.at(i);
    //     int components;
    //     nvjpegChromaSubsampling_t subsampling;
    //     int widths[NVJPEG_MAX_COMPONENT];
    //     int heights[NVJPEG_MAX_COMPONENT];
    //
    //     auto* x_data = x.data<T>();
    //
    //     nvjpegStatus_t info_status = platform::dynload::nvjpegGetImageInfo(
    //         batch_nvjpeg_handle, x_data, (size_t)x.numel(), &components,
    //         &subsampling, widths, heights);
    //
    //     PADDLE_ENFORCE_EQ(info_status, NVJPEG_STATUS_SUCCESS,
    //                       platform::errors::Fatal("nvjpegGetImageInfo failed: ",
    //                                               info_status));
    //
    //     int width = widths[0];
    //     int height = heights[0];
    //
    //     nvjpegOutputFormat_t output_format;
    //     int output_components;
    //
    //     if (mode == "unchanged") {
    //       if (components == 1) {
    //         output_format = NVJPEG_OUTPUT_Y;
    //         output_components = 1;
    //       } else if (components == 3) {
    //         output_format = NVJPEG_OUTPUT_RGB;
    //         output_components = 3;
    //       } else {
    //         platform::dynload::nvjpegJpegStateDestroy(nvjpeg_state);
    //         PADDLE_THROW(platform::errors::Fatal(
    //             "The provided mode is not supported for JPEG files on GPU"));
    //       }
    //     } else if (mode == "gray") {
    //       output_format = NVJPEG_OUTPUT_Y;
    //       output_components = 1;
    //     } else if (mode == "rgb") {
    //       output_format = NVJPEG_OUTPUT_RGB;
    //       output_components = 3;
    //     } else {
    //       platform::dynload::nvjpegJpegStateDestroy(nvjpeg_state);
    //       PADDLE_THROW(platform::errors::Fatal(
    //           "The provided mode is not supported for JPEG files on GPU"));
    //     }
    //
    //     nvjpegImage_t out_image;
    //     batch_InitNvjpegImage(&out_image);
    //
    //     // create nvjpeg stream
    //     // if (batch_nvjpeg_stream == nullptr) {
    //     //   cudaStreamCreateWithFlags(&batch_nvjpeg_stream,
    //     //   cudaStreamNonBlocking);
    //     // }
    //
    //     int sz = widths[0] * heights[0];
    //
    //     // auto* out = ctx.Output<framework::LoDTensor>("Out");
    //     std::vector<int64_t> out_shape = {output_components, height, width};
    //     out_array.at(i).Resize(framework::make_ddim(out_shape));
    //
    //     uint8_t* data = out_array.at(i).mutable_data<uint8_t>(dev);
    //   // transfer and decode to device buffer
    //
    //     for (int c = 0; c < output_components; c++) {
    //       out_image.channel[c] = data + c * sz;
    //       out_image.pitch[c] = width;
    //     }
    //
    //     nvjpegStatus_t decode_status = platform::dynload::nvjpegDecode(
    //         batch_nvjpeg_handle, nvjpeg_state, x_data, x.numel(), output_format,
    //         &out_image, nvjpeg_stream);
    //     // std:: cout << "task read ok: " << i << std:: endl;
    //     return 0;
    //   });
    // }
    //
    // for (size_t i = 0; i < tasks.size(); ++i) {
    //   tasks[i].wait();
    // }
    LOG(ERROR) << "GPUBatchDecodeJpegKernel Compute finish";
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(decode, ops::GPUBatchDecodeJpegKernel<uint8_t>)

#endif
