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

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/dynload/nvjpeg.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

namespace paddle {
namespace operators {

static cudaStream_t nvjpeg_stream = nullptr;
static nvjpegHandle_t nvjpeg_handle = nullptr;

void InitNvjpegImage(nvjpegImage_t* img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img->channel[c] = nullptr;
    img->pitch[c] = 0;
  }
}

template <typename T>
class GPUDecodeJpegKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Create nvJPEG handle
    if (nvjpeg_handle == nullptr) {
      nvjpegStatus_t create_status =
          platform::dynload::nvjpegCreateSimple(&nvjpeg_handle);

      PADDLE_ENFORCE_EQ(create_status, NVJPEG_STATUS_SUCCESS,
                        platform::errors::Fatal("nvjpegCreateSimple failed: ",
                                                create_status));
    }

    nvjpegJpegState_t nvjpeg_state;
    nvjpegStatus_t state_status =
        platform::dynload::nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

    PADDLE_ENFORCE_EQ(state_status, NVJPEG_STATUS_SUCCESS,
                      platform::errors::Fatal("nvjpegJpegStateCreate failed: ",
                                              state_status));

    int components;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT];
    int heights[NVJPEG_MAX_COMPONENT];

    auto* x = ctx.Input<framework::Tensor>("X");
    auto* x_data = x->data<T>();

    nvjpegStatus_t info_status = platform::dynload::nvjpegGetImageInfo(
        nvjpeg_handle, x_data, (size_t)x->numel(), &components, &subsampling,
        widths, heights);

    PADDLE_ENFORCE_EQ(
        info_status, NVJPEG_STATUS_SUCCESS,
        platform::errors::Fatal("nvjpegGetImageInfo failed: ", info_status));

    int width = widths[0];
    int height = heights[0];

    nvjpegOutputFormat_t output_format;
    int output_components;

    auto mode = ctx.Attr<std::string>("mode");
    if (mode == "unchanged") {
      if (components == 1) {
        output_format = NVJPEG_OUTPUT_Y;
        output_components = 1;
      } else if (components == 3) {
        output_format = NVJPEG_OUTPUT_RGB;
        output_components = 3;
      } else {
        platform::dynload::nvjpegJpegStateDestroy(nvjpeg_state);
        PADDLE_THROW(platform::errors::Fatal(
            "The provided mode is not supported for JPEG files on GPU"));
      }
    } else if (mode == "gray") {
      output_format = NVJPEG_OUTPUT_Y;
      output_components = 1;
    } else if (mode == "rgb") {
      output_format = NVJPEG_OUTPUT_RGB;
      output_components = 3;
    } else {
      platform::dynload::nvjpegJpegStateDestroy(nvjpeg_state);
      PADDLE_THROW(platform::errors::Fatal(
          "The provided mode is not supported for JPEG files on GPU"));
    }

    nvjpegImage_t out_image;
    InitNvjpegImage(&out_image);

    // create nvjpeg stream
    if (nvjpeg_stream == nullptr) {
      cudaStreamCreateWithFlags(&nvjpeg_stream, cudaStreamNonBlocking);
    }

    int sz = widths[0] * heights[0];

    auto* out = ctx.Output<framework::LoDTensor>("Out");
    std::vector<int64_t> out_shape = {output_components, height, width};
    out->Resize(phi::make_ddim(out_shape));

    T* data = out->mutable_data<T>(ctx.GetPlace());

    for (int c = 0; c < output_components; c++) {
      out_image.channel[c] = data + c * sz;
      out_image.pitch[c] = width;
    }

    nvjpegStatus_t decode_status = platform::dynload::nvjpegDecode(
        nvjpeg_handle, nvjpeg_state, x_data, x->numel(), output_format,
        &out_image, nvjpeg_stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(decode_jpeg, ops::GPUDecodeJpegKernel<uint8_t>)

#endif
