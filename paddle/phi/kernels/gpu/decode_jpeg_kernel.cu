// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/decode_jpeg_kernel.h"

#include "paddle/phi/backends/dynload/nvjpeg.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/backends/stream.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

static cudaStream_t nvjpeg_stream = nullptr;
static nvjpegHandle_t nvjpeg_handle = nullptr;

void InitNvjpegImage(nvjpegImage_t* img) {
  for (int c = 0; c < NVJPEG_MAX_COMPONENT; c++) {
    img->channel[c] = nullptr;
    img->pitch[c] = 0;
  }
}

template <typename T, typename Context>
void DecodeJpegKernel(const Context& dev_ctx,
                      const DenseTensor& x,
                      const std::string& mode,
                      DenseTensor* out) {
  // Create nvJPEG handle
  if (nvjpeg_handle == nullptr) {
    nvjpegStatus_t create_status =
        phi::dynload::nvjpegCreateSimple(&nvjpeg_handle);

    PADDLE_ENFORCE_EQ(
        create_status,
        NVJPEG_STATUS_SUCCESS,
        errors::Fatal("nvjpegCreateSimple failed: ", create_status));
  }

  nvjpegJpegState_t nvjpeg_state;
  nvjpegStatus_t state_status =
      phi::dynload::nvjpegJpegStateCreate(nvjpeg_handle, &nvjpeg_state);

  PADDLE_ENFORCE_EQ(
      state_status,
      NVJPEG_STATUS_SUCCESS,
      errors::Fatal("nvjpegJpegStateCreate failed: ", state_status));

  int components;
  nvjpegChromaSubsampling_t subsampling;
  int widths[NVJPEG_MAX_COMPONENT];
  int heights[NVJPEG_MAX_COMPONENT];

  auto* x_data = x.data<T>();

  nvjpegStatus_t info_status =
      phi::dynload::nvjpegGetImageInfo(nvjpeg_handle,
                                       x_data,
                                       (std::size_t)x.numel(),
                                       &components,
                                       &subsampling,
                                       widths,
                                       heights);
  PADDLE_ENFORCE_EQ(info_status,
                    NVJPEG_STATUS_SUCCESS,
                    errors::Fatal("nvjpegGetImageInfo failed: ", info_status));

  int width = widths[0];
  int height = heights[0];

  nvjpegOutputFormat_t output_format;
  int output_components;

  if (mode == "unchanged") {
    if (components == 1) {
      output_format = NVJPEG_OUTPUT_Y;
      output_components = 1;
    } else if (components == 3) {
      output_format = NVJPEG_OUTPUT_RGB;
      output_components = 3;
    } else {
      phi::dynload::nvjpegJpegStateDestroy(nvjpeg_state);
      PADDLE_THROW(errors::Fatal(
          "The provided mode is not supported for JPEG files on GPU"));
    }
  } else if (mode == "gray") {
    output_format = NVJPEG_OUTPUT_Y;
    output_components = 1;
  } else if (mode == "rgb") {
    output_format = NVJPEG_OUTPUT_RGB;
    output_components = 3;
  } else {
    phi::dynload::nvjpegJpegStateDestroy(nvjpeg_state);
    PADDLE_THROW(errors::Fatal(
        "The provided mode is not supported for JPEG files on GPU"));
  }

  nvjpegImage_t out_image;
  InitNvjpegImage(&out_image);

  // create nvjpeg stream
  if (nvjpeg_stream == nullptr) {
    cudaStreamCreateWithFlags(&nvjpeg_stream, cudaStreamNonBlocking);
  }

  int sz = widths[0] * heights[0];

  std::vector<int64_t> out_shape = {output_components, height, width};
  out->Resize(common::make_ddim(out_shape));

  T* data = dev_ctx.template Alloc<T>(out);

  for (int c = 0; c < output_components; c++) {
    out_image.channel[c] = data + c * sz;
    out_image.pitch[c] = width;
  }

  nvjpegStatus_t decode_status = phi::dynload::nvjpegDecode(nvjpeg_handle,
                                                            nvjpeg_state,
                                                            x_data,
                                                            x.numel(),
                                                            output_format,
                                                            &out_image,
                                                            nvjpeg_stream);
}
}  // namespace phi

PD_REGISTER_KERNEL(decode_jpeg,  // cuda_only
                   GPU,
                   ALL_LAYOUT,
                   phi::DecodeJpegKernel,
                   uint8_t) {
  kernel->InputAt(0).SetBackend(phi::Backend::ALL_BACKEND);
}

#endif
