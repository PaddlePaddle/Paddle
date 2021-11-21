/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include <vector>
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/dynload/nvjpeg.h"
#include "paddle/fluid/platform/stream/cuda_stream.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;

static int dev_malloc(void **p, size_t s) { return (int)cudaMalloc(p, s); }
static int dev_free(void *p) { return (int)cudaFree(p);  }

static int host_malloc(void** p, size_t s, unsigned int f) {
  return (int)cudaHostAlloc(p, s, f);
}

static int host_free(void* p) { return (int)cudaFreeHost(p);  }

class NvjpegDecoder {
  public:
    NvjpegDecoder(std::string mode) 
      : nvjpeg_streams_(2),
        page_id_(0),
        pinned_buffers_(2),
        mode_(mode) {
      // create cuda stream
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamCreateWithFlags(&cuda_stream_, cudaStreamNonBlocking));

      // create nvjpeg handle and stream
      // device_allocator_.dev_malloc = &cudaMalloc;
      // device_allocator_.dev_free = &cudaFree;
      // pinned_allocator_.pinned_malloc = &cudaMallocHost;
      // pinned_allocator_.pinned_free = &cudaFreeHost;
      PADDLE_ENFORCE_NVJPEG_SUCCESS(
          platform::dynload::nvjpegCreateEx(NVJPEG_BACKEND_DEFAULT, &device_allocator_,
                               &pinned_allocator_, 0, &handle_));
      for (size_t i; i < nvjpeg_streams_.size(); i++) {
        PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamCreate(handle_, &nvjpeg_streams_[i]));
      }

      // create decode params, decoder and state
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsCreate(handle_, &decode_params_));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderCreate(handle_, NVJPEG_BACKEND_DEFAULT, &decoder_));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderStateCreate(handle_, decoder_, &state_));

      // create device & pinned buffer
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferDeviceCreate(handle_, &device_allocator_, &device_buffer_));
      for (size_t i = 0; i < pinned_buffers_.size(); i++) {
        PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferPinnedCreate(handle_, &pinned_allocator_, &pinned_buffers_[i]));
      }
    }

    ~NvjpegDecoder() {
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(cuda_stream_));

      // destroy nvjpeg streams
      for (size_t i = 0; i < nvjpeg_streams_.size(); i++) {
        PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamDestroy(nvjpeg_streams_[i]));
      }

      // destroy decode params, decoder and state
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsDestroy(decode_params_));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecoderDestroy(decoder_));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStateDestroy(state_));

      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferDeviceDestroy(device_buffer_));
      for (size_t i = 0; i < pinned_buffers_.size(); i++) {
        PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegBufferPinnedDestroy(pinned_buffers_[i]));
      }

      // destroy nvjpeg handle and cuda stream at last
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDestroy(handle_));
      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamDestroy(cuda_stream_));
    }

    void ParseOutputInfo(
        const uint8_t* bit_stream, size_t bit_len, LoDTensor* out,
        nvjpegImage_t* out_image, framework::ExecutionContext ctx) {
      int components;
      nvjpegChromaSubsampling_t subsampling;
      int widths[NVJPEG_MAX_COMPONENT];
      int heights[NVJPEG_MAX_COMPONENT];

      PADDLE_ENFORCE_NVJPEG_SUCCESS(
          platform::dynload::nvjpegGetImageInfo(handle_, bit_stream, bit_len,
                             &components, &subsampling, widths, heights));

      int width = widths[0];
      int height = heights[0];

      nvjpegOutputFormat_t output_format;
      int output_components;

      if (mode_ == "unchanged") {
        if (components == 1) {
          output_format = NVJPEG_OUTPUT_Y;
          output_components = 1;
        } else if (components == 3) {
          output_format = NVJPEG_OUTPUT_RGB;
          output_components = 3;
        } else {
          PADDLE_THROW(platform::errors::Fatal(
              "The provided mode is not supported for JPEG files on GPU"));
        }
      } else if (mode_ == "gray") {
        output_format = NVJPEG_OUTPUT_Y;
        output_components = 1;
      } else if (mode_ == "rgb") {
        output_format = NVJPEG_OUTPUT_RGB;
        output_components = 3;
      } else {
        PADDLE_THROW(platform::errors::Fatal(
            "The provided mode is not supported for JPEG files on GPU"));
      }

      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeParamsSetOutputFormat(decode_params_, output_format));

      std::vector<int64_t> out_shape = {output_components, height, width};
      out->Resize(framework::make_ddim(out_shape));

      // allocate memory and assign to out_image
      auto* data = out->mutable_data<uint8_t>(ctx.GetPlace());
      for (int c = 0; c < output_components; c++) {
        out_image->channel[c] = data + c * width * height;
        out_image->pitch[c] = width;
      }
    }

    void Decode(const uint8_t* bit_stream, size_t bit_len, nvjpegImage_t* out_image) {
      auto buffer = pinned_buffers_[page_id_];
      auto stream = nvjpeg_streams_[page_id_];
      page_id_ ^= 1;

      // decode jpeg in host to pinned buffer
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegStateAttachPinnedBuffer(state_, buffer));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegJpegStreamParse(handle_, bit_stream, bit_len, false, false, stream));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegHost(handle_, decoder_, state_, decode_params_, stream));

      // transfer and decode to device buffer
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegStateAttachDeviceBuffer(state_, device_buffer_));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegTransferToDevice(handle_, decoder_, state_, stream, cuda_stream_));
      PADDLE_ENFORCE_NVJPEG_SUCCESS(platform::dynload::nvjpegDecodeJpegDevice(handle_, decoder_, state_, out_image, cuda_stream_));

      PADDLE_ENFORCE_CUDA_SUCCESS(cudaStreamSynchronize(cuda_stream_));
    }

    void Run(const uint8_t* bit_stream, size_t bit_len, LoDTensor* out, const framework::ExecutionContext& ctx) {
      nvjpegImage_t image;
      ParseOutputInfo(bit_stream, bit_len, out, &image, ctx);
      Decode(bit_stream, bit_len, &image);
    }

  private:
    DISABLE_COPY_AND_ASSIGN(NvjpegDecoder);

    cudaStream_t cuda_stream_ = nullptr;
    std::vector<nvjpegJpegStream_t> nvjpeg_streams_;

    nvjpegHandle_t handle_ = nullptr;
    nvjpegJpegState_t state_ = nullptr;
    nvjpegJpegDecoder_t decoder_ = nullptr;
    nvjpegDecodeParams_t decode_params_ = nullptr;

    nvjpegPinnedAllocator_t pinned_allocator_ = {&host_malloc, &host_free};
    nvjpegDevAllocator_t device_allocator_ = {&dev_malloc, &dev_free};
    std::vector<nvjpegBufferPinned_t> pinned_buffers_;
    nvjpegBufferDevice_t device_buffer_ = nullptr;

    int page_id_;

    const std::string mode_;
};


// class NvjpegDecoderWorkerPool {
//   public:
//     NvjpegDecoderWorkerPool(const int num_threads, )
//
//   private:
//     DISABLE_COPY_AND_ASSIGN(NvjpegDecoderWorkerPool);
//
//     struct NvjpegDecoderTask {
//       const uint8_t* bit_stream;
//       const size_t bit_len;
//       LoDTensor* out;
//     }
//
//     class NvjpegDecoderWorker {
//       public:
//         NvjpegDecoderWorker(
//             const std::string mode, framework::ExecutionContext ctx,
//             const int capacity)
//             : mode_(mode),
//               ctx_(ctx),
//               capacity_(capacity),
//               pool_(1) {
//
//         }
//
//       private:
//         const std::string mode_;
//         const framework::ExecutionContext ctx_;
//
//         BlockingQueue<std>
//         ThreadPool pool_;
//     }
//
// }

}  // namespace operators
}  // namespace paddle
