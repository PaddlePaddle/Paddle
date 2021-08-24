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

#include "paddle_infer_contrib.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle_infer {
namespace contrib {

using paddle::PaddleDType;

std::unique_ptr<Tensor> TensorUtils::CreateInferTensorForTest(
    const std::string& name, PlaceType place, void* p_scope) {
  auto var = static_cast<paddle::framework::Scope*>(p_scope)->Var(name);
  auto tensor = var->GetMutable<paddle::framework::LoDTensor>();
  (void)tensor;
  std::unique_ptr<Tensor> res(new Tensor(p_scope));
  res->input_or_output_ = true;
  res->SetName(name);
  res->SetPlace(place, 0 /*device id*/);
  return res;
}

void* TensorUtils::CudaMallocPinnedMemory(size_t size) {
#if defined(PADDLE_WITH_CUDA)
  void* ptr = nullptr;
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaMallocHost(&ptr, size));
  return ptr;
#else
  return nullptr;
#endif
}

void TensorUtils::CudaFreePinnedMemory(void* ptr) {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_CUDA_SUCCESS(cudaFreeHost(ptr));
#endif
}

void TensorUtils::CopyTensorImp(Tensor& dst, const Tensor& src,
                                void* exec_stream, CallbackFunc cb,
                                void* cb_params) {
  dst.Reshape(src.shape());
  PADDLE_ENFORCE(
      src.place() == PlaceType::kCPU || src.place() == PlaceType::kGPU,
      paddle::platform::errors::InvalidArgument(
          "CopyTensor only support PlaceType kCPU/kGPU now."));
  PADDLE_ENFORCE(
      dst.place() == PlaceType::kCPU || dst.place() == PlaceType::kGPU,
      paddle::platform::errors::InvalidArgument(
          "CopyTensor only support PlaceType kCPU/kGPU now."));
  // copy to cpu, gpu => cpu or cpu => cpu
  if (dst.place() == PlaceType::kCPU) {
    switch (src.type()) {
      case PaddleDType::INT32:
        src.CopyToCpuImp(dst.mutable_data<int32_t>(PlaceType::kCPU),
                         exec_stream, cb, cb_params);
        break;
      case PaddleDType::INT64:
        src.CopyToCpuImp(dst.mutable_data<int64_t>(PlaceType::kCPU),
                         exec_stream, cb, cb_params);
        break;
      case PaddleDType::FLOAT32:
        src.CopyToCpuImp(dst.mutable_data<float>(PlaceType::kCPU), exec_stream,
                         cb, cb_params);
        break;
      case PaddleDType::UINT8:
        src.CopyToCpuImp(dst.mutable_data<uint8_t>(PlaceType::kCPU),
                         exec_stream, cb, cb_params);
        break;
      case PaddleDType::INT8:
        src.CopyToCpuImp(dst.mutable_data<int8_t>(PlaceType::kCPU), exec_stream,
                         cb, cb_params);
        break;
      case PaddleDType::FLOAT16:
        src.CopyToCpuImp(
            dst.mutable_data<paddle::platform::float16>(PlaceType::kCPU),
            exec_stream, cb, cb_params);
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::Unimplemented(
            "Only INT32, INT64, UINT8, INT8, FLOAT16 and "
            "FLOAT32 is supported in Tensor. Others not implements"));
    }
    // gpu => gpu or cpu => gpu
  } else {
#if defined(PADDLE_WITH_CUDA)
    void* dst_data = nullptr;
    void* src_data = nullptr;
    size_t data_len = 0;
    int data_size = 0;
    PlaceType src_place;
    switch (src.type()) {
      case PaddleDType::INT32:
        dst_data =
            static_cast<void*>(dst.mutable_data<int32_t>(PlaceType::kGPU));
        src_data =
            static_cast<void*>(src.data<int32_t>(&src_place, &data_size));
        data_len = data_size * sizeof(int32_t);
        break;
      case PaddleDType::INT64:
        dst_data =
            static_cast<void*>(dst.mutable_data<int64_t>(PlaceType::kGPU));
        src_data =
            static_cast<void*>(src.data<int64_t>(&src_place, &data_size));
        data_len = data_size * sizeof(int64_t);
        break;
      case PaddleDType::FLOAT32:
        dst_data = static_cast<void*>(dst.mutable_data<float>(PlaceType::kGPU));
        src_data = static_cast<void*>(src.data<float>(&src_place, &data_size));
        data_len = data_size * sizeof(float);
        break;
      case PaddleDType::UINT8:
        dst_data =
            static_cast<void*>(dst.mutable_data<uint8_t>(PlaceType::kGPU));
        src_data =
            static_cast<void*>(src.data<uint8_t>(&src_place, &data_size));
        data_len = data_size * sizeof(uint8_t);
        break;
      case PaddleDType::INT8:
        dst_data =
            static_cast<void*>(dst.mutable_data<int8_t>(PlaceType::kGPU));
        src_data = static_cast<void*>(src.data<int8_t>(&src_place, &data_size));
        data_len = data_size * sizeof(int8_t);
        break;
      case PaddleDType::FLOAT16:
        dst_data = static_cast<void*>(
            dst.mutable_data<paddle::platform::float16>(PlaceType::kGPU));
        src_data = static_cast<void*>(
            src.data<paddle::platform::float16>(&src_place, &data_size));
        data_len = data_size * 2;
        break;
      default:
        PADDLE_THROW(paddle::platform::errors::Unimplemented(
            "Only INT32, INT64, UINT8, INT8, FLOAT16 and "
            "FLOAT32 is supported in Tensor. Others not implements"));
    }

    paddle::platform::DeviceContextPool& pool =
        paddle::platform::DeviceContextPool::Instance();
    paddle::platform::CUDAPlace gpu_place(dst.device_);
    auto* dev_ctx = static_cast<const paddle::platform::CUDADeviceContext*>(
        pool.Get(gpu_place));

    if (src.place() == PlaceType::kCPU) {
      paddle::memory::Copy(gpu_place, static_cast<void*>(dst_data),
                           paddle::platform::CPUPlace(), src_data, data_len,
                           dev_ctx->stream());
    } else {
      paddle::memory::Copy(gpu_place, static_cast<void*>(dst_data),
                           paddle::platform::CUDAPlace(), src_data, data_len,
                           dev_ctx->stream());
    }

    if (nullptr != exec_stream) {
      *(static_cast<cudaStream_t*>(exec_stream)) = dev_ctx->stream();
    } else if (cb) {
      cudaLaunchHostFunc(dev_ctx->stream(), cb, cb_params);
    } else {
      cudaStreamSynchronize(dev_ctx->stream());
    }
#else
    PADDLE_THROW(paddle::platform::errors::Unavailable(
        "Can not copy tensor to GPU CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  }
  return;
}

void TensorUtils::CopyTensor(Tensor& dst, const Tensor& src) {
  CopyTensorImp(dst, src, nullptr, nullptr, nullptr);
}

void TensorUtils::CopyTensorAsync(Tensor& dst, const Tensor& src,
                                  void* exec_stream) {
  CopyTensorImp(dst, src, exec_stream, nullptr, nullptr);
}

void TensorUtils::CopyTensorAsync(Tensor& dst, const Tensor& src,
                                  CallbackFunc cb, void* cb_params) {
  CopyTensorImp(dst, src, nullptr, cb, cb_params);
}

}  // namespace contrib
}  // namespace paddle_infer
