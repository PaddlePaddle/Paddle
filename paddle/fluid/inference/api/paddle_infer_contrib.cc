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

#include "paddle/fluid/inference/api/paddle_infer_contrib.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle_infer {
namespace contrib {

using paddle::PaddleDType;

void* TensorUtils::CudaMallocPinnedMemory(size_t size) {
#if defined(PADDLE_WITH_CUDA)
  void* ptr = nullptr;
  PADDLE_ENFORCE_GPU_SUCCESS(cudaMallocHost(&ptr, size));
  return ptr;
#else
  return nullptr;
#endif
}

void TensorUtils::CudaFreePinnedMemory(void* ptr) {
#if defined(PADDLE_WITH_CUDA)
  PADDLE_ENFORCE_GPU_SUCCESS(cudaFreeHost(ptr));
#endif
}

void TensorUtils::CopyTensorImpl(Tensor* p_dst, const Tensor& src,
                                 void* exec_stream, CallbackFunc cb,
                                 void* cb_params) {
  Tensor& dst = *p_dst;
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
        src.CopyToCpuImpl(dst.mutable_data<int32_t>(PlaceType::kCPU),
                          exec_stream, cb, cb_params);
        break;
      case PaddleDType::INT64:
        src.CopyToCpuImpl(dst.mutable_data<int64_t>(PlaceType::kCPU),
                          exec_stream, cb, cb_params);
        break;
      case PaddleDType::FLOAT32:
        src.CopyToCpuImpl(dst.mutable_data<float>(PlaceType::kCPU), exec_stream,
                          cb, cb_params);
        break;
      case PaddleDType::UINT8:
        src.CopyToCpuImpl(dst.mutable_data<uint8_t>(PlaceType::kCPU),
                          exec_stream, cb, cb_params);
        break;
      case PaddleDType::INT8:
        src.CopyToCpuImpl(dst.mutable_data<int8_t>(PlaceType::kCPU),
                          exec_stream, cb, cb_params);
        break;
      case PaddleDType::FLOAT16:
        src.CopyToCpuImpl(
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

void TensorUtils::CopyTensor(Tensor* p_dst, const Tensor& src) {
  CopyTensorImpl(p_dst, src, nullptr, nullptr, nullptr);
}

void TensorUtils::CopyTensorAsync(Tensor* p_dst, const Tensor& src,
                                  void* exec_stream) {
  CopyTensorImpl(p_dst, src, exec_stream, nullptr, nullptr);
}

void TensorUtils::CopyTensorAsync(Tensor* p_dst, const Tensor& src,
                                  CallbackFunc cb, void* cb_params) {
  CopyTensorImpl(p_dst, src, nullptr, cb, cb_params);
}

struct Status::Impl {
  int ec{0};
  std::string msg;
};

Status::Status() noexcept : impl_(new Impl) {}
Status::Status(const Status& status) noexcept : impl_(new Impl) {
  *impl_ = *status.impl_;
}

Status& Status::operator=(const Status& status) noexcept {
  *impl_ = *status.impl_;
  return *this;
}
Status::Status(std::exception_ptr e) noexcept : impl_(new Impl) {
  constexpr int kDefaultError{-1};
  impl_->ec = kDefaultError;
  try {
    std::rethrow_exception(e);
  } catch (paddle::platform::EnforceNotMet& e) {
    // Add one to the error code to make the number zero a non-error
    // status code.
    impl_->ec = e.code() + 1;
    impl_->msg = e.what();
  } catch (const std::exception& e) {
    impl_->msg = e.what();
  }
}
Status Status::OK() noexcept { return Status(); }
bool Status::ok() const noexcept { return impl_->ec == 0; }
Status::Code Status::code() const noexcept { return impl_->ec; }
const std::string& Status::error_message() const noexcept { return impl_->msg; }
bool Status::operator==(const Status& x) const noexcept {
  return code() == x.code() && error_message() == x.error_message();
}
bool Status::operator!=(const Status& x) const noexcept {
  return !(*this == x);
}

}  // namespace contrib
}  // namespace paddle_infer
