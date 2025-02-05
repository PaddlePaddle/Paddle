// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/data_layout_transform.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/api/paddle_inference_api.h"
#include "paddle/fluid/inference/api/paddle_tensor.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/allocator.h"
#include "paddle/phi/core/memory/memcpy.h"
#include "paddle/phi/core/vocab/string_array.h"
#ifdef PADDLE_WITH_ONNXRUNTIME
#include "onnxruntime_c_api.h"    // NOLINT
#include "onnxruntime_cxx_api.h"  // NOLINT
#endif

namespace paddle_infer {

using float16 = phi::dtype::float16;
using bfloat16 = phi::dtype::bfloat16;

void Tensor::Reshape(const std::vector<int> &shape) {
#ifdef PADDLE_WITH_ONNXRUNTIME
  if (is_ort_tensor_) {
    shape_.assign(shape.begin(), shape.end());
    return;
  }
#endif

  PADDLE_ENFORCE_EQ(
      name_.empty(),
      false,
      common::errors::PreconditionNotMet(
          "Need to SetName first, so that the corresponding tensor can "
          "be retrieved."));
  PADDLE_ENFORCE_EQ(input_or_output_,
                    true,
                    common::errors::PermissionDenied(
                        "Can't reshape the output tensor, it is readonly"));
  auto *scope = static_cast<paddle::framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE_NOT_NULL(
      var,
      common::errors::PreconditionNotMet(
          "No tensor called [%s] in the runtime scope", name_));
  auto *tensor = var->GetMutable<phi::DenseTensor>();
  tensor->Resize(common::make_ddim(shape));
}

void Tensor::ReshapeStrings(const size_t &shape) {
  PADDLE_ENFORCE_EQ(
      name_.empty(),
      false,
      common::errors::PreconditionNotMet(
          "Need to SetName first, so that the corresponding tensor can "
          "be retrieved."));
  PADDLE_ENFORCE_EQ(input_or_output_,
                    true,
                    common::errors::PermissionDenied(
                        "Can't reshape the output tensor, it is readonly"));
  auto *scope = static_cast<paddle::framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE_NOT_NULL(
      var,
      common::errors::PreconditionNotMet(
          "No tensor called [%s] in the runtime scope", name_));
  phi::Strings *tensor = var->GetMutable<phi::Strings>();
  tensor->resize(shape);
}

#define EAGER_GET_TENSOR(tensor_type)    \
  if (!tensor_) {                        \
    tensor_ = FindTensor<tensor_type>(); \
  }                                      \
  auto *tensor = static_cast<tensor_type *>(tensor_);

template <typename T>
T *Tensor::mutable_data(PlaceType place) {
#ifdef PADDLE_WITH_ONNXRUNTIME
  if (is_ort_tensor_) {
    return ORTGetMutableData<T>();
  }
#endif
  EAGER_GET_TENSOR(phi::DenseTensor);
  PADDLE_ENFORCE_GT(
      tensor->numel(),
      0,
      common::errors::PreconditionNotMet(
          "You should call Tensor::Reshape(const std::vector<int> "
          "&shape)"
          "function before retrieving mutable_data from input tensor."));
  switch (static_cast<int>(place)) {
    case static_cast<int>(PlaceType::kCPU): {
      return tensor->mutable_data<T>(phi::CPUPlace());
    }
    case static_cast<int>(PlaceType::kGPU): {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      phi::GPUPlace gpu_place(device_);
      auto *dev_ctxs = reinterpret_cast<const std::map<
          phi::Place,
          std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
          device_contexts_);
      auto *dev_ctx =
          static_cast<phi::GPUContext *>(dev_ctxs->at(gpu_place).get().get());
      return dev_ctx->Alloc<T>(tensor, tensor->numel() * sizeof(T));
#else
      return tensor->mutable_data<T>(phi::GPUPlace(device_));
#endif
    }
    case static_cast<int>(PlaceType::kXPU): {
      return tensor->mutable_data<T>(phi::XPUPlace(device_));
    }
    case static_cast<int>(PlaceType::kCUSTOM): {
      return tensor->mutable_data<T>(phi::CustomPlace(device_type_, device_));
    }
    default:
      PADDLE_THROW(common::errors::Unavailable(
          "Only CPU / CUDA / XPU places is supported. The place `%d` is "
          "not supported.",
          static_cast<int>(place)));
      break;
  }
  return nullptr;
}

template <typename T>
T *Tensor::data(PlaceType *place, int *size) const {
  EAGER_GET_TENSOR(phi::DenseTensor);
  auto *res = tensor->data<T>();

  if (phi::is_cpu_place(tensor->place())) {
    *place = PlaceType::kCPU;
  } else if (phi::is_gpu_place(tensor->place())) {
    *place = PlaceType::kGPU;
  } else if (phi::is_xpu_place(tensor->place())) {
    *place = PlaceType::kXPU;
  } else if (phi::is_custom_place(tensor->place())) {
    *place = PlaceType::kCUSTOM;
  } else {
    *place = PlaceType::kUNK;
  }

  *size = static_cast<int>(tensor->numel());
  return res;
}

DataType Tensor::type() const {
#ifdef PADDLE_WITH_ONNXRUNTIME
  if (is_ort_tensor_) {
    return dtype_;
  }
#endif
  EAGER_GET_TENSOR(phi::DenseTensor);
  auto type = paddle::framework::TransToProtoVarType(tensor->dtype());
  if (type == paddle::framework::proto::VarType::FP64) {
    return DataType::FLOAT64;
  } else if (type == paddle::framework::proto::VarType::FP32) {
    return DataType::FLOAT32;
  } else if (type == paddle::framework::proto::VarType::FP16) {
    return DataType::FLOAT16;
  } else if (type == paddle::framework::proto::VarType::BF16) {
    return DataType::BFLOAT16;
  } else if (type == paddle::framework::proto::VarType::INT64) {
    return DataType::INT64;
  } else if (type == paddle::framework::proto::VarType::INT32) {
    return DataType::INT32;
  } else if (type == paddle::framework::proto::VarType::UINT8) {
    return DataType::UINT8;
  } else if (type == paddle::framework::proto::VarType::INT8) {
    return DataType::INT8;
  } else if (type == paddle::framework::proto::VarType::BOOL) {
    return DataType::BOOL;
  }
  return DataType::FLOAT32;
}

PlaceType Tensor::place() const { return place_; }

template <typename T>
void Tensor::CopyFromCpu(const T *data) {
  EAGER_GET_TENSOR(phi::DenseTensor);
  PADDLE_ENFORCE_GE(tensor->numel(),
                    0,
                    common::errors::PreconditionNotMet(
                        "You should call Tensor::Reshape(const "
                        "std::vector<int> &shape)"
                        "function before copying data from cpu."));
  size_t ele_size = tensor->numel() * sizeof(T);

  if (place_ == PlaceType::kCPU) {
    auto *t_data = tensor->mutable_data<T>(phi::CPUPlace());
    std::memcpy(static_cast<void *>(t_data), data, ele_size);
  } else if (place_ == PlaceType::kGPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)

    phi::GPUPlace gpu_place(device_);
    auto *dev_ctxs = reinterpret_cast<const std::map<
        phi::Place,
        std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        device_contexts_);
    auto *dev_ctx =
        static_cast<phi::GPUContext *>(dev_ctxs->at(gpu_place).get().get());
    auto *t_data = dev_ctx->Alloc<T>(tensor, tensor->numel() * sizeof(T));

    paddle::memory::Copy(gpu_place,
                         static_cast<void *>(t_data),
                         phi::CPUPlace(),
                         data,
                         ele_size,
                         dev_ctx->stream());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  } else if (place_ == PlaceType::kXPU) {
#ifdef PADDLE_WITH_XPU
    phi::XPUPlace xpu_place(device_);
    auto *t_data = tensor->mutable_data<T>(xpu_place);
    paddle::memory::Copy(xpu_place,
                         static_cast<void *>(t_data),
                         phi::CPUPlace(),
                         data,
                         ele_size);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with XPU place because paddle is not compiled "
        "with XPU."));
#endif
  } else if (place_ == PlaceType::kCUSTOM) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    phi::CustomPlace custom_place(device_type_, device_);
    auto *t_data = tensor->mutable_data<T>(custom_place);
    auto *dev_ctx =
        static_cast<const phi::CustomContext *>(pool.Get(custom_place));
    paddle::memory::Copy(custom_place,
                         static_cast<void *>(t_data),
                         phi::CPUPlace(),
                         data,
                         ele_size,
                         dev_ctx->stream());
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with Custom place because paddle is not "
        "compiled "
        "with XPU."));
#endif
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "The analysis predictor supports CPU, GPU, XPU and CUSTOM_DEVICE "
        "now."));
  }
}

template <typename T>
struct DataTypeInfo;

template <>
struct DataTypeInfo<double> {
  phi::DataType TYPE = phi::DataType::FLOAT64;
};

template <>
struct DataTypeInfo<float> {
  phi::DataType TYPE = phi::DataType::FLOAT32;
};

template <>
struct DataTypeInfo<float16> {
  phi::DataType TYPE = phi::DataType::FLOAT16;
};

template <>
struct DataTypeInfo<bfloat16> {
  phi::DataType TYPE = phi::DataType::BFLOAT16;
};

template <>
struct DataTypeInfo<int64_t> {
  phi::DataType TYPE = phi::DataType::INT64;
};

template <>
struct DataTypeInfo<int8_t> {
  phi::DataType TYPE = phi::DataType::INT8;
};

template <>
struct DataTypeInfo<uint8_t> {
  phi::DataType TYPE = phi::DataType::UINT8;
};

template <>
struct DataTypeInfo<int32_t> {
  phi::DataType TYPE = phi::DataType::INT32;
};

template <>
struct DataTypeInfo<bool> {
  phi::DataType TYPE = phi::DataType::BOOL;
};

phi::DataLayout LayoutConvert(DataLayout layout) {
  PADDLE_ENFORCE_EQ(
      layout,
      DataLayout::kNCHW,
      common::errors::InvalidArgument("Only NCHW is supported now."));
  return phi::DataLayout::NCHW;
}

template <typename T>
void Tensor::ShareExternalData(const T *data,
                               const std::vector<int> &shape,
                               PlaceType place,
                               DataLayout layout) {
  EAGER_GET_TENSOR(phi::DenseTensor)
  size_t size =
      std::accumulate(shape.begin(), shape.end(), 1, std::multiplies<int>()) *
      sizeof(T);
  phi::DenseTensorMeta meta(
      DataTypeInfo<T>().TYPE, common::make_ddim(shape), LayoutConvert(layout));
  if (place == PlaceType::kCPU) {
    phi::DenseTensor dtensor(std::make_shared<phi::Allocation>(
                                 const_cast<T *>(data), size, phi::CPUPlace()),
                             meta);
    *tensor = std::move(dtensor);
  } else if (place == PlaceType::kGPU) {
    phi::DenseTensor dtensor(
        std::make_shared<phi::Allocation>(
            const_cast<T *>(data), size, phi::GPUPlace(device_)),
        meta);
    *tensor = std::move(dtensor);
  } else if (place == PlaceType::kXPU) {
    phi::DenseTensor dtensor(
        std::make_shared<phi::Allocation>(
            const_cast<T *>(data), size, phi::XPUPlace(device_)),
        meta);
    *tensor = std::move(dtensor);
  } else if (place == PlaceType::kCUSTOM) {
    phi::DenseTensor dtensor(std::make_shared<phi::Allocation>(
                                 const_cast<T *>(data),
                                 size,
                                 phi::CustomPlace(device_type_, device_)),
                             meta);
    *tensor = std::move(dtensor);
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "PlaceType must be one of [PlaceType::kCPU, PlaceType::kGPU, "
        "PlaceType::kXPU]."));
  }
}

void Tensor::CopyStringsFromCpu(const paddle_infer::Strings *data) {
  EAGER_GET_TENSOR(phi::Strings);
  PADDLE_ENFORCE_GE(tensor->size(),
                    0,
                    common::errors::PreconditionNotMet(
                        "You should call Tensor::Reshape(const "
                        "std::size_t &shape)function before copying"
                        "the string data from cpu."));
  *tensor = *data;
}

template <typename T>
void Tensor::CopyToCpuImpl(T *data,
                           void *exec_stream,
                           CallbackFunc cb,
                           void *cb_params) const {
  EAGER_GET_TENSOR(phi::DenseTensor);
  auto ele_num = tensor->numel();
  auto *t_data = tensor->data<T>();
  auto t_place = tensor->place();

  if (phi::is_cpu_place(t_place)) {
#ifdef PADDLE_WITH_DNNL
    if (tensor->layout() == phi::DataLayout::ONEDNN) {
      phi::DenseTensor out;
      auto mem_allocation =
          std::make_shared<paddle::memory::allocation::Allocation>(
              static_cast<void *>(data), ele_num * sizeof(T), phi::CPUPlace());
      out.ResetHolder(mem_allocation);
      phi::funcs::TransDataLayoutFromOneDNN(
          tensor->layout(),
          phi::OneDNNContext::tls().get_cur_paddle_data_layout(),
          *tensor,
          &out,
          phi::CPUPlace(),
          true);
    } else {
      std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
    }
#else
    std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
#endif
  } else if (phi::is_ipu_place(t_place)) {
#ifdef PADDLE_WITH_IPU
    std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with IPU place because paddle is not compiled "
        "with IPU."));
#endif
  } else if (place_ == PlaceType::kGPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    auto gpu_place = t_place;
    auto *dev_ctxs = reinterpret_cast<const std::map<
        phi::Place,
        std::shared_future<std::unique_ptr<phi::DeviceContext>>> *>(
        device_contexts_);
    auto *dev_ctx =
        static_cast<phi::GPUContext *>(dev_ctxs->at(gpu_place).get().get());
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void *>(data),
                         gpu_place,
                         t_data,
                         ele_num * sizeof(T),
                         dev_ctx->stream());
#ifdef PADDLE_WITH_HIP
    hipStreamSynchronize(dev_ctx->stream());
#else
    // async, return stream
    if (nullptr != exec_stream) {
      *(static_cast<cudaStream_t *>(exec_stream)) = dev_ctx->stream();
      // async with callback
    } else if (cb) {
      cudaLaunchHostFunc(dev_ctx->stream(), cb, cb_params);
      // sync
    } else {
      cudaStreamSynchronize(dev_ctx->stream());
    }
#endif
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  } else if (place_ == PlaceType::kXPU) {
#ifdef PADDLE_WITH_XPU
    auto xpu_place = t_place;
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void *>(data),
                         xpu_place,
                         t_data,
                         ele_num * sizeof(T));
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with XPU place because paddle is not compiled "
        "with XPU."));
#endif
  } else {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    phi::DeviceContextPool &pool = phi::DeviceContextPool::Instance();
    auto custom_place = t_place;
    auto *dev_ctx =
        static_cast<const phi::CustomContext *>(pool.Get(custom_place));
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void *>(data),
                         custom_place,
                         t_data,
                         ele_num * sizeof(T),
                         dev_ctx->stream());
    dev_ctx->GetStream()->Synchronize();
#else
    PADDLE_THROW(common::errors::InvalidArgument(
        "The analysis predictor supports CPU, GPU and XPU now."));
#endif
  }
}

template <typename T>
void Tensor::CopyToCpu(T *data) const {
#ifdef PADDLE_WITH_ONNXRUNTIME
  if (is_ort_tensor_) {
    ORTCopyToCpu<T>(data);
    return;
  }
#endif

  CopyToCpuImpl<T>(data, nullptr, nullptr, nullptr);
}

template <typename T>
void Tensor::CopyToCpuAsync(T *data, void *exec_stream) const {
  CopyToCpuImpl<T>(data, exec_stream, nullptr, nullptr);
}

template <typename T>
void Tensor::CopyToCpuAsync(T *data, CallbackFunc cb, void *cb_params) const {
  CopyToCpuImpl<T>(data, nullptr, cb, cb_params);
}

template PD_INFER_DECL void Tensor::CopyFromCpu<double>(const double *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<float>(const float *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<int64_t>(const int64_t *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<int32_t>(const int32_t *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<uint8_t>(const uint8_t *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<int8_t>(const int8_t *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<float16>(const float16 *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<bfloat16>(const bfloat16 *data);
template PD_INFER_DECL void Tensor::CopyFromCpu<bool>(const bool *data);

template PD_INFER_DECL void Tensor::ShareExternalData<double>(
    const double *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<float>(
    const float *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<int64_t>(
    const int64_t *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<int32_t>(
    const int32_t *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<uint8_t>(
    const uint8_t *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<int8_t>(
    const int8_t *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<float16>(
    const float16 *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<bfloat16>(
    const bfloat16 *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);
template PD_INFER_DECL void Tensor::ShareExternalData<bool>(
    const bool *data,
    const std::vector<int> &shape,
    PlaceType place,
    DataLayout layout);

template PD_INFER_DECL void Tensor::CopyToCpu<double>(double *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<float>(float *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<int64_t>(int64_t *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<int32_t>(int32_t *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<uint8_t>(uint8_t *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<int8_t>(int8_t *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<float16>(float16 *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<bfloat16>(bfloat16 *data) const;
template PD_INFER_DECL void Tensor::CopyToCpu<bool>(bool *data) const;

template PD_INFER_DECL void Tensor::CopyToCpuImpl<double>(
    double *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<float>(float *data,
                                                         void *exec_stream,
                                                         CallbackFunc cb,
                                                         void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<int64_t>(
    int64_t *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<int32_t>(
    int32_t *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<uint8_t>(
    uint8_t *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<int8_t>(
    int8_t *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<float16>(
    float16 *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<bfloat16>(
    bfloat16 *data, void *exec_stream, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuImpl<bool>(bool *data,
                                                        void *exec_stream,
                                                        CallbackFunc cb,
                                                        void *cb_params) const;

template PD_INFER_DECL void Tensor::CopyToCpuAsync<double>(
    double *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<float>(
    float *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<int64_t>(
    int64_t *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<int32_t>(
    int32_t *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<uint8_t>(
    uint8_t *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<int8_t>(
    int8_t *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<float16>(
    float16 *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<bfloat16>(
    bfloat16 *data, void *exec_stream) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<bool>(
    bool *data, void *exec_stream) const;

template PD_INFER_DECL void Tensor::CopyToCpuAsync<double>(
    double *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<float>(
    float *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<int64_t>(
    int64_t *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<int32_t>(
    int32_t *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<uint8_t>(
    uint8_t *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<int8_t>(
    int8_t *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<float16>(
    float16 *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<bfloat16>(
    bfloat16 *data, CallbackFunc cb, void *cb_params) const;
template PD_INFER_DECL void Tensor::CopyToCpuAsync<bool>(bool *data,
                                                         CallbackFunc cb,
                                                         void *cb_params) const;

template PD_INFER_DECL double *Tensor::data<double>(PlaceType *place,
                                                    int *size) const;
template PD_INFER_DECL float *Tensor::data<float>(PlaceType *place,
                                                  int *size) const;
template PD_INFER_DECL int64_t *Tensor::data<int64_t>(PlaceType *place,
                                                      int *size) const;
template PD_INFER_DECL int32_t *Tensor::data<int32_t>(PlaceType *place,
                                                      int *size) const;
template PD_INFER_DECL uint8_t *Tensor::data<uint8_t>(PlaceType *place,
                                                      int *size) const;
template PD_INFER_DECL int8_t *Tensor::data<int8_t>(PlaceType *place,
                                                    int *size) const;
template PD_INFER_DECL float16 *Tensor::data<float16>(PlaceType *place,
                                                      int *size) const;
template PD_INFER_DECL bfloat16 *Tensor::data<bfloat16>(PlaceType *place,
                                                        int *size) const;
template PD_INFER_DECL bool *Tensor::data<bool>(PlaceType *place,
                                                int *size) const;

template PD_INFER_DECL double *Tensor::mutable_data<double>(PlaceType place);
template PD_INFER_DECL float *Tensor::mutable_data<float>(PlaceType place);
template PD_INFER_DECL int64_t *Tensor::mutable_data<int64_t>(PlaceType place);
template PD_INFER_DECL int32_t *Tensor::mutable_data<int32_t>(PlaceType place);
template PD_INFER_DECL uint8_t *Tensor::mutable_data<uint8_t>(PlaceType place);
template PD_INFER_DECL int8_t *Tensor::mutable_data<int8_t>(PlaceType place);
template PD_INFER_DECL float16 *Tensor::mutable_data<float16>(PlaceType place);
template PD_INFER_DECL bfloat16 *Tensor::mutable_data<bfloat16>(
    PlaceType place);
template PD_INFER_DECL bool *Tensor::mutable_data<bool>(PlaceType place);

Tensor::Tensor(void *scope, const void *device_contexts)
    : dtype_(DataType::FLOAT16),
      input_or_output_(false),
      scope_{scope},
      device_contexts_(device_contexts),
      place_(PlaceType::kCPU),
      device_(0) {}

template <typename T>
void *Tensor::FindTensor() const {
  PADDLE_ENFORCE_EQ(
      name_.empty(),
      false,
      common::errors::PreconditionNotMet(
          "Need to SetName first, so that the corresponding tensor can "
          "be retrieved."));
  auto *scope = static_cast<paddle::framework::Scope *>(scope_);
  auto *var = scope->FindVar(name_);
  PADDLE_ENFORCE_NOT_NULL(
      var,
      common::errors::PreconditionNotMet(
          "No tensor called [%s] in the runtime scope", name_));
  auto *tensor = var->GetMutable<T>();
  return tensor;
}

std::vector<int> Tensor::shape() const {
#ifdef PADDLE_WITH_ONNXRUNTIME
  if (is_ort_tensor_) {
    std::vector<int> shape;
    // input handle
    if (idx_ < 0) {
      shape.assign(shape_.begin(), shape_.end());
    } else {  // output handle
      auto binding = binding_.lock();
      PADDLE_ENFORCE_NOT_NULL(binding,
                              common::errors::PreconditionNotMet(
                                  "output tensor [%s] no binding ptr", name_));
      std::vector<Ort::Value> outputs = binding->GetOutputValues();
      Ort::Value &value = outputs[idx_];
      auto info = value.GetTensorTypeAndShapeInfo();
      auto ort_shape = info.GetShape();
      shape.assign(ort_shape.begin(), ort_shape.end());
    }
    return shape;
  }
#endif
  EAGER_GET_TENSOR(phi::DenseTensor);
  PADDLE_ENFORCE_NOT_NULL(
      tensor_,
      common::errors::PreconditionNotMet(
          "Not found tensor called %s in the scope", name_));
// oneDNN may does layout transform internally, so need to reorder before
// return
#ifdef PADDLE_WITH_DNNL
  if (tensor->layout() == phi::DataLayout::ONEDNN) {
    phi::DataLayout out_layout =
        phi::OneDNNContext::tls().get_cur_paddle_data_layout();
    // Set default as NCHW in case not specified
    out_layout = out_layout == phi::DataLayout::kAnyLayout
                     ? phi::DataLayout::kNCHW
                     : out_layout;
    // In these data layouts, channel dimension is either on 2nd position: nChw
    // or
    // at last nhwC, so for dim==2 these layouts are the same and nothing should
    // be done. Similarly for dim==1 when you have just one possible
    // combination.
    if (tensor->dims().size() < 3)
      return common::vectorize<int>(tensor->dims());
    if (out_layout == phi::DataLayout::kNHWC ||
        out_layout == phi::DataLayout::kNDHWC) {
      auto dims = common::vectorize<int>(tensor->dims());
      std::rotate(dims.begin() + 1, dims.begin() + 2, dims.end());
      return dims;
    } else {
      return common::vectorize<int>(tensor->dims());
    }
  }
#endif
  return common::vectorize<int>(tensor->dims());
}

void Tensor::SetLoD(const std::vector<std::vector<size_t>> &x) {
  EAGER_GET_TENSOR(phi::DenseTensor);
  phi::LegacyLoD lod;
  for (auto &level : x) {
    lod.emplace_back(level);
  }
  tensor->set_lod(lod);
}

std::vector<std::vector<size_t>> Tensor::lod() const {
  EAGER_GET_TENSOR(phi::DenseTensor);
  std::vector<std::vector<size_t>> res;
  for (auto &level : tensor->lod()) {
    res.emplace_back(level);
  }
  return res;
}

void Tensor::SetName(const std::string &name) { name_ = name; }

const std::string &Tensor::name() const { return name_; }

void Tensor::SetPlace(PlaceType place,
                      int device,
                      const std::string device_type) {
  place_ = place;
  device_ = device;
  device_type_ = device_type;
}

#ifdef PADDLE_WITH_ONNXRUNTIME
void Tensor::SetOrtMark(bool is_ort_tensor) { is_ort_tensor_ = is_ort_tensor; }

void Tensor::SetOrtBinding(const std::shared_ptr<Ort::IoBinding> binding) {
  binding_ = binding;
}

template <typename T>
T *Tensor::ORTGetMutableData() {
  auto binding = binding_.lock();
  PADDLE_ENFORCE_NOT_NULL(binding,
                          common::errors::PreconditionNotMet(
                              "output tensor [%s] no binding ptr", name_));
  std::vector<Ort::Value> outputs = binding->GetOutputValues();
  Ort::Value &value = outputs[idx_];
  return value.GetTensorMutableData<T>();
}

template <typename T>
void Tensor::ORTCopyToCpu(T *data) const {
  auto binding = binding_.lock();
  PADDLE_ENFORCE_NOT_NULL(binding,
                          common::errors::PreconditionNotMet(
                              "output tensor [%s] no binding ptr", name_));
  std::vector<Ort::Value> outputs = binding->GetOutputValues();
  Ort::Value &value = outputs[idx_];
  auto info = value.GetTensorTypeAndShapeInfo();
  size_t size = info.GetElementCount() * sizeof(T);

  if (place_ == PlaceType::kCPU) {
    std::memcpy(static_cast<void *>(data), value.GetTensorData<void *>(), size);
  } else {
    PADDLE_THROW(common::errors::Unavailable(
        "CopyToCpu error.The current ONNXRuntime backend doesn't support "
        "GPU."));
  }
}

template void Tensor::ORTCopyToCpu<float>(float *data) const;
template void Tensor::ORTCopyToCpu<int32_t>(int32_t *data) const;
template void Tensor::ORTCopyToCpu<uint8_t>(uint8_t *data) const;
template void Tensor::ORTCopyToCpu<int8_t>(int8_t *data) const;
template void Tensor::ORTCopyToCpu<float16>(float16 *data) const;
template void Tensor::ORTCopyToCpu<bfloat16>(bfloat16 *data) const;
#endif

namespace experimental {
template <typename T>
void InternalUtils::CopyFromCpuWithIoStream(paddle_infer::Tensor *t,
                                            const T *data,
                                            cudaStream_t stream) {
  if (t->tensor_ == nullptr) {
    PADDLE_ENFORCE_EQ(
        t->name_.empty(),
        false,
        common::errors::PreconditionNotMet(
            "Need to SetName first, so that the corresponding tensor can "
            "be retrieved."));
    auto *scope = static_cast<paddle::framework::Scope *>(t->scope_);
    auto *var = scope->FindVar(t->name_);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::PreconditionNotMet(
            "No tensor called [%s] in the runtime scope", t->name_));
    auto *tensor = var->GetMutable<phi::DenseTensor>();
    t->tensor_ = tensor;
  }

  auto *tensor = static_cast<phi::DenseTensor *>(t->tensor_);
  PADDLE_ENFORCE_GE(tensor->numel(),
                    0,
                    common::errors::PreconditionNotMet(
                        "You should call Tensor::Reshape(const "
                        "std::vector<int> &shape)"
                        "function before copying data from cpu."));
  size_t ele_size = tensor->numel() * sizeof(T);
  if (t->place_ == PlaceType::kCPU) {
    auto *t_data = tensor->mutable_data<T>(phi::CPUPlace());
    std::memcpy(static_cast<void *>(t_data), data, ele_size);
  } else if (t->place_ == PlaceType::kGPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    phi::GPUPlace gpu_place(t->device_);
    auto *t_data = tensor->mutable_data<T>(gpu_place);
    paddle::memory::Copy(gpu_place,
                         static_cast<void *>(t_data),
                         phi::CPUPlace(),
                         data,
                         ele_size,
                         stream);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "CopyFromCpuWithIoStream only supports CPU and GPU now."));
  }
}

template <typename T>
void InternalUtils::CopyToCpuWithIoStream(paddle_infer::Tensor *t,
                                          T *data,
                                          cudaStream_t stream) {
  if (t->tensor_ == nullptr) {
    PADDLE_ENFORCE_EQ(
        t->name_.empty(),
        false,
        common::errors::PreconditionNotMet(
            "Need to SetName first, so that the corresponding tensor can "
            "be retrieved."));
    auto *scope = static_cast<paddle::framework::Scope *>(t->scope_);
    auto *var = scope->FindVar(t->name_);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        common::errors::PreconditionNotMet(
            "No tensor called [%s] in the runtime scope", t->name_));
    auto *tensor = var->GetMutable<phi::DenseTensor>();
    t->tensor_ = tensor;
  }

  auto *tensor = static_cast<phi::DenseTensor *>(t->tensor_);
  auto ele_num = tensor->numel();
  auto *t_data = tensor->data<T>();
  auto t_place = tensor->place();

  if (phi::is_cpu_place(t_place)) {
#ifdef PADDLE_WITH_DNNL
    if (tensor->layout() == phi::DataLayout::ONEDNN) {
      phi::DenseTensor out;
      auto mem_allocation =
          std::make_shared<paddle::memory::allocation::Allocation>(
              static_cast<void *>(data), ele_num * sizeof(T), phi::CPUPlace());
      out.ResetHolder(mem_allocation);
      phi::funcs::TransDataLayoutFromOneDNN(
          tensor->layout(),
          phi::OneDNNContext::tls().get_cur_paddle_data_layout(),
          *tensor,
          &out,
          phi::CPUPlace(),
          true);
    } else {
      std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
    }
#else
    std::memcpy(static_cast<void *>(data), t_data, ele_num * sizeof(T));
#endif
  } else if (t->place_ == PlaceType::kGPU) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    paddle::memory::Copy(phi::CPUPlace(),
                         static_cast<void *>(data),
                         t_place,
                         t_data,
                         ele_num * sizeof(T),
                         stream);
#else
    PADDLE_THROW(common::errors::Unavailable(
        "Can not create tensor with CUDA place because paddle is not compiled "
        "with CUDA."));
#endif
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "CopyToCpuWithIoStream only supports CPU and GPU now."));
  }
}

template void InternalUtils::CopyFromCpuWithIoStream<double>(
    paddle_infer::Tensor *t, const double *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<float>(
    paddle_infer::Tensor *t, const float *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<int64_t>(
    paddle_infer::Tensor *t, const int64_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<int32_t>(
    paddle_infer::Tensor *t, const int32_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<uint8_t>(
    paddle_infer::Tensor *t, const uint8_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<int8_t>(
    paddle_infer::Tensor *t, const int8_t *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<float16>(
    paddle_infer::Tensor *t, const float16 *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<bfloat16>(
    paddle_infer::Tensor *t, const bfloat16 *data, cudaStream_t stream);
template void InternalUtils::CopyFromCpuWithIoStream<bool>(
    paddle_infer::Tensor *t, const bool *data, cudaStream_t stream);

template void InternalUtils::CopyToCpuWithIoStream<double>(
    paddle_infer::Tensor *t, double *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<float>(
    paddle_infer::Tensor *t, float *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<int64_t>(
    paddle_infer::Tensor *t, int64_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<int32_t>(
    paddle_infer::Tensor *t, int32_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<uint8_t>(
    paddle_infer::Tensor *t, uint8_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<int8_t>(
    paddle_infer::Tensor *t, int8_t *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<float16>(
    paddle_infer::Tensor *t, float16 *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<bfloat16>(
    paddle_infer::Tensor *t, bfloat16 *data, cudaStream_t stream);
template void InternalUtils::CopyToCpuWithIoStream<bool>(
    paddle_infer::Tensor *t, bool *data, cudaStream_t stream);

}  // namespace experimental

}  // namespace paddle_infer
