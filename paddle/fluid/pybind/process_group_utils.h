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

#pragma once

#include "paddle/phi/api/include/tensor.h"
#include "paddle/phi/backends/device_guard.h"
#include "paddle/phi/backends/device_manager.h"
#include "paddle/phi/core/device_context.h"
#include "paddle/phi/kernels/funcs/concat_and_split_functor.h"

namespace paddle {
namespace pybind {

template <typename DeviceContext, typename T>
struct ConcatDenseTensor {
  void operator()(const DeviceContext &context,
                  const std::vector<phi::DenseTensor> &in,
                  phi::DenseTensor *out,
                  int axis = 0) {
    phi::funcs::ConcatFunctor<DeviceContext, T> concat_functor;
    concat_functor(context, in, axis, out);
  }
};

template <typename DeviceContext, typename T>
struct SplitDenseTensor {
  void operator()(const DeviceContext &context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor *> *out,
                  int axis = 0) {
    std::vector<const phi::DenseTensor *> shape_refer;
    shape_refer.reserve(out->size());
    for (auto *p_tensor : *out) {
      shape_refer.emplace_back(p_tensor);
    }
    phi::funcs::SplitFunctor<DeviceContext, T> split_functor;
    split_functor(context, in, shape_refer, axis, out);
  }
};

#ifdef PADDLE_WITH_CUSTOM_DEVICE
template <typename T>
struct ConcatDenseTensor<phi::CustomContext, T> {
  void operator()(const phi::CustomContext &context,
                  const std::vector<phi::DenseTensor> &in,
                  phi::DenseTensor *out,
                  int axis UNUSED = 0) {
    VLOG(10) << "ConcatDenseTensor: " << in.size();
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            "concat",
            phi::KernelKey(phi::TransToPhiBackend(context.GetPlace()),
                           phi::DataLayout::ALL_LAYOUT,
                           phi::CppTypeToDataType<T>::Type()));
    const auto &kernel = kernel_result.kernel;
    using kernel_signature =
        void (*)(const phi::DeviceContext &,
                 const std::vector<const phi::DenseTensor *> &,
                 const phi::Scalar &,
                 phi::DenseTensor *);
    auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();
    std::vector<const phi::DenseTensor *> inputs;
    (*kernel_fn)(context, inputs, phi::Scalar(0), out);
  }
};

template <typename T>
struct SplitDenseTensor<phi::CustomContext, T> {
  void operator()(const phi::CustomContext &context,
                  const phi::DenseTensor &in,
                  std::vector<phi::DenseTensor *> *out,
                  int axis UNUSED = 0) {
    VLOG(10) << "SplitDenseTensor: " << out->size();
    auto kernel_result =
        phi::KernelFactory::Instance().SelectKernelOrThrowError(
            "split_with_num",
            phi::KernelKey(phi::TransToPhiBackend(context.GetPlace()),
                           phi::DataLayout::ALL_LAYOUT,
                           phi::CppTypeToDataType<T>::Type()));
    const auto &kernel = kernel_result.kernel;
    using kernel_signature = void (*)(const phi::DeviceContext &,
                                      const phi::DenseTensor &,
                                      int,
                                      const phi::Scalar &,
                                      std::vector<phi::DenseTensor *>);
    auto *kernel_fn = kernel.GetVariadicKernelFn<kernel_signature>();

    auto in_dims = common::vectorize(in.dims());
    auto origin_out_dims = common::vectorize(out->at(0)->dims());
    for (auto *tensor : *out) {
      if (origin_out_dims.size() != in_dims.size()) {
        std::vector<int> new_dims({1});
        new_dims.insert(
            new_dims.end(), origin_out_dims.begin(), origin_out_dims.end());
        tensor->Resize(common::make_ddim(new_dims));
      }
    }
    (*kernel_fn)(context, in, out->size(), phi::Scalar(0), *out);
    for (auto *tensor : *out) {
      auto tensor_dims = common::vectorize(tensor->dims());
      if (tensor_dims.size() != origin_out_dims.size()) {
        tensor->Resize(common::make_ddim(origin_out_dims));
      }
    }
  }
};
#endif

template <typename DeviceContext>
void ConcatDenseTensorWithType(const DeviceContext &dev_ctx,
                               const std::vector<phi::DenseTensor> &t_list,
                               phi::DenseTensor *p_out,
                               phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      ConcatDenseTensor<DeviceContext, bool>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensor<DeviceContext, uint8_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT8:
      ConcatDenseTensor<DeviceContext, int8_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensor<DeviceContext, int32_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensor<DeviceContext, int64_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT16:
      ConcatDenseTensor<DeviceContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::BFLOAT16:
      ConcatDenseTensor<DeviceContext, phi::dtype::bfloat16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensor<DeviceContext, float>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT64:
      ConcatDenseTensor<DeviceContext, double>()(dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}

#ifdef PADDLE_WITH_XPU
template <>
void ConcatDenseTensorWithType(const phi::XPUContext &dev_ctx,
                               const std::vector<phi::DenseTensor> &t_list,
                               phi::DenseTensor *p_out,
                               phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      ConcatDenseTensor<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::BFLOAT16:
      ConcatDenseTensor<phi::XPUContext, phi::dtype::bfloat16>()(
          dev_ctx, t_list, p_out);
      break;
    case phi::DataType::FLOAT32:
      ConcatDenseTensor<phi::XPUContext, float>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT32:
      ConcatDenseTensor<phi::XPUContext, int32_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::INT64:
      ConcatDenseTensor<phi::XPUContext, int64_t>()(dev_ctx, t_list, p_out);
      break;
    case phi::DataType::UINT8:
      ConcatDenseTensor<phi::XPUContext, uint8_t>()(dev_ctx, t_list, p_out);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it concats tensors.", type));
  }
}
#endif

template <typename DeviceContext>
void SplitDenseTensorWithType(const DeviceContext &dev_ctx,
                              const phi::DenseTensor &t_in,
                              std::vector<phi::DenseTensor *> *p_list,
                              phi::DataType type) {
  switch (type) {
    case phi::DataType::BOOL:
      SplitDenseTensor<DeviceContext, bool>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::UINT8:
      SplitDenseTensor<DeviceContext, uint8_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT8:
      SplitDenseTensor<DeviceContext, int8_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT32:
      SplitDenseTensor<DeviceContext, int32_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT64:
      SplitDenseTensor<DeviceContext, int64_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT16:
      SplitDenseTensor<DeviceContext, phi::dtype::float16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::BFLOAT16:
      SplitDenseTensor<DeviceContext, phi::dtype::bfloat16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDenseTensor<DeviceContext, float>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT64:
      SplitDenseTensor<DeviceContext, double>()(dev_ctx, t_in, p_list);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}

#ifdef PADDLE_WITH_XPU
template <>
void SplitDenseTensorWithType(const phi::XPUContext &dev_ctx,
                              const phi::DenseTensor &t_in,
                              std::vector<phi::DenseTensor *> *p_list,
                              phi::DataType type) {
  switch (type) {
    case phi::DataType::FLOAT16:
      SplitDenseTensor<phi::XPUContext, phi::dtype::float16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::BFLOAT16:
      SplitDenseTensor<phi::XPUContext, phi::dtype::bfloat16>()(
          dev_ctx, t_in, p_list);
      break;
    case phi::DataType::FLOAT32:
      SplitDenseTensor<phi::XPUContext, float>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT32:
      SplitDenseTensor<phi::XPUContext, int32_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::INT64:
      SplitDenseTensor<phi::XPUContext, int64_t>()(dev_ctx, t_in, p_list);
      break;
    case phi::DataType::UINT8:
      SplitDenseTensor<phi::XPUContext, uint8_t>()(dev_ctx, t_in, p_list);
      break;
    default:
      PADDLE_THROW(common::errors::Unimplemented(
          "Data type (%s) is not supported when it splits tensors.", type));
  }
}
#endif

void ConcatTensor(const phi::DeviceContext &dev_ctx,
                  const std::vector<phi::DenseTensor> &tensor_list,
                  const Tensor *tensor) {
  auto *dense_tensor =
      std::dynamic_pointer_cast<phi::DenseTensor>(tensor->impl()).get();

  const auto &place = dev_ctx.GetPlace();
  if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    ConcatDenseTensorWithType(static_cast<const phi::GPUContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not support GPU, please "
        "recompile or reinstall Paddle with GPU support."));
#endif
  } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    ConcatDenseTensorWithType(static_cast<const phi::XPUContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not support XPU, please "
        "recompile or reinstall Paddle with XPU support."));
#endif
  } else if (phi::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    ConcatDenseTensorWithType(static_cast<const phi::CustomContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't concat tensor since it's not compiled with "
        "CUSTOM_DEVICE, please recompile or reinstall Paddle with "
        "CUSTOM_DEVICE support."));
#endif
  } else if (phi::is_cpu_place(place)) {
    ConcatDenseTensorWithType(static_cast<const phi::CPUContext &>(dev_ctx),
                              tensor_list,
                              dense_tensor,
                              tensor->dtype());
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Concat tensor not supported on place (%s)", place));
  }
}

void SplitTensor(const phi::DeviceContext &dev_ctx,
                 const phi::DenseTensor &tensor,
                 const std::vector<Tensor> *tensor_list) {
  std::vector<phi::DenseTensor *> dense_list;
  for (auto &tensor : *tensor_list) {
    auto *p_tensor =
        std::dynamic_pointer_cast<phi::DenseTensor>(tensor.impl()).get();
    dense_list.emplace_back(p_tensor);
  }

  const auto &place = dev_ctx.GetPlace();
  if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    SplitDenseTensorWithType(static_cast<const phi::GPUContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't split tensor since it's not support GPU, please "
        "recompile or reinstall Paddle with GPU support."));
#endif
  } else if (phi::is_xpu_place(place)) {
#ifdef PADDLE_WITH_XPU
    SplitDenseTensorWithType(static_cast<const phi::XPUContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with XPU, "
        "please recompile or reinstall Paddle with XPU support."));
#endif
  } else if (phi::is_custom_place(place)) {
#ifdef PADDLE_WITH_CUSTOM_DEVICE
    SplitDenseTensorWithType(static_cast<const phi::CustomContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
#else
    PADDLE_THROW(common::errors::PermissionDenied(
        "Paddle can't split tensor since it's not compiled with CUSTOM_DEVICE, "
        "please recompile or reinstall Paddle with CUSTOM_DEVICE support."));
#endif
  } else if (phi::is_cpu_place(place)) {
    SplitDenseTensorWithType(static_cast<const phi::CPUContext &>(dev_ctx),
                             tensor,
                             &dense_list,
                             tensor.dtype());
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Split tensor not supported on place (%s)", place));
  }
}

inline std::vector<int64_t> GetDefaultSplitSizes(const phi::DenseTensor &tensor,
                                                 int world_size) {
  return std::vector<int64_t>(world_size, tensor.dims()[0] / world_size);
}

}  //  namespace pybind
}  //  namespace paddle
