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

#include "paddle/fluid/eager/accumulation/gradient_accumulation.h"
#include <algorithm>
#include <memory>
#include <utility>
#include "paddle/fluid/eager/eager_tensor.h"
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/pten/api/all.h"
#include "paddle/pten/core/convert_utils.h"
#include "paddle/pten/include/core.h"
#include "unsupported/Eigen/CXX11/Tensor"
#ifdef PADDLE_WITH_XPU
#include "xpu/refactor/math.h"
#endif
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#endif

namespace egr {
template <typename T>
class TensorAddFunctor : public boost::static_visitor<> {
 public:
  TensorAddFunctor(int64_t numel, const T* x, T* y)
      : numel_(numel), x_(x), y_(y) {}

  void operator()(const paddle::platform::CPUPlace& place) {
    paddle::platform::CPUDeviceContext* ctx =
        dynamic_cast<paddle::platform::CPUDeviceContext*>(
            paddle::platform::DeviceContextPool::Instance().Get(place));
    auto blas =
        paddle::operators::math::GetBlas<paddle::platform::CPUDeviceContext, T>(
            *ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }

// TODO(jiabin): Support xpu here from gradient_accumulator.cc

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void operator()(const paddle::platform::CUDAPlace& place) {
    paddle::platform::CUDADeviceContext* ctx =
        dynamic_cast<paddle::platform::CUDADeviceContext*>(
            paddle::platform::DeviceContextPool::Instance().Get(place));
    auto blas =
        paddle::operators::math::GetBlas<paddle::platform::CUDADeviceContext,
                                         T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }
#else
  void operator()(const paddle::platform::CUDAPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

  // TODO(jiabin): Support Npu here from gradient_accumulator.cc
  // there is NO blas in CUDAPinnedPlace
  void operator()(const paddle::platform::CUDAPinnedPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }

#ifdef PADDLE_WITH_ASCEND_CL
  void operator()(const paddle::platform::NPUPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#else
  void operator()(const paddle::platform::NPUPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

#ifdef PADDLE_WITH_XPU
  void operator()(const paddle::platform::XPUPlace& place) {
    paddle::platform::XPUDeviceContext* ctx =
        dynamic_cast<paddle::platform::XPUDeviceContext*>(
            paddle::platform::DeviceContextPool::Instance().Get(place));
    xpu::add<T>(ctx->x_context(), x_, y_, y_, static_cast<int>(numel_));
  }
#else
  void operator()(const paddle::platform::XPUPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

#ifdef PADDLE_WITH_IPU
  void operator()(const paddle::platform::IPUPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#else
  void operator()(const paddle::platform::IPUPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

  void operator()(const paddle::platform::NPUPinnedPlace& place) {
    PADDLE_THROW(paddle::platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }

 private:
  int64_t numel_;
  const T* x_;
  T* y_;
};

template <typename DeviceContext, typename T>
void TensorAddImpl(const std::shared_ptr<pten::DenseTensor>& src,
                   pten::DenseTensor* dst,
                   const paddle::platform::Place& place) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  paddle::platform::DeviceContext* ctx = pool.Get(place);
  auto dev_ctx = dynamic_cast<DeviceContext*>(ctx);
  paddle::operators::math::ElementwiseAddTo<DeviceContext, T> func;
  func(dev_ctx, *(src.get()), dst);
}

template <typename DeviceContext, typename T>
void TensorAddImpl(const paddle::framework::Tensor& src,
                   paddle::framework::Tensor* dst,
                   const paddle::platform::Place& place) {
  paddle::platform::DeviceContextPool& pool =
      paddle::platform::DeviceContextPool::Instance();
  paddle::platform::DeviceContext* ctx = pool.Get(place);
  auto dev_ctx = dynamic_cast<DeviceContext*>(ctx);
  paddle::operators::math::ElementwiseAddTo<DeviceContext, T> func;
  func(dev_ctx, src, dst);
}

void TensorAdd(const egr::EagerTensor& src, egr::EagerTensor* dst) {
  // TODO(jiabin): Support other tensor type later
  std::shared_ptr<pten::DenseTensor> dst_tensor =
      std::dynamic_pointer_cast<pten::DenseTensor>(dst->impl());
  std::shared_ptr<pten::DenseTensor> src_tensor =
      std::dynamic_pointer_cast<pten::DenseTensor>(src.impl());

  auto numel = src_tensor->numel();

  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      dst_tensor->numel(), numel,
      paddle::platform::errors::PreconditionNotMet(
          "The number of elements of source tensor and destination tensor "
          "should be equal, but got the number of elements of source tensor is "
          "%zu and the number of elements of destination tensor is %zu.",
          numel, dst_tensor->numel()));

  auto data_type = pten::TransToProtoVarType(src_tensor->dtype());
  auto place = src_tensor->place();

  PADDLE_ENFORCE_EQ(pten::TransToProtoVarType(dst_tensor->dtype()), data_type,
                    paddle::platform::errors::PreconditionNotMet(
                        "The data type of source tensor and destination tensor "
                        "should be equal, Otherwise, the calculation results "
                        "will be incorrect."));

#define PADDLE_TENSOR_ADD(cpp_type)                                          \
  if (data_type == paddle::framework::DataTypeTrait<cpp_type>::DataType()) { \
    TensorAddFunctor<cpp_type> func(numel, src_tensor->data<cpp_type>(),     \
                                    dst_tensor->mutable_data<cpp_type>());   \
    boost::apply_visitor(func, place);                                       \
    return;                                                                  \
  }

  // TODO(jiabin): Support NPU here
  PADDLE_TENSOR_ADD(float);
// NOTE(phlrain): xpu only support float
#ifndef PADDLE_WITH_XPU
  PADDLE_TENSOR_ADD(double);
  // NOTE(chenweihang): only support complex grad tensor accumulated,
  // support selected rows if needed in the future
  PADDLE_TENSOR_ADD(paddle::platform::complex<float>);
  PADDLE_TENSOR_ADD(paddle::platform::complex<double>);
#endif
#undef PADDLE_TENSOR_ADD

  if (data_type == paddle::framework::proto::VarType::FP16) {
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      return TensorAddImpl<paddle::platform::CUDADeviceContext,
                           paddle::platform::float16>(src_tensor,
                                                      dst_tensor.get(), place);
#else
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Gradient accumulation of data type (%s) on place (%s) is not "
          "supported in imperative mode",
          paddle::framework::DataTypeToString(data_type), place));
#endif
    } else if (paddle::platform::is_cpu_place(place)) {
      return TensorAddImpl<paddle::platform::CPUDeviceContext,
                           paddle::platform::float16>(src_tensor,
                                                      dst_tensor.get(), place);
    }
  }
  PADDLE_THROW(paddle::platform::errors::Unimplemented(
      "Gradient accumulation of data type (%s) on place (%s) is not "
      "supported in imperative mode",
      paddle::framework::DataTypeToString(data_type), place));
}

void VariableAdd(const egr::EagerTensor& src, egr::EagerTensor* dst) {
  // TODO(jiabin): Support other tensor type later
  auto* dst_tensor =
      dst->MutableVar()->GetMutable<paddle::framework::LoDTensor>();
  auto& src_tensor = src.Var().Get<paddle::framework::LoDTensor>();

  auto numel = src_tensor.numel();

  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      dst_tensor->numel(), numel,
      paddle::platform::errors::PreconditionNotMet(
          "The number of elements of source tensor and destination tensor "
          "should be equal, but got the number of elements of source tensor is "
          "%zu and the number of elements of destination tensor is %zu.",
          numel, dst_tensor->numel()));

  auto data_type = src_tensor.type();
  auto place = src_tensor.place();

  PADDLE_ENFORCE_EQ(dst_tensor->type(), data_type,
                    paddle::platform::errors::PreconditionNotMet(
                        "The data type of source tensor and destination tensor "
                        "should be equal, Otherwise, the calculation results "
                        "will be incorrect."));

#define PADDLE_TENSOR_ADD(cpp_type)                                          \
  if (data_type == paddle::framework::DataTypeTrait<cpp_type>::DataType()) { \
    TensorAddFunctor<cpp_type> func(                                         \
        numel, src_tensor.data<cpp_type>(),                                  \
        dst_tensor->mutable_data<cpp_type>(place));                          \
    boost::apply_visitor(func, place);                                       \
    return;                                                                  \
  }

  // TODO(jiabin): Support NPU here
  PADDLE_TENSOR_ADD(float);
// NOTE(phlrain): xpu only support float
#ifndef PADDLE_WITH_XPU
  PADDLE_TENSOR_ADD(double);
  // NOTE(chenweihang): only support complex grad tensor accumulated,
  // support selected rows if needed in the future
  PADDLE_TENSOR_ADD(paddle::platform::complex<float>);
  PADDLE_TENSOR_ADD(paddle::platform::complex<double>);
#endif
#undef PADDLE_TENSOR_ADD

  if (data_type == paddle::framework::proto::VarType::FP16) {
    if (paddle::platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      return TensorAddImpl<paddle::platform::CUDADeviceContext,
                           paddle::platform::float16>(src_tensor, dst_tensor,
                                                      place);
#else
      PADDLE_THROW(paddle::platform::errors::Unimplemented(
          "Gradient accumulation of data type (%s) on place (%s) is not "
          "supported in imperative mode",
          paddle::framework::DataTypeToString(data_type), place));
#endif
    } else if (paddle::platform::is_cpu_place(place)) {
      return TensorAddImpl<paddle::platform::CPUDeviceContext,
                           paddle::platform::float16>(src_tensor, dst_tensor,
                                                      place);
    }
  }
  PADDLE_THROW(paddle::platform::errors::Unimplemented(
      "Gradient accumulation of data type (%s) on place (%s) is not "
      "supported in imperative mode",
      paddle::framework::DataTypeToString(data_type), place));
}

}  // namespace egr
