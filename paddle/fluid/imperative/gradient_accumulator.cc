// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/imperative/gradient_accumulator.h"

#include <algorithm>
#include <memory>
#include <utility>

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
#include "paddle/fluid/platform/complex.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/float16.h"
#include "paddle/fluid/platform/profiler.h"
#ifdef PADDLE_WITH_XPU
#include "xpu/refactor/math.h"
#endif
#ifdef PADDLE_WITH_ASCEND_CL
#include "paddle/fluid/platform/device/npu/npu_op_runner.h"
#endif

namespace paddle {
namespace imperative {

static void MoveOrCopyVar(framework::Variable* dst, framework::Variable* src,
                          bool force_copy) {
  if (!force_copy) {
    VLOG(6) << "Just Move Variable when sum gradients within this graph";
    *dst = std::move(*src);
    return;
  }

  VLOG(6) << "Copy occurs when sum gradients within this graph";
  if (src->IsType<framework::LoDTensor>()) {
    auto& src_tensor = src->Get<framework::LoDTensor>();
    if (!dst->IsType<framework::LoDTensor>()) {
      dst->Clear();
    }
    auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
    framework::TensorCopy(src_tensor, src_tensor.place(), dst_tensor);
    dst_tensor->set_lod(src_tensor.lod());
  } else if (src->IsType<framework::SelectedRows>()) {
    auto& src_selected_rows = src->Get<framework::SelectedRows>();
    if (!dst->IsType<framework::SelectedRows>()) {
      dst->Clear();
    }
    auto* dst_selected_rows = dst->GetMutable<framework::SelectedRows>();
    framework::TensorCopy(src_selected_rows.value(),
                          src_selected_rows.value().place(),
                          dst_selected_rows->mutable_value());
    dst_selected_rows->set_rows(src_selected_rows.rows());
    dst_selected_rows->set_height(src_selected_rows.height());
  } else {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Only support LoDTensor and SelectedRows for sum gradient"));
  }
}

template <typename T>
class TensorAddFunctor : public boost::static_visitor<> {
 public:
  TensorAddFunctor(int64_t numel, const T* x, T* y)
      : numel_(numel), x_(x), y_(y) {}

  void operator()(const platform::CPUPlace& place) {
    platform::CPUDeviceContext* ctx = dynamic_cast<platform::CPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CPUDeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }

#ifdef PADDLE_WITH_XPU
  void operator()(const platform::XPUPlace& place) {
    using XPUType = typename XPUTypeTrait<T>::Type;
    platform::XPUDeviceContext* ctx = dynamic_cast<platform::XPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    int r = xpu::add<XPUType>(
        ctx->x_context(), reinterpret_cast<const XPUType*>(x_),
        reinterpret_cast<const XPUType*>(y_), reinterpret_cast<XPUType*>(y_),
        static_cast<int>(numel_));
    PADDLE_ENFORCE_EQ(
        r, XPU_SUCCESS,
        platform::errors::External("XPU add kernel return wrong value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
#else
  void operator()(const platform::XPUPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  void operator()(const platform::CUDAPlace& place) {
    platform::CUDADeviceContext* ctx =
        dynamic_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CUDADeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }
#else
  void operator()(const platform::CUDAPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

#ifdef PADDLE_WITH_MLU
  void operator()(const platform::MLUPlace& place) {
    // TODO(fwg): SUPPORT it
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#else
  void operator()(const platform::MLUPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

#ifdef PADDLE_WITH_ASCEND_CL
  void operator()(const platform::NPUPlace& place) {
    // TODO(zhiqiu): SUPPORT it
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#else
  void operator()(const platform::NPUPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
#endif

  void operator()(const platform::NPUPinnedPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
  // there is NO blas in CUDAPinnedPlace
  void operator()(const platform::CUDAPinnedPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }
  // there is NO support in IPUPlace
  void operator()(const platform::IPUPlace& place) {
    PADDLE_THROW(platform::errors::PermissionDenied(
        "Gradient accumulation on place (%s) "
        "is not supported in imperative mode",
        place));
  }

 private:
  int64_t numel_;
  const T* x_;
  T* y_;
};

#ifdef PADDLE_WITH_XPU
template <typename T>
void XPUTensorAddFunctor(const platform::Place& place,
                         const framework::Tensor& src, framework::Tensor* dst) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  platform::XPUDeviceContext* ctx = dynamic_cast<platform::XPUDeviceContext*>(
      platform::DeviceContextPool::Instance().Get(place));
  const XPUType* x = reinterpret_cast<const XPUType*>(src.data<T>());
  XPUType* y = reinterpret_cast<XPUType*>(dst->mutable_data<T>(place));
  int r = xpu::add<XPUType>(ctx->x_context(), x, y, y,
                            static_cast<int>(src.numel()));
  PADDLE_ENFORCE_EQ(
      r, XPU_SUCCESS,
      platform::errors::External("XPU add kernel return wrong value[%d %s]", r,
                                 XPUAPIErrorMsg[r]));
}
#endif

template <typename DeviceContext, typename T>
void TensorAddImpl(const framework::Tensor& src, framework::Tensor* dst,
                   const platform::Place& place) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  paddle::platform::DeviceContext* ctx = pool.Get(place);
  auto dev_ctx = dynamic_cast<DeviceContext*>(ctx);
  operators::math::ElementwiseAddTo<DeviceContext, T> func;
  func(dev_ctx, src, dst);
}

void TensorAdd(const framework::Variable& src, framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_tensor = src.Get<framework::LoDTensor>();

  auto numel = src_tensor.numel();

  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      dst_tensor->numel(), numel,
      platform::errors::PreconditionNotMet(
          "The number of elements of source tensor and destination tensor "
          "should be equal, but got the number of elements of source tensor is "
          "%zu and the number of elements of destination tensor is %zu.",
          numel, dst_tensor->numel()));

  auto data_type = src_tensor.type();
  auto place = src_tensor.place();

  PADDLE_ENFORCE_EQ(dst_tensor->type(), data_type,
                    platform::errors::PreconditionNotMet(
                        "The data type of source tensor and destination tensor "
                        "should be equal, Otherwise, the calculation results "
                        "will be incorrect."));

#define PADDLE_TENSOR_ADD(cpp_type)                                  \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) { \
    TensorAddFunctor<cpp_type> func(                                 \
        numel, src_tensor.data<cpp_type>(),                          \
        dst_tensor->mutable_data<cpp_type>(place));                  \
    boost::apply_visitor(func, place);                               \
    return;                                                          \
  }

#ifdef PADDLE_WITH_ASCEND_CL
  if (platform::is_npu_place(place)) {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    platform::DeviceContext* ctx = pool.Get(place);
    auto dev_ctx = dynamic_cast<platform::NPUDeviceContext*>(ctx);
    if (data_type == framework::DataTypeTrait<float>::DataType()) {
      dst_tensor->mutable_data<float>(place);
    } else if (data_type == framework::DataTypeTrait<double>::DataType()) {
      dst_tensor->mutable_data<double>(place);
    } else if (data_type ==
               framework::DataTypeTrait<platform::float16>::DataType()) {
      dst_tensor->mutable_data<platform::float16>(place);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Gradient accumulation of data type (%s) on place (%s) is not "
          "supported in imperative mode",
          framework::DataTypeToString(data_type), place));
    }
    const auto& runner = operators::NpuOpRunner(
        "Add", {*dst_tensor, src_tensor}, {*dst_tensor}, {});
    runner.Run(dev_ctx->stream());
    return;
  }
#endif

#ifdef PADDLE_WITH_XPU
  if (platform::is_xpu_place(place)) {
    if (data_type == framework::DataTypeTrait<float>::DataType()) {
      XPUTensorAddFunctor<float>(place, src_tensor, dst_tensor);
    } else if (data_type ==
               framework::DataTypeTrait<platform::float16>::DataType()) {
      XPUTensorAddFunctor<platform::float16>(place, src_tensor, dst_tensor);
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Gradient accumulation of data type (%s) on place (%s) is not "
          "supported in imperative mode",
          framework::DataTypeToString(data_type), place));
    }
    return;
  }
#endif

  PADDLE_TENSOR_ADD(float);

#ifndef PADDLE_WITH_XPU
  // NOTE(phlrain): xpu only support float
  PADDLE_TENSOR_ADD(double);
  // NOTE(chenweihang): only support complex grad tensor accumulated,
  // support selected rows if needed in the future
  PADDLE_TENSOR_ADD(platform::complex<float>);
  PADDLE_TENSOR_ADD(platform::complex<double>);
#endif

#undef PADDLE_TENSOR_ADD

  if (data_type == framework::proto::VarType::FP16) {
    if (platform::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      return TensorAddImpl<platform::CUDADeviceContext, platform::float16>(
          src_tensor, dst_tensor, place);
#else
      PADDLE_THROW(platform::errors::Unimplemented(
          "Gradient accumulation of data type (%s) on place (%s) is not "
          "supported in imperative mode",
          framework::DataTypeToString(data_type), place));
#endif
    } else if (platform::is_cpu_place(place)) {
      return TensorAddImpl<platform::CPUDeviceContext, platform::float16>(
          src_tensor, dst_tensor, place);
    }
  }
  PADDLE_THROW(platform::errors::Unimplemented(
      "Gradient accumulation of data type (%s) on place (%s) is not "
      "supported in imperative mode",
      framework::DataTypeToString(data_type), place));
}

void SelectedRowsAddToTensor(const framework::Variable& src,
                             framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_selected_rows = src.Get<framework::SelectedRows>();
  auto place = dst_tensor->place();
  auto data_type = src_selected_rows.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

#define PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(dev_ctx_type, cpp_type)           \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {         \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);              \
    paddle::operators::math::SelectedRowsAddToTensor<dev_ctx_type, cpp_type> \
        functor;                                                             \
    functor(*(dynamic_cast<dev_ctx_type*>(dev_ctx)), src_selected_rows,      \
            dst_tensor);                                                     \
    return;                                                                  \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (paddle::platform::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CUDADeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CUDADeviceContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CPUDeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(platform::CPUDeviceContext, double);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif

#undef PADDLE_SELECTED_ROWS_ADD_TO_TENSOR

  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsAddToTensor",
      framework::DataTypeToString(data_type)));
}

static void SelectedRowsAddTensor(
    const framework::Variable& src_selected_rows_var,
    const framework::Variable& src_tensor_var,
    framework::Variable* dst_tensor_var) {
  const auto& src_selected_rows =
      src_selected_rows_var.Get<framework::SelectedRows>();
  const auto& src_tensor = src_tensor_var.Get<framework::LoDTensor>();
  const auto& place = src_tensor.place();
  auto data_type = src_tensor.type();
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);

  auto* dst_tensor = dst_tensor_var->GetMutable<framework::LoDTensor>();
  dst_tensor->Resize(src_tensor.dims());
  dst_tensor->mutable_data(place, data_type);

#define PADDLE_SELECTED_ROWS_ADD_TENSOR(dev_ctx_type, cpp_type)            \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {       \
    paddle::operators::math::SelectedRowsAddTensor<dev_ctx_type, cpp_type> \
        functor;                                                           \
    functor(*(dynamic_cast<dev_ctx_type*>(dev_ctx)), src_selected_rows,    \
            src_tensor, dst_tensor);                                       \
    return;                                                                \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (platform::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD_TENSOR(platform::CUDADeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD_TENSOR(platform::CUDADeviceContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD_TENSOR(platform::CPUDeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD_TENSOR(platform::CPUDeviceContext, double);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif

  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsAddToTensor",
      framework::DataTypeToString(data_type)));

#undef PADDLE_SELECTED_ROWS_ADD_TENSOR
}

// Note(chenweihang): when two selected rows need to be added,
//   adding one to another is not equal to merging two selected rows
//   to one then add it to a empty selected rows, the after is correct
std::shared_ptr<VariableWrapper> SelectedRowsMerge(
    const framework::Variable& src1, const framework::Variable& src2) {
  auto& src_selected_rows1 = src1.Get<framework::SelectedRows>();
  auto& src_selected_rows2 = src2.Get<framework::SelectedRows>();
  auto place = src_selected_rows1.value().place();
  auto data_type = src_selected_rows1.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  std::vector<const framework::SelectedRows*> src_selected_rows;
  src_selected_rows.emplace_back(&src_selected_rows1);
  src_selected_rows.emplace_back(&src_selected_rows2);
  auto dst_var = std::make_shared<VariableWrapper>("Temp");
  auto* dst_selected_rows =
      dst_var->MutableVar()->GetMutable<framework::SelectedRows>();

#define PADDLE_SELECTED_ROWS_ADD(dev_ctx_type, cpp_type)                  \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {      \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);           \
    paddle::operators::math::scatter::MergeAdd<dev_ctx_type, cpp_type>    \
        merge_add;                                                        \
    merge_add(*(dynamic_cast<dev_ctx_type*>(dev_ctx)), src_selected_rows, \
              dst_selected_rows);                                         \
    return dst_var;                                                       \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (paddle::platform::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD(platform::CUDADeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD(platform::CUDADeviceContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD(platform::CPUDeviceContext, float);
    PADDLE_SELECTED_ROWS_ADD(platform::CPUDeviceContext, double);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif

#undef PADDLE_SELECTED_ROWS_ADD

  PADDLE_THROW(platform::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsMerge",
      framework::DataTypeToString(data_type)));
}

void VariableWrapperAdd(std::shared_ptr<VariableWrapper> var,
                        VariableWrapper* dst_var, bool unchange_input) {
  auto& src = var->Var();
  auto* dst = dst_var->MutableVar();
  if (dst->IsType<framework::LoDTensor>()) {
    if (src.IsType<framework::LoDTensor>()) {
      TensorAdd(src, dst);
    } else if (src.IsType<framework::SelectedRows>()) {
      SelectedRowsAddToTensor(src, dst);
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unexpected branch, output variable type is %s",
          framework::ToTypeName(dst->Type())));
    }
  } else {
    if (src.IsType<framework::LoDTensor>()) {
      if (unchange_input) {
        framework::Variable new_dst;
        SelectedRowsAddTensor(*dst, src, &new_dst);
        *dst = std::move(new_dst);
      } else {
        auto* src_mutable = var->MutableVar();
        SelectedRowsAddToTensor(*dst, src_mutable);
        *dst = std::move(*(var->MutableVar()));
      }
    } else if (src.IsType<framework::SelectedRows>()) {
      auto temp = SelectedRowsMerge(src, *dst);
      *dst = std::move(*(temp->MutableVar()));
    } else {
      PADDLE_THROW(platform::errors::InvalidArgument(
          "Unexpected branch, output variable type is %s",
          framework::ToTypeName(dst->Type())));
    }
  }
}

static platform::Place GetPlaceOfVar(
    const std::shared_ptr<VariableWrapper>& var) {
  platform::Place place;
  if (var->Var().IsType<framework::LoDTensor>()) {
    place = var->Var().Get<framework::LoDTensor>().place();
  } else if (var->Var().IsType<framework::SelectedRows>()) {
    place = var->Var().Get<framework::SelectedRows>().place();
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "only support LoDTensor and SelectedRows in dygraph"));
  }
  return place;
}

void GradientAccumulator::AccumulateGrad() {
  /**
   * If the leaf gradient has been calculated done, the inner_var_
   * should be added to the var_.
   */
  if (!var_->IsLeafGrad() || !SumGradCompleted() || !HasInnerVar()) {
    return;
  }
  PADDLE_ENFORCE_EQ(HasInnerVar(), true,
                    platform::errors::InvalidArgument(
                        "Leaf tensor should have inner var to store results of "
                        "this auto-grad"));
  PADDLE_ENFORCE_EQ(inner_var_->Var().IsInitialized(), true,
                    platform::errors::InvalidArgument(
                        "Interior var of Leaf tensor should be initialized."));
  auto* src = inner_var_->MutableVar();
  auto* dst = var_->MutableVar();
  if (!var_->IsEmpty()) {
    VLOG(6) << "Leaf Var(" << var_->Name()
            << ")'s Gradient has been initizlized, will accumulate on "
               "previous gradient.";
    if (dst->IsType<framework::LoDTensor>()) {
      if (src->IsType<framework::LoDTensor>()) {
        TensorAdd(*src, dst);
      } else if (src->IsType<framework::SelectedRows>()) {
        SelectedRowsAddToTensor(*src, dst);
      }
    } else if (dst->IsType<framework::SelectedRows>()) {
      if (src->IsType<framework::LoDTensor>()) {
        SelectedRowsAddToTensor(*dst, src);
        *dst = std::move(*src);
      } else if (src->IsType<framework::SelectedRows>()) {
        auto temp = SelectedRowsMerge(*src, *dst);
        *dst = std::move(*(temp->MutableVar()));
      }
    } else {
      PADDLE_THROW(platform::errors::PermissionDenied(
          "Only support LoDTensor and SelectedRows for gradient var"));
    }
  } else {
    VLOG(6)
        << "Leaf Var(" << var_->Name()
        << ")'s Gradient has not been initialized, not accumulate. Just move";
    *(dst) = std::move(*src);
    var_->SetType(inner_var_->Type());
    var_->SetDataType(inner_var_->DataType());
    var_->SetIsEmpty(false);
  }
  inner_var_.reset();
}

void GradientAccumulator::CallGradientHooks() {
  PADDLE_ENFORCE_EQ(var_->IsLeafGrad(), true,
                    platform::errors::Unavailable(
                        "Only leaf gradient Tensor can deal with by gradient "
                        "hook in gradient accumulator."));
  PADDLE_ENFORCE_EQ(
      SumGradCompleted(), true,
      platform::errors::PreconditionNotMet(
          "Only can call gradient hooks after sum gradient completed."));
  PADDLE_ENFORCE_EQ(
      HasInnerVar(), true,
      platform::errors::PreconditionNotMet(
          "Leaf Tensor's inner var is nullptr when call gradient hook."));
  PADDLE_ENFORCE_EQ(
      inner_var_->Var().IsInitialized(), true,
      platform::errors::PreconditionNotMet("Leaf Tensor's inner var "
                                           "is not initialized when "
                                           "call gradient hook."));
  if (var_->HasVariableWrapperHook()) {
    VLOG(3) << "Call " << var_->GetVariableWrapperHooks().size()
            << " hooks of leaf gradient accumulator's inner var `"
            << var_->Name() << "`.";
    auto tmp_var = inner_var_;
    VLOG(3) << "Input var " << var_->Name() << "'s hook size - "
            << var_->GetVariableWrapperHooks().size();
    for (const auto& hook_pair : var_->GetVariableWrapperHooks()) {
      tmp_var = (*hook_pair.second)(tmp_var);
    }
    inner_var_ = tmp_var;
  }
}

void GradientAccumulator::CallReduceHooks() {
  PADDLE_ENFORCE_EQ(
      var_->IsLeafGrad(), true,
      platform::errors::Unavailable("Only leaf gradient Tensor can deal with "
                                    "by reduce hook in gradient accumulator."));
  PADDLE_ENFORCE_EQ(SumGradCompleted(), true,
                    platform::errors::PreconditionNotMet(
                        "Only can call reduce hooks after the gradient "
                        "summation is completed in current batch."));
  PADDLE_ENFORCE_EQ(HasInnerVar(), false,
                    platform::errors::PreconditionNotMet(
                        "Only can call reduce hooks after the "
                        "gradient accumulation is completed in "
                        "current batch or across batchs."));
  if (var_->HasVoidHook()) {
    for (const auto& hook : var_->GetVoidHooks()) {
      VLOG(3) << "call gradient accumulator backward hooks.";
      (*hook)();
    }
  }
}

void EagerGradientAccumulator::SumGrad(std::shared_ptr<VariableWrapper> var,
                                       size_t trace_id, bool unchange_input) {
  /**
   * If var has grad node, it indicates that this var would be an input
   * of a grad op. Therefore, it should not be changed.
   */
  if (var->HasGradNode()) {
    unchange_input = true;
  }

  auto* dst_var = Var();
  platform::Place place = GetPlaceOfVar(var);
  if (!dst_var->OverridedStopGradient()) {
    if (CurCnt() == 0) {
      MoveOrCopyVar(dst_var->MutableVar(), var->MutableVar(), unchange_input);
    } else {
      VLOG(6) << "Sum Gradient for: " << dst_var->Name()
              << " within this graph.";
      VariableWrapperAdd(var, dst_var, unchange_input);
    }
  } else {
    if (!dst_var->Var().IsInitialized() ||
        !dst_var->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << dst_var->Name() << " as zero ";
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      if (!dst_var->Var().IsInitialized()) {
        auto* tensor =
            dst_var->MutableVar()->GetMutable<framework::LoDTensor>();
        VLOG(6) << "Dims of " << dst_var->Name() << " is set as: "
                << var->Var().Get<framework::LoDTensor>().dims();
        tensor->Resize(var->Var().Get<framework::LoDTensor>().dims());
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      } else {
        auto* tensor =
            dst_var->MutableVar()->GetMutable<framework::LoDTensor>();
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
  }

  // Type may be changed after OP run, such as VarTypeInference
  // so synchronous VariableWrapper with Variable.
  if (dst_var->Var().IsType<framework::LoDTensor>()) {
    dst_var->SetType(framework::proto::VarType::LOD_TENSOR);
  } else if (dst_var->Var().IsType<framework::SelectedRows>()) {
    dst_var->SetType(framework::proto::VarType::SELECTED_ROWS);
  }

  // Increase curent count
  IncreaseCurCnt();
}

void SortedGradientAccumulator::SumGrad(std::shared_ptr<VariableWrapper> var,
                                        size_t trace_id, bool unchange_input) {
  auto* dst_var = Var();
  platform::Place place = GetPlaceOfVar(var);
  if (!dst_var->OverridedStopGradient()) {
    if (ref_cnt_ == 1) {
      MoveOrCopyVar(dst_var->MutableVar(), var->MutableVar(),
                    unchange_input || var->HasGradNode());
    } else {
      if (tmp_grad_vars_.empty()) {
        tmp_grad_vars_.reserve(ref_cnt_);
      }

      tmp_grad_vars_.emplace_back(std::move(var), trace_id, unchange_input);

      if (tmp_grad_vars_.size() != ref_cnt_) {
        return;
      }

      VLOG(6) << "Sum Gradient for: " << dst_var->Name()
              << " within this graph.";
      std::sort(tmp_grad_vars_.begin(), tmp_grad_vars_.end(),
                [](const SavedVarInfo& info1, const SavedVarInfo& info2) {
                  return info1.trace_id > info2.trace_id;
                });

      for (auto& var_info : tmp_grad_vars_) {
        if (var_info.var->HasGradNode()) {
          var_info.unchange_input = true;
        }
      }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (paddle::platform::is_gpu_place(place)) {
        // sum selected rows firstly
        for (auto& var_info : tmp_grad_vars_) {
          if (!var_info.var->Var().IsType<framework::SelectedRows>()) {
            continue;
          }

          if (CurCnt() == 0) {
            MoveOrCopyVar(dst_var->MutableVar(), var_info.var->MutableVar(),
                          var_info.unchange_input);
          } else {
            VariableWrapperAdd(var_info.var, dst_var, var_info.unchange_input);
          }

          var_info.var = nullptr;
          // Increase count
          IncreaseCurCnt();
        }

        for (auto& var_info : tmp_grad_vars_) {
          if (!var_info.var) {
            continue;
          }

          PADDLE_ENFORCE_EQ(var_info.var->Var().IsType<framework::LoDTensor>(),
                            true, platform::errors::PermissionDenied(
                                      "Gradient var must be LoDTensor"));
          if (CurCnt() == 0) {
            MoveOrCopyVar(dst_var->MutableVar(), var_info.var->MutableVar(),
                          var_info.unchange_input);
          } else {
            VariableWrapperAdd(var_info.var, dst_var, var_info.unchange_input);
          }

          var_info.var = nullptr;
          // Increase count
          IncreaseCurCnt();
        }
      } else {
#endif
        for (auto& var_info : tmp_grad_vars_) {
          if (!var_info.var) {
            continue;
          }
          PADDLE_ENFORCE_EQ(
              var_info.var->Var().IsType<framework::LoDTensor>() ||
                  var_info.var->Var().IsType<framework::SelectedRows>(),
              true, platform::errors::PermissionDenied("The type of Gradient "
                                                       "var must be LoDTensor "
                                                       "or SelectedRows"));
          if (CurCnt() == 0) {
            MoveOrCopyVar(dst_var->MutableVar(), var_info.var->MutableVar(),
                          var_info.unchange_input);
          } else {
            VariableWrapperAdd(var_info.var, dst_var, var_info.unchange_input);
          }
          var_info.var = nullptr;
          // Increase count
          IncreaseCurCnt();
        }
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      }
#endif
      tmp_grad_vars_.clear();
    }
  } else {
    if (!dst_var->Var().IsInitialized() ||
        !dst_var->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var->Name() << " as zero";
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      if (!dst_var->Var().IsInitialized()) {
        auto* tensor =
            dst_var->MutableVar()->GetMutable<framework::LoDTensor>();
        VLOG(6) << "Dims of " << dst_var->Name() << " is set as: "
                << var->Var().Get<framework::LoDTensor>().dims();
        tensor->Resize(var->Var().Get<framework::LoDTensor>().dims());
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      } else {
        auto* tensor =
            dst_var->MutableVar()->GetMutable<framework::LoDTensor>();
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
    // looks like tmp_grad_vars will not have any member but just in case
    tmp_grad_vars_.clear();
  }

  if (dst_var->Var().IsType<framework::LoDTensor>()) {
    dst_var->SetType(framework::proto::VarType::LOD_TENSOR);
  } else if (dst_var->Var().IsType<framework::SelectedRows>()) {
    dst_var->SetType(framework::proto::VarType::SELECTED_ROWS);
  }
}

}  // namespace imperative
}  // namespace paddle
