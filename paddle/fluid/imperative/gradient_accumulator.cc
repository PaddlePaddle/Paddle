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

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows_utils.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/phi/common/bfloat16.h"
#include "paddle/phi/common/complex.h"
#include "paddle/phi/common/float16.h"
#include "paddle/phi/core/platform/device_context.h"
#include "paddle/phi/core/platform/profiler.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/selected_rows_functor.h"
#ifdef PADDLE_WITH_XPU
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "xpu/refactor/math.h"
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include "paddle/phi/backends/device_manager.h"
#endif
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

namespace paddle::imperative {

static void MoveOrCopyVar(framework::Variable* dst,
                          framework::Variable* src,
                          bool force_copy) {
  if (!force_copy) {
    VLOG(6) << "Just Move Variable when sum gradients within this graph";
    *dst = std::move(*src);
    return;
  }

  VLOG(6) << "Copy occurs when sum gradients within this graph";
  if (src->IsType<phi::DenseTensor>()) {
    auto& src_tensor = src->Get<phi::DenseTensor>();
    if (!dst->IsType<phi::DenseTensor>()) {
      dst->Clear();
    }
    auto* dst_tensor = dst->GetMutable<phi::DenseTensor>();
    framework::TensorCopy(src_tensor, src_tensor.place(), dst_tensor);
    dst_tensor->set_lod(src_tensor.lod());
  } else if (src->IsType<phi::SelectedRows>()) {
    auto& src_selected_rows = src->Get<phi::SelectedRows>();
    if (!dst->IsType<phi::SelectedRows>()) {
      dst->Clear();
    }
    auto* dst_selected_rows = dst->GetMutable<phi::SelectedRows>();
    framework::TensorCopy(src_selected_rows.value(),
                          src_selected_rows.value().place(),
                          dst_selected_rows->mutable_value());
    dst_selected_rows->set_rows(src_selected_rows.rows());
    dst_selected_rows->set_height(src_selected_rows.height());
  } else {
    PADDLE_THROW(common::errors::PermissionDenied(
        "Only support DenseTensor and SelectedRows for sum gradient"));
  }
}

#ifdef PADDLE_WITH_XPU
template <typename T>
void XPUTensorAddFunctor(const phi::Place& place,
                         const phi::DenseTensor& src,
                         phi::DenseTensor* dst) {
  using XPUType = typename XPUTypeTrait<T>::Type;
  phi::XPUContext* ctx = dynamic_cast<phi::XPUContext*>(
      phi::DeviceContextPool::Instance().Get(place));
  const XPUType* x = reinterpret_cast<const XPUType*>(src.data<T>());
  XPUType* y = reinterpret_cast<XPUType*>(dst->mutable_data<T>(place));
  int r = -1;
  int numel = static_cast<int>(src.numel());
  if (std::is_same<T, double>::value) {
    xpu::ctx_guard RAII_GUARD(ctx->x_context());
    float* x_cast_to_fp32 = RAII_GUARD.alloc<float>(numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(x_cast_to_fp32);
    float* y_cast_to_fp32 = RAII_GUARD.alloc<float>(numel);
    PADDLE_ENFORCE_XDNN_NOT_NULL(y_cast_to_fp32);
    r = xpu::cast<XPUType, float>(ctx->x_context(), x, x_cast_to_fp32, numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    r = xpu::cast<XPUType, float>(ctx->x_context(), y, y_cast_to_fp32, numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
    r = xpu::add<float>(ctx->x_context(),
                        x_cast_to_fp32,
                        y_cast_to_fp32,
                        y_cast_to_fp32,
                        numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
    r = xpu::cast<float, XPUType>(ctx->x_context(), y_cast_to_fp32, y, numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "cast");
  } else {
    r = xpu::add<XPUType>(ctx->x_context(), x, y, y, numel);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "add");
  }
}
#endif

template <typename TType>
TType* GetInnerMutableTensor(framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<TType>();
  return dst_tensor;
}

template <typename TType>
TType* GetInnerMutableTensor(paddle::Tensor* dst) {
  auto* dst_tensor = static_cast<TType*>(dst->impl().get());
  return dst_tensor;
}

template <typename TType>
const TType& GetInnerTensor(const framework::Variable& src) {
  return src.Get<TType>();
}

template <typename TType>
TType& GetInnerTensor(const paddle::Tensor& src) {
  PADDLE_ENFORCE_EQ(
      (src.has_allocation()),
      true,
      common::errors::Fatal("We only add tensor with value if a tensor is "
                            "NOT INITIALIZED, it should just move instead of "
                            "calling this method."));
  auto* src_tensor = static_cast<TType*>(src.impl().get());
  return *src_tensor;
}

template <typename TType>
TType* GetEmptyInnerTensor(paddle::Tensor* dst) {
  PADDLE_ENFORCE_EQ(
      dst->defined(),
      false,
      common::errors::Fatal(
          "The underlying Tensor implementation should be nullptr"));
  dst->set_impl(std::make_shared<TType>());
  auto* dst_tensor = static_cast<TType*>(dst->impl().get());
  return dst_tensor;
}

template <typename TType>
TType* GetEmptyInnerTensor(paddle::imperative::VariableWrapper* dst) {
  auto* dst_tensor = dst->MutableVar()->GetMutable<TType>();
  return dst_tensor;
}

template <typename VarType>
void TensorAdd(const VarType& src, VarType* dst) {
  phi::DenseTensor* dst_tensor = GetInnerMutableTensor<phi::DenseTensor>(dst);
  const phi::DenseTensor& src_tensor = GetInnerTensor<phi::DenseTensor>(src);

  paddle::experimental::CheckAndTrans2Contiguous(
      const_cast<phi::DenseTensor*>(&src_tensor));
  paddle::experimental::CheckAndTrans2Contiguous(dst_tensor);

  auto numel = src_tensor.numel();

  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(
      dst_tensor->numel(),
      numel,
      common::errors::PreconditionNotMet(
          "The number of elements of source tensor and destination tensor "
          "should be equal, but got the number of elements of source tensor is "
          "%zu and the number of elements of destination tensor is %zu.",
          numel,
          dst_tensor->numel()));

  auto data_type = framework::TransToProtoVarType(src_tensor.dtype());
  auto place = src_tensor.place();

  // if src and dst are in different place, copy dst to src's place
  if (dst_tensor->place() != place) {
    paddle::framework::TensorCopySync(*dst_tensor, place, dst_tensor);
  }

  // AddKernel already support inputs of different dtype. For AMP master_grad,
  // the dtype of source tensor and destination tensor will be different. So the
  // check requiring input dtypes to be the same have been removed.
#define PADDLE_TENSOR_ADD(T, CONTEXT)                                          \
  if (data_type == framework::DataTypeTrait<T>::DataType()) {                  \
    auto cpu_ctx =                                                             \
        static_cast<CONTEXT*>(phi::DeviceContextPool::Instance().Get(place));  \
    phi::AddKernel<T, CONTEXT>(*cpu_ctx, *dst_tensor, src_tensor, dst_tensor); \
    return;                                                                    \
  }

  if (phi::is_gpu_place(place)) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    PADDLE_TENSOR_ADD(float, phi::GPUContext);
    PADDLE_TENSOR_ADD(double, phi::GPUContext);
    PADDLE_TENSOR_ADD(phi::dtype::float16, phi::GPUContext);
    PADDLE_TENSOR_ADD(phi::dtype::bfloat16, phi::GPUContext);
    PADDLE_TENSOR_ADD(phi::dtype::complex<float>, phi::GPUContext);
    PADDLE_TENSOR_ADD(phi::dtype::complex<double>, phi::GPUContext);
#endif
  }

#define TENSOR_ADD_EIGEN(T)                             \
  auto cpu_ctx = static_cast<phi::CPUContext*>(         \
      phi::DeviceContextPool::Instance().Get(place));   \
  auto in = phi::EigenVector<T>::Flatten(src_tensor);   \
  auto out = phi::EigenVector<T>::Flatten(*dst_tensor); \
  auto& p = *(cpu_ctx->eigen_device());                 \
  out.device(p) = out + in;                             \
  return;

  if (phi::is_cpu_place(place)) {
    PADDLE_TENSOR_ADD(float, phi::CPUContext);
    PADDLE_TENSOR_ADD(double, phi::CPUContext);
    PADDLE_TENSOR_ADD(phi::dtype::complex<float>, phi::CPUContext);
    PADDLE_TENSOR_ADD(phi::dtype::complex<double>, phi::CPUContext);
    if (data_type == framework::proto::VarType::BF16) {
      TENSOR_ADD_EIGEN(phi::dtype::bfloat16);
    }
    if (data_type == framework::proto::VarType::FP16) {
      TENSOR_ADD_EIGEN(phi::dtype::float16);
    }
  }

#define PADDLE_TENSOR_ADD_CUSTOM(T)                              \
  if (data_type == framework::DataTypeTrait<T>::DataType()) {    \
    phi::CustomContext* ctx = static_cast<phi::CustomContext*>(  \
        phi::DeviceContextPool::Instance().Get(place));          \
    phi::stream::Stream stream(place, ctx->stream());            \
    auto device = phi::DeviceManager::GetDeviceWithPlace(place); \
    device->BlasAXPBY<T>(stream,                                 \
                         static_cast<size_t>(numel),             \
                         1.,                                     \
                         src_tensor.data<T>(),                   \
                         1.,                                     \
                         dst_tensor->mutable_data<T>(place));    \
    return;                                                      \
  }

  if (phi::is_custom_place(place)) {
#if defined(PADDLE_WITH_CUSTOM_DEVICE)
    PADDLE_TENSOR_ADD_CUSTOM(float);
    PADDLE_TENSOR_ADD_CUSTOM(double);
    PADDLE_TENSOR_ADD_CUSTOM(phi::dtype::complex<float>);
    PADDLE_TENSOR_ADD_CUSTOM(phi::dtype::complex<double>);
#endif
  }

#ifdef PADDLE_WITH_XPU
  if (phi::is_xpu_place(place)) {
    if (data_type == framework::DataTypeTrait<float>::DataType()) {
      XPUTensorAddFunctor<float>(place, src_tensor, dst_tensor);
    } else if (data_type ==
               framework::DataTypeTrait<phi::dtype::float16>::DataType()) {
      XPUTensorAddFunctor<phi::dtype::float16>(place, src_tensor, dst_tensor);
    } else if (data_type == framework::DataTypeTrait<double>::DataType()) {
      XPUTensorAddFunctor<double>(place, src_tensor, dst_tensor);
    } else if (data_type ==
               framework::DataTypeTrait<phi::dtype::bfloat16>::DataType()) {
      XPUTensorAddFunctor<phi::dtype::bfloat16>(place, src_tensor, dst_tensor);
    } else {
      PADDLE_THROW(common::errors::Unimplemented(
          "Gradient accumulation of data type (%s) on place (%s) is not "
          "supported in imperative mode",
          framework::DataTypeToString(data_type),
          place));
    }
    return;
  }
#endif

  PADDLE_THROW(common::errors::Unimplemented(
      "Gradient accumulation of data type (%s) on place (%s) is not "
      "supported in imperative mode",
      framework::DataTypeToString(data_type),
      place));
}

template void TensorAdd<framework::Variable>(const framework::Variable& src,
                                             framework::Variable* dst);
template void TensorAdd<paddle::Tensor>(const paddle::Tensor& src,
                                        paddle::Tensor* dst);

template <typename VarType>
void SelectedRowsAddToTensor(const VarType& src, VarType* dst) {
  phi::DenseTensor* dst_tensor = GetInnerMutableTensor<phi::DenseTensor>(dst);
  const phi::SelectedRows& src_selected_rows =
      GetInnerTensor<phi::SelectedRows>(src);

  paddle::experimental::CheckAndTrans2Contiguous(
      const_cast<phi::SelectedRows*>(&src_selected_rows)->mutable_value());
  paddle::experimental::CheckAndTrans2Contiguous(dst_tensor);

  auto place = dst_tensor->place();
  auto data_type =
      framework::TransToProtoVarType(src_selected_rows.value().dtype());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();

#define PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(dev_ctx_type, cpp_type)       \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {     \
    phi::DeviceContext* dev_ctx = pool.Get(place);                       \
    phi::funcs::SelectedRowsAddToTensor<dev_ctx_type, cpp_type> functor; \
    functor(*(dynamic_cast<dev_ctx_type*>(dev_ctx)),                     \
            src_selected_rows,                                           \
            dst_tensor);                                                 \
    return;                                                              \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(phi::GPUContext, float);
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(phi::GPUContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(phi::CPUContext, float);
    PADDLE_SELECTED_ROWS_ADD_TO_TENSOR(phi::CPUContext, double);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif

#undef PADDLE_SELECTED_ROWS_ADD_TO_TENSOR

  PADDLE_THROW(common::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsAddToTensor",
      framework::DataTypeToString(data_type)));
}

template void SelectedRowsAddToTensor(const framework::Variable& src,
                                      framework::Variable* dst);
template void SelectedRowsAddToTensor(const paddle::Tensor& src,
                                      paddle::Tensor* dst);

template <typename VarType>
void SelectedRowsAddTensor(const VarType& src_selected_rows_var,
                           const VarType& src_tensor_var,
                           VarType* dst_tensor_var) {
  const phi::SelectedRows& src_selected_rows =
      GetInnerTensor<phi::SelectedRows>(src_selected_rows_var);
  const phi::DenseTensor& src_tensor =
      GetInnerTensor<phi::DenseTensor>(src_tensor_var);

  paddle::experimental::CheckAndTrans2Contiguous(
      const_cast<phi::SelectedRows*>(&src_selected_rows)->mutable_value());

  const auto& place = src_tensor.place();
  auto data_type = framework::TransToProtoVarType(src_tensor.dtype());
  auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);

  phi::DenseTensor* dst_tensor =
      GetInnerMutableTensor<phi::DenseTensor>(dst_tensor_var);
  dst_tensor->Resize(src_tensor.dims());
  dst_tensor->mutable_data(place, src_tensor.dtype());

#define PADDLE_SELECTED_ROWS_ADD_TENSOR(dev_ctx_type, cpp_type)        \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {   \
    phi::funcs::SelectedRowsAddTensor<dev_ctx_type, cpp_type> functor; \
    functor(*(dynamic_cast<dev_ctx_type*>(dev_ctx)),                   \
            src_selected_rows,                                         \
            src_tensor,                                                \
            dst_tensor);                                               \
    return;                                                            \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD_TENSOR(phi::GPUContext, float);
    PADDLE_SELECTED_ROWS_ADD_TENSOR(phi::GPUContext, double);
  } else {
#endif
    PADDLE_SELECTED_ROWS_ADD_TENSOR(phi::CPUContext, float);
    PADDLE_SELECTED_ROWS_ADD_TENSOR(phi::CPUContext, double);
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif

  PADDLE_THROW(common::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsAddToTensor",
      framework::DataTypeToString(data_type)));

#undef PADDLE_SELECTED_ROWS_ADD_TENSOR
}

template void SelectedRowsAddTensor(
    const framework::Variable& src_selected_rows_var,
    const framework::Variable& src_tensor_var,
    framework::Variable* dst_tensor_var);
template void SelectedRowsAddTensor(const paddle::Tensor& src_selected_rows_var,
                                    const paddle::Tensor& src_tensor_var,
                                    paddle::Tensor* dst_tensor_var);

// Note(chenweihang): when two selected rows need to be added,
//   adding one to another is not equal to merging two selected rows
//   to one then add it to a empty selected rows, the after is correct
template <typename ReturnVarType, typename VarType>
std::shared_ptr<ReturnVarType> SelectedRowsMerge(const VarType& src1,
                                                 const VarType& src2) {
  const phi::SelectedRows& src_selected_rows1 =
      GetInnerTensor<phi::SelectedRows>(src1);
  const phi::SelectedRows& src_selected_rows2 =
      GetInnerTensor<phi::SelectedRows>(src2);

  paddle::experimental::CheckAndTrans2Contiguous(
      const_cast<phi::SelectedRows*>(&src_selected_rows1)->mutable_value());
  paddle::experimental::CheckAndTrans2Contiguous(
      const_cast<phi::SelectedRows*>(&src_selected_rows2)->mutable_value());

  auto place = src_selected_rows1.value().place();
  auto data_type =
      framework::TransToProtoVarType(src_selected_rows1.value().dtype());
  phi::DeviceContextPool& pool = phi::DeviceContextPool::Instance();

  std::vector<const phi::SelectedRows*> src_selected_rows;
  src_selected_rows.emplace_back(&src_selected_rows1);
  src_selected_rows.emplace_back(&src_selected_rows2);

  auto dst_var = std::make_shared<ReturnVarType>("Temp");
  phi::SelectedRows* dst_selected_rows =
      GetEmptyInnerTensor<phi::SelectedRows>(dst_var.get());

#define PADDLE_SELECTED_ROWS_ADD(dev_ctx_type, cpp_type)             \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) { \
    phi::DeviceContext* dev_ctx = pool.Get(place);                   \
    phi::funcs::scatter::MergeAdd<dev_ctx_type, cpp_type> merge_add; \
    merge_add(*(dynamic_cast<dev_ctx_type*>(dev_ctx)),               \
              src_selected_rows,                                     \
              dst_selected_rows);                                    \
    return dst_var;                                                  \
  }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (phi::is_gpu_place(place)) {
    PADDLE_SELECTED_ROWS_ADD(phi::GPUContext, float);
    PADDLE_SELECTED_ROWS_ADD(phi::GPUContext, double);
  } else {
#endif
#if defined(PADDLE_WITH_XPU)
    if (phi::is_xpu_place(place)) {
      PADDLE_SELECTED_ROWS_ADD(phi::XPUContext, float);
    } else {
#endif
      PADDLE_SELECTED_ROWS_ADD(phi::CPUContext, float);
      PADDLE_SELECTED_ROWS_ADD(phi::CPUContext, double);
#if defined(PADDLE_WITH_XPU)
    }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  }
#endif

#undef PADDLE_SELECTED_ROWS_ADD
  PADDLE_THROW(common::errors::InvalidArgument(
      "Not supported data type %s for SelectedRowsMerge",
      framework::DataTypeToString(data_type)));
}

template std::shared_ptr<paddle::Tensor> SelectedRowsMerge(
    const paddle::Tensor& src1, const paddle::Tensor& src2);
template std::shared_ptr<paddle::imperative::VariableWrapper> SelectedRowsMerge(
    const framework::Variable& src1, const framework::Variable& src2);

void VariableWrapperAdd(std::shared_ptr<VariableWrapper> var,
                        VariableWrapper* dst_var,
                        bool unchange_input) {
  auto& src = var->Var();
  auto* dst = dst_var->MutableVar();
  if (dst->IsType<phi::DenseTensor>()) {
    if (src.IsType<phi::DenseTensor>()) {
      TensorAdd<framework::Variable>(src, dst);
    } else if (src.IsType<phi::SelectedRows>()) {
      SelectedRowsAddToTensor(src, dst);
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unexpected branch, output variable type is %s",
          framework::ToTypeName(dst->Type())));
    }
  } else {
    if (src.IsType<phi::DenseTensor>()) {
      if (unchange_input) {
        framework::Variable new_dst;
        SelectedRowsAddTensor(*dst, src, &new_dst);
        *dst = std::move(new_dst);
      } else {
        auto* src_mutable = var->MutableVar();
        SelectedRowsAddToTensor(*dst, src_mutable);
        *dst = std::move(*(var->MutableVar()));
      }
    } else if (src.IsType<phi::SelectedRows>()) {
      auto temp = SelectedRowsMerge<VariableWrapper>(src, *dst);
      *dst = std::move(*(temp->MutableVar()));
    } else {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Unexpected branch, output variable type is %s",
          framework::ToTypeName(dst->Type())));
    }
  }
}

static phi::Place GetPlaceOfVar(const std::shared_ptr<VariableWrapper>& var) {
  phi::Place place;
  if (var->Var().IsType<phi::DenseTensor>()) {  // NOLINT
    place = var->Var().Get<phi::DenseTensor>().place();
  } else if (var->Var().IsType<phi::SelectedRows>()) {
    place = var->Var().Get<phi::SelectedRows>().place();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "only support DenseTensor and SelectedRows in dygraph"));
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
  PADDLE_ENFORCE_EQ(HasInnerVar(),
                    true,
                    common::errors::InvalidArgument(
                        "Leaf tensor should have inner var to store results of "
                        "this auto-grad"));
  PADDLE_ENFORCE_EQ(inner_var_->Var().IsInitialized(),
                    true,
                    common::errors::InvalidArgument(
                        "Interior var of Leaf tensor should be initialized."));
  auto* src = inner_var_->MutableVar();
  auto* dst = var_->MutableVar();
  if (!var_->IsEmpty()) {
    VLOG(6) << "Leaf Var(" << var_->Name()
            << ")'s Gradient has been initialized, will accumulate on "
               "previous gradient.";
    if (dst->IsType<phi::DenseTensor>()) {
      if (src->IsType<phi::DenseTensor>()) {
        TensorAdd<framework::Variable>(*src, dst);
      } else if (src->IsType<phi::SelectedRows>()) {
        SelectedRowsAddToTensor(*src, dst);
      }
    } else if (dst->IsType<phi::SelectedRows>()) {
      if (src->IsType<phi::DenseTensor>()) {
        SelectedRowsAddToTensor(*dst, src);
        *dst = std::move(*src);
      } else if (src->IsType<phi::SelectedRows>()) {
        auto temp = SelectedRowsMerge<VariableWrapper>(*src, *dst);
        *dst = std::move(*(temp->MutableVar()));
      }
    } else {
      PADDLE_THROW(common::errors::PermissionDenied(
          "Only support DenseTensor and SelectedRows for gradient var"));
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
  PADDLE_ENFORCE_EQ(var_->IsLeafGrad(),
                    true,
                    common::errors::Unavailable(
                        "Only leaf gradient Tensor can deal with by gradient "
                        "hook in gradient accumulator."));
  PADDLE_ENFORCE_EQ(
      SumGradCompleted(),
      true,
      common::errors::PreconditionNotMet(
          "Only can call gradient hooks after sum gradient completed."));
  PADDLE_ENFORCE_EQ(HasInnerVar(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Leaf Tensor's inner var is nullptr when "
                        "call gradient hook."));
  PADDLE_ENFORCE_EQ(
      inner_var_->Var().IsInitialized(),
      true,
      common::errors::PreconditionNotMet("Leaf Tensor's inner var "
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
      CheckVar(inner_var_, tmp_var);
    }
    inner_var_ = tmp_var;
  }
}

void GradientAccumulator::CallReduceHooks() {
  PADDLE_ENFORCE_EQ(
      var_->IsLeafGrad(),
      true,
      common::errors::Unavailable("Only leaf gradient Tensor can deal with "
                                  "by reduce hook in gradient accumulator."));
  PADDLE_ENFORCE_EQ(SumGradCompleted(),
                    true,
                    common::errors::PreconditionNotMet(
                        "Only can call reduce hooks after the gradient "
                        "summation is completed in current batch."));
  PADDLE_ENFORCE_EQ(HasInnerVar(),
                    false,
                    common::errors::PreconditionNotMet(
                        "Only can call reduce hooks after the "
                        "gradient accumulation is completed in "
                        "current batch or across batches."));
  if (var_->HasVoidHook()) {
    for (const auto& hook : var_->GetVoidHooks()) {
      VLOG(3) << "call gradient accumulator backward hooks.";
      (*hook)();
    }
  }
}

void EagerGradientAccumulator::SumGrad(std::shared_ptr<VariableWrapper> var,
                                       size_t trace_id,
                                       bool unchange_input) {
  /**
   * If var has grad node, it indicates that this var would be an input
   * of a grad op. Therefore, it should not be changed.
   */
  if (var->HasGradNode()) {
    unchange_input = true;
  }

  auto* dst_var = Var();
  phi::Place place = GetPlaceOfVar(var);
  if (!dst_var->OverriddenStopGradient()) {
    if (CurCnt() == 0) {
      MoveOrCopyVar(dst_var->MutableVar(), var->MutableVar(), unchange_input);
    } else {
      VLOG(6) << "Sum Gradient for: " << dst_var->Name()
              << " within this graph.";
      VariableWrapperAdd(var, dst_var, unchange_input);
    }
  } else {
    if (!dst_var->Var().IsInitialized() ||
        !dst_var->Var().Get<phi::DenseTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << dst_var->Name() << " as zero ";
      auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      if (!dst_var->Var().IsInitialized()) {
        auto* tensor = dst_var->MutableVar()->GetMutable<phi::DenseTensor>();
        VLOG(6) << "Dims of " << dst_var->Name()
                << " is set as: " << var->Var().Get<phi::DenseTensor>().dims();
        tensor->Resize(var->Var().Get<phi::DenseTensor>().dims());
        tensor->mutable_data(place, phi::TransToPhiDataType(var->DataType()));
        phi::funcs::set_constant(*dev_ctx, tensor, 0.0f);
      } else {
        auto* tensor = dst_var->MutableVar()->GetMutable<phi::DenseTensor>();
        tensor->mutable_data(place, phi::TransToPhiDataType(var->DataType()));
        phi::funcs::set_constant(*dev_ctx, tensor, 0.0f);
      }
    }
  }

  // Type may be changed after OP run, such as VarTypeInference
  // so synchronous VariableWrapper with Variable.
  if (dst_var->Var().IsType<phi::DenseTensor>()) {
    dst_var->SetType(framework::proto::VarType::DENSE_TENSOR);
  } else if (dst_var->Var().IsType<phi::SelectedRows>()) {
    dst_var->SetType(framework::proto::VarType::SELECTED_ROWS);
  }

  // Increase current count
  IncreaseCurCnt();
}

void SortedGradientAccumulator::SumGrad(std::shared_ptr<VariableWrapper> var,
                                        size_t trace_id,
                                        bool unchange_input) {
  auto* dst_var = Var();
  phi::Place place = GetPlaceOfVar(var);
  if (!dst_var->OverriddenStopGradient()) {
    if (ref_cnt_ == 1) {
      MoveOrCopyVar(dst_var->MutableVar(),
                    var->MutableVar(),
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
      std::sort(tmp_grad_vars_.begin(),
                tmp_grad_vars_.end(),
                [](const SavedVarInfo& info1, const SavedVarInfo& info2) {
                  return info1.trace_id > info2.trace_id;
                });

      for (auto& var_info : tmp_grad_vars_) {
        if (var_info.var->HasGradNode()) {
          var_info.unchange_input = true;
        }
      }

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      if (phi::is_gpu_place(place)) {  // NOLINT
        // sum selected rows firstly
        for (auto& var_info : tmp_grad_vars_) {
          if (!var_info.var->Var().IsType<phi::SelectedRows>()) {
            continue;
          }

          if (CurCnt() == 0) {
            MoveOrCopyVar(dst_var->MutableVar(),
                          var_info.var->MutableVar(),
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

          PADDLE_ENFORCE_EQ(var_info.var->Var().IsType<phi::DenseTensor>(),
                            true,
                            common::errors::PermissionDenied(
                                "Gradient var must be DenseTensor"));
          if (CurCnt() == 0) {
            MoveOrCopyVar(dst_var->MutableVar(),
                          var_info.var->MutableVar(),
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
              var_info.var->Var().IsType<phi::DenseTensor>() ||
                  var_info.var->Var().IsType<phi::SelectedRows>(),
              true,
              common::errors::PermissionDenied("The type of Gradient "
                                               "var must be DenseTensor "
                                               "or SelectedRows"));
          if (CurCnt() == 0) {
            MoveOrCopyVar(dst_var->MutableVar(),
                          var_info.var->MutableVar(),
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
        !dst_var->Var().Get<phi::DenseTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var->Name() << " as zero";
      auto* dev_ctx = phi::DeviceContextPool::Instance().Get(place);
      if (!dst_var->Var().IsInitialized()) {
        auto* tensor = dst_var->MutableVar()->GetMutable<phi::DenseTensor>();
        VLOG(6) << "Dims of " << dst_var->Name()
                << " is set as: " << var->Var().Get<phi::DenseTensor>().dims();
        tensor->Resize(var->Var().Get<phi::DenseTensor>().dims());
        tensor->mutable_data(place, phi::TransToPhiDataType(var->DataType()));
        phi::funcs::set_constant(*dev_ctx, tensor, 0.0f);
      } else {
        auto* tensor = dst_var->MutableVar()->GetMutable<phi::DenseTensor>();
        tensor->mutable_data(place, phi::TransToPhiDataType(var->DataType()));
        phi::funcs::set_constant(*dev_ctx, tensor, 0.0f);
      }
    }
    // looks like tmp_grad_vars will not have any member but just in case
    tmp_grad_vars_.clear();
  }

  if (dst_var->Var().IsType<phi::DenseTensor>()) {
    dst_var->SetType(framework::proto::VarType::DENSE_TENSOR);
  } else if (dst_var->Var().IsType<phi::SelectedRows>()) {
    dst_var->SetType(framework::proto::VarType::SELECTED_ROWS);
  }
}
}  // namespace paddle::imperative
