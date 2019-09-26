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
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace imperative {

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

#ifdef PADDLE_WITH_CUDA
  void operator()(const platform::CUDAPlace& place) {
    platform::CUDADeviceContext* ctx =
        dynamic_cast<platform::CUDADeviceContext*>(
            platform::DeviceContextPool::Instance().Get(place));
    auto blas = operators::math::GetBlas<platform::CUDADeviceContext, T>(*ctx);
    blas.AXPY(numel_, 1., x_, y_);
  }
#else
  void operator()(const platform::CUDAPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }
#endif

  // there is NO blas in CUDAPinnedPlace
  void operator()(const platform::CUDAPinnedPlace& place) {
    PADDLE_THROW("Do NOT support gradient merge in place %s", place);
  }

 private:
  int64_t numel_;
  const T* x_;
  T* y_;
};

void TensorAdd(const framework::Variable& src, framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_tensor = src.Get<framework::LoDTensor>();

  auto numel = src_tensor.numel();

  // FIXME(minqiyang): loss_grad op will pass a zero grad of label
  // ugly fix for it
  if (numel == 0) {
    return;
  }

  PADDLE_ENFORCE_EQ(dst_tensor->numel() == numel, true,
                    "dst_numel %d vs. src_numel %d", dst_tensor->numel(),
                    numel);

  auto data_type = src_tensor.type();
  auto place = src_tensor.place();

#define PADDLE_TENSOR_ADD_MACRO(cpp_type)                            \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) { \
    TensorAddFunctor<cpp_type> func(                                 \
        numel, src_tensor.data<cpp_type>(),                          \
        dst_tensor->mutable_data<cpp_type>(place));                  \
    boost::apply_visitor(func, place);                               \
    return;                                                          \
  }

  PADDLE_TENSOR_ADD_MACRO(float);
  PADDLE_TENSOR_ADD_MACRO(double);

#undef PADDLE_TENSOR_ADD_MACRO

  PADDLE_THROW("Not supported data type %s for AddTo",
               framework::DataTypeToString(data_type));
}

void EagerGradientAccumulator::Add(std::shared_ptr<VarBase> var,
                                   size_t trace_id) {
  auto* dst_var = var_->MutableVar();
  auto place = var->Var().Get<framework::LoDTensor>().place();
  if (!var_->OverridedStopGradient()) {
    VLOG(3) << "Sum Gradient for: " << var_->Name();
    if (cur_cnt_ == 0) {
      *dst_var = std::move(*(var->MutableVar()));
    } else {
      TensorAdd(var->Var(), dst_var);
    }
  } else {
    if (!var_->Var().IsInitialized() ||
        !var_->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var->Name() << " as zero";
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
      tensor->mutable_data(place, var->DataType());
      operators::math::set_constant(*dev_ctx, tensor, 0.0);
    }
  }
  ++cur_cnt_;
}

void SortedGradientAccumulator::Add(std::shared_ptr<VarBase> var,
                                    size_t trace_id) {
  auto* dst_var = var_->MutableVar();
  auto place = var->Var().Get<framework::LoDTensor>().place();
  if (!var_->OverridedStopGradient()) {
    if (ref_cnt_ == 1) {
      *dst_var = std::move(*(var->MutableVar()));
    } else {
      if (tmp_grad_vars_.empty()) {
        tmp_grad_vars_.reserve(ref_cnt_);
      }

      tmp_grad_vars_.emplace_back(std::move(var), trace_id);

      if (tmp_grad_vars_.size() != ref_cnt_) {
        return;
      }

      std::sort(tmp_grad_vars_.begin(), tmp_grad_vars_.end(),
                [](const std::pair<std::shared_ptr<VarBase>, size_t>& p1,
                   const std::pair<std::shared_ptr<VarBase>, size_t>& p2) {
                  return p1.second > p2.second;
                });

      *dst_var = std::move(*(tmp_grad_vars_[0].first->MutableVar()));
      for (size_t i = 1; i < tmp_grad_vars_.size(); ++i) {
        TensorAdd(tmp_grad_vars_[i].first->Var(), dst_var);
      }

      tmp_grad_vars_.clear();
    }
  } else {
    if (!var_->Var().IsInitialized() ||
        !var_->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var->Name() << " as zero";
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
      tensor->mutable_data(place, var->DataType());
      operators::math::set_constant(*dev_ctx, tensor, 0.0);
    }
    // looks like tmp_grad_vars will not have any member but just in case
    tmp_grad_vars_.clear();
  }
}

}  // namespace imperative
}  // namespace paddle
