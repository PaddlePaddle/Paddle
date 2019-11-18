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
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/imperative/layer.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/selected_rows_functor.h"
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

void SelectedRowsAddToTensor(const framework::Variable& src,
                             framework::Variable* dst) {
  auto* dst_tensor = dst->GetMutable<framework::LoDTensor>();
  auto& src_selectedrows = src.Get<framework::SelectedRows>();
  auto place = dst_tensor->place();
  auto data_type = src_selectedrows.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  if (paddle::platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
#define PADDLE_SelectedRowsAddToTensor_MACRO(cpp_type)                      \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {        \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);             \
    paddle::operators::math::SelectedRowsAddToTensor<                       \
        paddle::platform::CUDADeviceContext, cpp_type>                      \
        functor;                                                            \
    functor(*(dynamic_cast<paddle::platform::CUDADeviceContext*>(dev_ctx)), \
            src_selectedrows, dst_tensor);                                  \
    return;                                                                 \
  }

    PADDLE_SelectedRowsAddToTensor_MACRO(float);
    PADDLE_SelectedRowsAddToTensor_MACRO(double);

#undef PADDLE_SelectedRowsAddToTensor_MACRO

    PADDLE_THROW("Not supported data type %s for AddTo",
                 framework::DataTypeToString(data_type));
#else
    PADDLE_THROW("CUDA is not support.");
#endif
  } else {
#define PADDLE_SelectedRowsAddToTensor_MACRO(cpp_type)                     \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {       \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);            \
    paddle::operators::math::SelectedRowsAddToTensor<                      \
        paddle::platform::CPUDeviceContext, cpp_type>                      \
        functor;                                                           \
    functor(*(dynamic_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), \
            src_selectedrows, dst_tensor);                                 \
    return;                                                                \
  }

    PADDLE_SelectedRowsAddToTensor_MACRO(float);
    PADDLE_SelectedRowsAddToTensor_MACRO(double);

#undef PADDLE_SelectedRowsAddToTensor_MACRO

    PADDLE_THROW("Not supported data type %s for AddTo",
                 framework::DataTypeToString(data_type));
  }
}

void SelectedRowsAddToSelectedRows(const framework::Variable& src,
                                   framework::Variable* dst) {
  auto& src_selectedrows = src.Get<framework::SelectedRows>();
  auto place = src_selectedrows.value().place();
  auto data_type = src_selectedrows.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  std::vector<const paddle::framework::SelectedRows*> inputs;
  inputs.push_back(&src_selectedrows);
  auto* output = dst->GetMutable<paddle::framework::SelectedRows>();

  if (paddle::platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
#define PADDLE_SelectedRowsAddToSelectedRows_MACRO(cpp_type)                  \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {          \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);               \
    paddle::operators::math::scatter::MergeAdd<                               \
        paddle::platform::CUDADeviceContext, cpp_type>                        \
        merge_add;                                                            \
    merge_add(*(dynamic_cast<paddle::platform::CUDADeviceContext*>(dev_ctx)), \
              inputs, output);                                                \
    return;                                                                   \
  }

    PADDLE_SelectedRowsAddToSelectedRows_MACRO(float);
    PADDLE_SelectedRowsAddToSelectedRows_MACRO(double);

#undef PADDLE_SelectedRowsAddToSelectedRows_MACRO

    PADDLE_THROW("Not supported data type %s for AddTo",
                 framework::DataTypeToString(data_type));
#else
    PADDLE_THROW("CUDA is not support.");
#endif
  } else {
#define PADDLE_SelectedRowsAddToSelectedRows_MACRO(cpp_type)                 \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {         \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);              \
    paddle::operators::math::scatter::MergeAdd<                              \
        paddle::platform::CPUDeviceContext, cpp_type>                        \
        merge_add;                                                           \
    merge_add(*(dynamic_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), \
              inputs, output);                                               \
    return;                                                                  \
  }

    PADDLE_SelectedRowsAddToSelectedRows_MACRO(float);
    PADDLE_SelectedRowsAddToSelectedRows_MACRO(double);

#undef PADDLE_SelectedRowsAddToSelectedRows_MACRO

    PADDLE_THROW("Not supported data type %s for AddTo",
                 framework::DataTypeToString(data_type));
  }
}

std::shared_ptr<VarBase> SelectedRowsAddSelectedRows(
    const framework::Variable& src, framework::Variable* dst) {
  auto* dst_selectedrows = dst->GetMutable<framework::SelectedRows>();
  auto& src_selectedrows = src.Get<framework::SelectedRows>();
  auto place = src_selectedrows.value().place();
  auto data_type = src_selectedrows.value().type();
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();

  std::vector<const paddle::framework::SelectedRows*> inputs;
  inputs.push_back(&src_selectedrows);
  inputs.push_back(dst_selectedrows);
  auto temp = std::make_shared<VarBase>(false, "temp");
  auto* output =
      temp->MutableVar()->GetMutable<paddle::framework::SelectedRows>();

  if (paddle::platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
#define PADDLE_SelectedRowsAddSelectedRows_MACRO(cpp_type)                    \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {          \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);               \
    paddle::operators::math::scatter::MergeAdd<                               \
        paddle::platform::CUDADeviceContext, cpp_type>                        \
        merge_add;                                                            \
    merge_add(*(dynamic_cast<paddle::platform::CUDADeviceContext*>(dev_ctx)), \
              inputs, output);                                                \
    return temp;                                                              \
  }

    PADDLE_SelectedRowsAddSelectedRows_MACRO(float);
    PADDLE_SelectedRowsAddSelectedRows_MACRO(double);

#undef PADDLE_SelectedRowsAddSelectedRows_MACRO

    PADDLE_THROW("Not supported data type %s for AddTo",
                 framework::DataTypeToString(data_type));
#else
    PADDLE_THROW("CUDA is not support.");
#endif
  } else {
#define PADDLE_SelectedRowsAddSelectedRows_MACRO(cpp_type)                   \
  if (data_type == framework::DataTypeTrait<cpp_type>::DataType()) {         \
    paddle::platform::DeviceContext* dev_ctx = pool.Get(place);              \
    paddle::operators::math::scatter::MergeAdd<                              \
        paddle::platform::CPUDeviceContext, cpp_type>                        \
        merge_add;                                                           \
    merge_add(*(dynamic_cast<paddle::platform::CPUDeviceContext*>(dev_ctx)), \
              inputs, output);                                               \
    return temp;                                                             \
  }

    PADDLE_SelectedRowsAddSelectedRows_MACRO(float);
    PADDLE_SelectedRowsAddSelectedRows_MACRO(double);

#undef PADDLE_SelectedRowsAddSelectedRows_MACRO

    PADDLE_THROW("Not supported data type %s for AddTo",
                 framework::DataTypeToString(data_type));
  }

  return temp;
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

void VarBaseAdd(std::shared_ptr<VarBase> var, VarBase* var_) {
  auto& src = var->Var();
  auto* dst = var_->MutableVar();
  if (dst->IsType<framework::LoDTensor>()) {
    if (src.IsType<framework::LoDTensor>()) {
      TensorAdd(src, dst);
    } else if (src.IsType<framework::SelectedRows>()) {
      SelectedRowsAddToTensor(src, dst);
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   framework::ToTypeName(dst->Type()));
    }
  } else {
    if (src.IsType<framework::LoDTensor>()) {
      auto* src_temp = var->MutableVar();
      SelectedRowsAddToTensor(*dst, src_temp);
      *dst = std::move(*(var->MutableVar()));
      var_->SetType(framework::proto::VarType::LOD_TENSOR);
    } else if (src.IsType<framework::SelectedRows>()) {
      std::shared_ptr<VarBase> output = SelectedRowsAddSelectedRows(src, dst);
      *dst = std::move(*(output->MutableVar()));
    } else {
      PADDLE_THROW("Unexpected branch, output variable type is %s",
                   framework::ToTypeName(dst->Type()));
    }
  }
}

void EagerGradientAccumulator::Add(std::shared_ptr<VarBase> var,
                                   size_t trace_id) {
  auto* dst_var = var_->MutableVar();
  paddle::platform::Place place;
  if (var->Var().IsType<framework::LoDTensor>()) {
    place = var->Var().Get<framework::LoDTensor>().place();
  } else if (var->Var().IsType<framework::SelectedRows>()) {
    place = var->Var().Get<framework::SelectedRows>().place();
  } else {
    PADDLE_THROW("only support LoDTensor and SelectedRows in dygraph");
  }
  if (!var_->OverridedStopGradient()) {
    VLOG(3) << "Sum Gradient for: " << var_->Name();
    if (cur_cnt_ == 0) {
      if (var->Var().IsType<framework::SelectedRows>()) {
        auto temp = std::make_shared<VarBase>(false, "temp");
        SelectedRowsAddToSelectedRows(*(var->MutableVar()), temp->MutableVar());
        var_->SetType(framework::proto::VarType::SELECTED_ROWS);
        *dst_var = std::move(*(temp->MutableVar()));
      } else {
        *dst_var = std::move(*(var->MutableVar()));
      }
    } else {
      VarBaseAdd(var, var_);
    }
  } else {
    if (!var_->Var().IsInitialized() ||
        !var_->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var_->Name() << " as zero ";

      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      if (!var_->Var().IsInitialized()) {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        VLOG(6) << "Dims of " << var_->Name() << " is set as: "
                << var->Var().Get<framework::LoDTensor>().dims();
        tensor->Resize(var->Var().Get<framework::LoDTensor>().dims());
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      } else {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
  }
  ++cur_cnt_;
}

void SortedGradientAccumulator::Add(std::shared_ptr<VarBase> var,
                                    size_t trace_id) {
  auto* dst_var = var_->MutableVar();
  paddle::platform::Place place;
  if (var->Var().IsType<framework::LoDTensor>()) {
    place = var->Var().Get<framework::LoDTensor>().place();
  } else if (var->Var().IsType<framework::SelectedRows>()) {
    place = var->Var().Get<framework::SelectedRows>().place();
  } else {
    PADDLE_THROW("only support LoDTensor and SelectedRows in dygraph");
  }
  if (!var_->OverridedStopGradient()) {
    if (ref_cnt_ == 1) {
      if (var->Var().IsType<framework::SelectedRows>()) {
        auto temp = std::make_shared<VarBase>(false, "temp");
        SelectedRowsAddToSelectedRows(*(var->MutableVar()), temp->MutableVar());
        var_->SetType(framework::proto::VarType::SELECTED_ROWS);
        *dst_var = std::move(*(temp->MutableVar()));
      } else {
        *dst_var = std::move(*(var->MutableVar()));
      }
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

      if (paddle::platform::is_gpu_place(place)) {
        bool is_initialized = false;
        for (size_t i = 0; i < tmp_grad_vars_.size(); ++i) {
          if (tmp_grad_vars_[i]
                  .first->Var()
                  .IsType<framework::SelectedRows>()) {
            if (!is_initialized) {
              is_initialized = true;
              auto temp = std::make_shared<VarBase>(false, "temp");
              SelectedRowsAddToSelectedRows(
                  *(tmp_grad_vars_[i].first->MutableVar()), temp->MutableVar());
              var_->SetType(framework::proto::VarType::SELECTED_ROWS);
              *dst_var = std::move(*(temp->MutableVar()));
            } else {
              VarBaseAdd(tmp_grad_vars_[i].first, var_);
            }
          }
        }
        if (!is_initialized) {
          is_initialized = true;
          *dst_var = std::move(*(tmp_grad_vars_[0].first->MutableVar()));
        }
        for (size_t i = 0; i < tmp_grad_vars_.size(); ++i) {
          if (tmp_grad_vars_[i].first->Var().IsType<framework::LoDTensor>()) {
            VarBaseAdd(tmp_grad_vars_[i].first, var_);
          }
        }
      } else {
        if (tmp_grad_vars_[0].first->Var().IsType<framework::SelectedRows>()) {
          auto temp = std::make_shared<VarBase>(false, "temp");
          SelectedRowsAddToSelectedRows(
              *(tmp_grad_vars_[0].first->MutableVar()), temp->MutableVar());
          var_->SetType(framework::proto::VarType::SELECTED_ROWS);
          *dst_var = std::move(*(temp->MutableVar()));
        } else {
          *dst_var = std::move(*(tmp_grad_vars_[0].first->MutableVar()));
        }
        for (size_t i = 1; i < tmp_grad_vars_.size(); ++i) {
          VarBaseAdd(tmp_grad_vars_[i].first, var_);
        }
      }
      tmp_grad_vars_.clear();
    }
  } else {
    if (!var_->Var().IsInitialized() ||
        !var_->Var().Get<framework::LoDTensor>().IsInitialized()) {
      VLOG(6) << "Set StopGradient Grad: " << var->Name() << " as zero";
      auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
      if (!var_->Var().IsInitialized()) {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        VLOG(6) << "Dims of " << var_->Name() << " is set as: "
                << var->Var().Get<framework::LoDTensor>().dims();
        tensor->Resize(var->Var().Get<framework::LoDTensor>().dims());
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      } else {
        auto* tensor = var_->MutableVar()->GetMutable<framework::LoDTensor>();
        tensor->mutable_data(place, var->DataType());
        operators::math::set_constant(*dev_ctx, tensor, 0.0);
      }
    }
    // looks like tmp_grad_vars will not have any member but just in case
    tmp_grad_vars_.clear();
  }
}

}  // namespace imperative
}  // namespace paddle
