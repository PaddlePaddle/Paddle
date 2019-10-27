/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/details/double_check.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace framework {
namespace details {
void DoubleCheckOperator::Wait(const platform::Place& place) {
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    auto* dev_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    dev_ctx->Wait();
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  }
}

void DoubleCheckOperator::GetCastInputAndOutputs(
    const Scope& scope, const platform::Place& place,
    const OperatorBase& base_op,
    std::map<std::string, std::string>* diff_var_names) {
  std::vector<std::string> inputs;
  for (auto it = base_op.Inputs().begin(); it != base_op.Inputs().end(); it++) {
    auto& var_names = it->second;
    for (size_t i = 0; i < var_names.size(); ++i) {
      inputs.push_back(var_names[i]);
    }
  }

  std::vector<std::string> outputs;
  for (auto it = base_op.Outputs().begin(); it != base_op.Outputs().end();
       it++) {
    auto& var_names = it->second;
    for (size_t i = 0; i < var_names.size(); ++i) {
      auto var = scope.FindVar(var_names[i]);

      const framework::Tensor* tensor{nullptr};
      if (var->IsType<framework::LoDTensor>()) {
        tensor = &var->Get<framework::LoDTensor>();
      } else if (var->IsType<framework::SelectedRows>()) {
        tensor = &var->Get<framework::SelectedRows>().value();
      } else {
        continue;
      }

      auto tensor_dtype = tensor->type();
      if (tensor_dtype != framework::proto::VarType::FP16) {
        continue;
      }

      outputs.push_back(var_names[i]);
    }
  }

  if (outputs.size() < 1) {
    return;
  }

  PADDLE_ENFORCE_EQ(inputs.size(), 1, "inputs size:%llu", inputs.size());
  PADDLE_ENFORCE_EQ(outputs.size(), 1, "outputs size:%llu", outputs.size());
  (*diff_var_names)[outputs[0]] = inputs[0];
}

void DoubleCheckOperator::Run(const Scope& scope,
                              const platform::Place& place) {
  std::string type = base_op_->Type();
  VLOG(10) << "begin to double check " << base_op_->Type();

  if (type == "fill_constant" || type == "reshape2" ||
      type == "reshape2_grad" || type == "reshape" || type == "reshape_grad" ||
      type == "transpose2" || type == "transpose2_grad") {
    base_op_->Run(scope, place);
    VLOG(10) << "end double check " << type << ", need not to check";
    return;
  }

  if (type == "cast") {
    base_op_->Run(scope, place);
    Wait(place);

    VLOG(10) << "PrepareNameMap";

    std::map<std::string, std::string> diff_var_names;
    VariableNameMap outputs;
    GetCastInputAndOutputs(scope, place, *base_op_, &diff_var_names);

    if (diff_var_names.size() == 0) {
      VLOG(10) << "end double check " << type << ", no fp16 should be checked";
      return;
    }

    for (auto it : diff_var_names) {
      VLOG(10) << "var_name: " << it.first << " and " << it.second;
      Diff(scope, place, it.first, it.second);
    }

    VLOG(10) << "end double check " << type;
    return;
  }

  base_op_->Run(scope, place);
  // Wait var's initailization.
  Wait(place);

  VariableNameMap inputs;
  VariableNameMap outputs;

  std::map<std::string, std::string> input_diff_var_names;
  std::map<std::string, std::string> output_diff_var_names;

  Scope* var_scope = const_cast<Scope*>(&scope);
  VLOG(10) << "PrepareNameMap";
  PrepareNameMap(var_scope, place, base_op_->Inputs(), &inputs,
                 &input_diff_var_names, base_handle_.Inputs());
  PrepareNameMap(var_scope, place, base_op_->Outputs(), &outputs,
                 &output_diff_var_names, base_handle_.Outputs());

  if (input_diff_var_names.size() == 0 && output_diff_var_names.size() == 0) {
    VLOG(10) << "end double check " << base_op_->Type()
             << ", no fp16 should be checked";
    return;
  }

  VLOG(10) << "double check " << base_op_->Type() << " fp16 content";
  auto check_op = paddle::framework::OpRegistry::CreateOp(type, inputs, outputs,
                                                          base_op_->Attrs());
  check_op->Run(scope, place);

  for (auto it : output_diff_var_names) {
    VLOG(10) << "var_name: " << it.first << " and " << it.second;
    Diff(scope, place, it.first, it.second);
  }

  // Wait difference complation.
  Wait(place);
  VLOG(10) << "end double check " << base_op_->Type();
}

struct RangeFunctor {
  RangeFunctor(const platform::float16* a, const float* b) : a_(a), b_(b) {}
  inline HOSTDEVICE void operator()(size_t id) const {
    PADDLE_ENFORCE((fabs(static_cast<float>(a_[id]) - b_[id]) < 0.1),
                   "fabs(%f - %f) > 0.1", static_cast<float>(a_[id]), b_[id]);
  }
  const platform::float16* a_;
  const float* b_;
};
void DoubleCheckOperator::Diff(const Scope& scope, const platform::Place& place,
                               const std::string& a, const std::string& b) {
  auto var_a = scope.FindVar(a);
  const Tensor* tensor_a{nullptr};
  if (var_a->IsType<framework::LoDTensor>()) {
    tensor_a = &var_a->Get<framework::LoDTensor>();
  } else if (var_a->IsType<framework::SelectedRows>()) {
    tensor_a = &var_a->Get<framework::SelectedRows>().value();
  }

  auto var_b = scope.FindVar(b);
  const Tensor* tensor_b{nullptr};
  if (var_a->IsType<framework::LoDTensor>()) {
    tensor_b = &var_b->Get<framework::LoDTensor>();
  } else if (var_a->IsType<framework::SelectedRows>()) {
    tensor_b = &var_b->Get<framework::SelectedRows>().value();
  }

  RangeFunctor functor(tensor_a->data<platform::float16>(),
                       tensor_b->data<float>());
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    auto* dev_ctx = reinterpret_cast<platform::CUDADeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    platform::ForRange<platform::CUDADeviceContext> for_range(
        *dev_ctx, tensor_a->numel());
    for_range(functor);
#else
    PADDLE_THROW("PaddlePaddle should compile with GPU.");
#endif
  } else {
    auto* dev_ctx = reinterpret_cast<platform::CPUDeviceContext*>(
        platform::DeviceContextPool::Instance().Get(place));
    platform::ForRange<platform::CPUDeviceContext> for_range(*dev_ctx,
                                                             tensor_a->numel());
    for_range(functor);
  }
}

void DoubleCheckOperator::PrepareNameMap(
    Scope* scope, const platform::Place& place,
    const framework::VariableNameMap& name_map,
    framework::VariableNameMap* dst_name_map,
    std::map<std::string, std::string>* diff_var_names,
    const std::vector<VarHandleBase*>& var_handles) {
  VLOG(10) << "name map size:" << name_map.size();
  for (auto it = name_map.begin(); it != name_map.end(); it++) {
    std::vector<std::string>& dst_var_names = (*dst_name_map)[it->first];
    dst_var_names.reserve(it->second.size());

    auto& var_names = it->second;
    VLOG(10) << "var_name vector size:" << var_names.size();
    for (size_t i = 0; i < var_names.size(); ++i) {
      auto var_name = var_names[i];
      auto var = scope->FindVar(var_name);
      std::string fp32_var_name = var_name + "_fp32_double_check_fake_name";

      const framework::Tensor* tensor{nullptr};
      if (var->IsType<framework::LoDTensor>()) {
        tensor = &var->Get<framework::LoDTensor>();
      } else if (var->IsType<framework::SelectedRows>()) {
        tensor = &var->Get<framework::SelectedRows>().value();
      } else {
        dst_var_names.push_back(var_name);
        continue;
      }

      VarHandleBase* var_handle{nullptr};
      for (auto var_h : var_handles) {
        if (var_h->Name() == var_name) {
          var_handle = var_h;
          break;
        }
      }

      PADDLE_ENFORCE_NOT_NULL(var_handle, "var name should in var_handles:%s",
                              var_name);
      if (var_handle->Node()->Var()->GetDataType() !=
          framework::proto::VarType::FP16) {
        dst_var_names.push_back(var_name);
        continue;
      }

      dst_var_names.push_back(fp32_var_name);
      (*diff_var_names)[var_name] = fp32_var_name;
      if (scope->FindVar(fp32_var_name) != nullptr) {
        continue;
      }

      VLOG(10) << "alloc new data and cast from:" << var_name << " to "
               << fp32_var_name << " place:" << place
               << ", numel:" << tensor->numel();
      auto fp32_var = scope->Var(fp32_var_name);
      framework::Tensor* fp32_tensor{nullptr};
      if (var->IsType<framework::LoDTensor>()) {
        fp32_tensor = fp32_var->GetMutable<framework::LoDTensor>();
      } else if (var->IsType<framework::SelectedRows>()) {
        fp32_tensor =
            fp32_var->GetMutable<framework::SelectedRows>()->mutable_value();
      }
      fp32_tensor->mutable_data(place, framework::proto::VarType::FP32);

      // VLOG(10) << "cast data from:" << var_name
      // << " to new var:" << fp32_var_name;
      framework::AttributeMap cast_op_attrs;
      cast_op_attrs["in_dtype"] = framework::proto::VarType::FP16;
      cast_op_attrs["out_dtype"] = framework::proto::VarType::FP32;
      auto cast_op = paddle::framework::OpRegistry::CreateOp(
          "cast", {{"X", {var_name}}}, {{"Out", {fp32_var_name}}},
          cast_op_attrs);
      cast_op->Run(*scope, place);
    }
  }

  return;
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
