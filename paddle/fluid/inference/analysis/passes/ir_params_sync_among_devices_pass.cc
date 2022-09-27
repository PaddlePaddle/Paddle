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

#include "paddle/fluid/inference/analysis/passes/ir_params_sync_among_devices_pass.h"

#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/common/data_type.h"

namespace paddle {
namespace inference {
namespace analysis {

#ifdef PADDLE_WITH_ASCEND_CL
void IrParamsSyncAmongDevicesPass::CopyParamsToNpu(Argument *argument) {
  if (!argument->use_npu()) return;

  auto &graph = argument->main_graph();
  std::vector<std::string> repetitive_params;

  if (graph.Has(framework::ir::kRepetitiveParamAttr))
    repetitive_params = graph.Get<std::vector<std::string>>(
        framework::ir::kRepetitiveParamAttr);

  LOG(INFO) << "Sync params from CPU to NPU";

  PADDLE_ENFORCE_EQ(argument->npu_device_id_valid(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "The npu_device_id field should be valid"));
  platform::Place place = platform::NPUPlace(argument->npu_device_id());
  auto *scope = argument->scope_ptr();
  std::vector<std::string> all_vars = scope->LocalVarNames();

  for (auto &var_name : all_vars) {
    auto *var = scope->FindLocalVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PreconditionNotMet("The var should not be nullptr"));

    if (var->IsType<framework::LoDTensor>() ||
        var->IsType<framework::Tensor>()) {
      auto *t = var->GetMutable<framework::LoDTensor>();

      platform::CPUPlace cpu_place;
      framework::LoDTensor temp_tensor;
      temp_tensor.Resize(t->dims());
      temp_tensor.mutable_data<float>(cpu_place);

      paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
      t->clear();
      paddle::framework::TensorCopySync(temp_tensor, place, t);
    }
  }
}

#else

void IrParamsSyncAmongDevicesPass::CopyParamsToGpu(Argument *argument) {
  // The parameters are on the cpu, therefore, synchronization is not necessary.
  if (!argument->use_gpu()) return;

  auto &graph = argument->main_graph();
  std::vector<std::string> repetitive_params;

  if (graph.Has(framework::ir::kRepetitiveParamAttr))
    repetitive_params = graph.Get<std::vector<std::string>>(
        framework::ir::kRepetitiveParamAttr);

  LOG(INFO) << "Sync params from CPU to GPU";

  PADDLE_ENFORCE_EQ(argument->gpu_device_id_valid(),
                    true,
                    platform::errors::PreconditionNotMet(
                        "The gpu_device_id field should be valid"));
  platform::Place place = platform::CUDAPlace(argument->gpu_device_id());
  auto *scope = argument->scope_ptr();
  std::vector<std::string> all_vars = scope->LocalVarNames();

  // We get all the vars from local_scope instead of the ProgramDesc.
  // Because there exists the case that new parameter variables are not added to
  // the program in the analysis pass.
  bool reserve_cpu_weights = false;
  bool with_dynamic_shape = false;
  if (argument->Has("max_input_shape") && argument->Has("min_input_shape") &&
      argument->Has("optim_input_shape")) {
    with_dynamic_shape = (argument->max_input_shape().size() > 0 &&
                          argument->min_input_shape().size() > 0 &&
                          argument->optim_input_shape().size() > 0);
  }
  with_dynamic_shape =
      with_dynamic_shape || (argument->Has("tensorrt_tuned_dynamic_shape") &&
                             argument->tensorrt_tuned_dynamic_shape());
  if (with_dynamic_shape) {
    reserve_cpu_weights = true;
  }

  int64_t params_total_bytes{0};
  for (auto *node : paddle::framework::ir::TopologySortOperations(graph)) {
    if (!node->IsOp()) continue;
    if (node->Op()->Type() == "feed" || node->Op()->Type() == "fetch") continue;
    for (auto *var_node : node->inputs) {
      if (!var_node->Var()->Persistable()) continue;
      auto var_name = var_node->Var()->Name();
      auto *var = scope->FindLocalVar(var_name);
      if (var->IsType<framework::LoDTensor>() ||
          var->IsType<framework::Tensor>()) {
        auto *t = var->GetMutable<framework::LoDTensor>();
        params_total_bytes += t->numel() * experimental::SizeOf(t->dtype());
      }
    }
  }

  {
    // Alloc memory in pool to store all parameters.
    framework::Tensor ts;
    ts.mutable_data(place, params_total_bytes);
  }

  std::unordered_set<std::string> visited;
  for (auto *node : paddle::framework::ir::TopologySortOperations(graph)) {
    if (!node->IsOp()) continue;
    if (node->Op()->Type() == "feed" || node->Op()->Type() == "fetch") continue;
    for (auto *var_node : node->inputs) {
      if (!var_node->Var()->Persistable()) continue;
      auto var_name = var_node->Var()->Name();
      if (std::count(
              repetitive_params.begin(), repetitive_params.end(), var_name)) {
        if (!reserve_cpu_weights) {
          scope->EraseVars({var_name});
        }
        continue;
      }
      if (visited.count(var_name)) continue;
      visited.insert(var_name);
      auto *var = scope->FindLocalVar(var_name);
      PADDLE_ENFORCE_NOT_NULL(var,
                              platform::errors::PreconditionNotMet(
                                  "The var should not be nullptr"));
      if (var->IsType<framework::LoDTensor>() ||
          var->IsType<framework::Tensor>()) {
        auto *t = var->GetMutable<framework::LoDTensor>();
        auto var_data_type = var_node->Var()->GetDataType();
        VLOG(5) << "var_name is " << var_name << ", data type is "
                << var_data_type;
        if (var_data_type == paddle::framework::proto::VarType::FP16) {
          framework::Tensor half_tensor;
          half_tensor.set_type(paddle::experimental::DataType::FLOAT16);
          half_tensor.Resize(t->dims());
          auto *half_data =
              half_tensor.mutable_data<float16>(platform::CPUPlace());
          for (int i = 0; i < t->numel(); i++) {
            auto *data = t->mutable_data<float16>(platform::CPUPlace());
            half_data[i] = static_cast<float16>(data[i]);
          }
          t->clear();
          paddle::framework::TensorCopySync(half_tensor, place, t);
        } else if (var_data_type == paddle::framework::proto::VarType::BF16) {
          framework::Tensor bf16_tensor;
          bf16_tensor.set_type(paddle::experimental::DataType::BFLOAT16);
          bf16_tensor.Resize(t->dims());
          auto *bf16_data = bf16_tensor.mutable_data<platform::bfloat16>(
              platform::CPUPlace());
          for (int i = 0; i < t->numel(); i++) {
            auto *data = t->mutable_data<bfloat16>(platform::CPUPlace());
            bf16_data[i] = static_cast<platform::bfloat16>(data[i]);
          }
          t->clear();
          paddle::framework::TensorCopySync(bf16_tensor, place, t);
        } else {
          platform::CPUPlace cpu_place;
          framework::LoDTensor temp_tensor;
          temp_tensor.Resize(t->dims());
          paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
          t->clear();
          paddle::framework::TensorCopySync(temp_tensor, place, t);
        }
      }
    }
  }
}

#endif

void IrParamsSyncAmongDevicesPass::RunImpl(Argument *argument) {
  PADDLE_ENFORCE_EQ(
      argument->scope_valid(),
      true,
      platform::errors::PreconditionNotMet("The scope field should be valid"));

#ifdef PADDLE_WITH_ASCEND_CL
  if (!argument->use_npu_valid()) return;
  CopyParamsToNpu(argument);
#else
  if (!argument->use_gpu_valid()) return;
  CopyParamsToGpu(argument);
#endif
}

std::string IrParamsSyncAmongDevicesPass::repr() const {
  return "ir-params-sync-among-devices-pass";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
