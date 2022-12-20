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

#include <cstdlib>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/bfloat16.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/phi/common/data_type.h"

DEFINE_bool(
    custom_model_save_cpu,
    false,
    "Keep old mode for developers, the model is saved on cpu not device.");

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

    if (var->IsType<phi::DenseTensor>() || var->IsType<phi::DenseTensor>()) {
      auto *t = var->GetMutable<phi::DenseTensor>();

      platform::CPUPlace cpu_place;
      phi::DenseTensor temp_tensor;
      temp_tensor.Resize(t->dims());
      temp_tensor.mutable_data<float>(cpu_place);

      paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
      t->clear();
      paddle::framework::TensorCopySync(temp_tensor, place, t);
    }
  }
}
#endif

#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
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
      if (var->IsType<phi::DenseTensor>() || var->IsType<phi::DenseTensor>()) {
        auto *t = var->GetMutable<phi::DenseTensor>();
        auto var_data_type = var_node->Var()->GetDataType();
        VLOG(5) << "var_name is " << var_name << ", data type is "
                << var_data_type;
        platform::CPUPlace cpu_place;
        phi::DenseTensor temp_tensor;
        temp_tensor.Resize(t->dims());
        paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
        t->clear();
        paddle::framework::TensorCopySync(temp_tensor, place, t);
      }
    }
  }
}
#endif

#ifdef PADDLE_WITH_CUSTOM_DEVICE
void IrParamsSyncAmongDevicesPass::CopyParamsToCustomDevice(
    Argument *argument) {
  if (!argument->use_custom_device()) return;

  // On old mode, the model is saved on cpu not device.
  if (argument->custom_device_type() == "OpenCL") {
    PADDLE_ENFORCE_EQ(
        FLAGS_custom_model_save_cpu,
        false,
        phi::errors::InvalidArgument(
            "'FLAGS_custom_model_save_cpu = false' is only for the developers "
            "who have not completed custom device memory settings. Setting to "
            "true will make "
            "model memory reserve on the cpu, and make inference slower."));
  }

  if (FLAGS_custom_model_save_cpu) return;

  auto &graph = argument->main_graph();
  std::vector<std::string> repetitive_params;

  if (graph.Has(framework::ir::kRepetitiveParamAttr))
    repetitive_params = graph.Get<std::vector<std::string>>(
        framework::ir::kRepetitiveParamAttr);

  LOG(INFO) << "Sync params from CPU to CustomDevice"
            << argument->custom_device_type() << "/"
            << argument->custom_device_id();

  platform::Place place = platform::CustomPlace(argument->custom_device_type(),
                                                argument->custom_device_id());
  auto *scope = argument->scope_ptr();
  std::vector<std::string> all_vars = scope->LocalVarNames();

  for (auto &var_name : all_vars) {
    auto *var = scope->FindLocalVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        platform::errors::PreconditionNotMet("The var should not be nullptr"));

    if (var->IsType<phi::DenseTensor>() || var->IsType<phi::DenseTensor>()) {
      auto *t = var->GetMutable<phi::DenseTensor>();

      platform::CPUPlace cpu_place;
      phi::DenseTensor temp_tensor;
      temp_tensor.Resize(t->dims());

      paddle::framework::TensorCopySync(*t, cpu_place, &temp_tensor);
      t->clear();
      paddle::framework::TensorCopySync(temp_tensor, place, t);
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
  if (argument->use_npu_valid()) {
    CopyParamsToNpu(argument);
  }
#endif
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
  if (argument->use_gpu_valid()) {
    CopyParamsToGpu(argument);
  }
#endif
#ifdef PADDLE_WITH_CUSTOM_DEVICE
  if (argument->use_custom_device_valid()) {
    CopyParamsToCustomDevice(argument);
  }
#endif
}

std::string IrParamsSyncAmongDevicesPass::repr() const {
  return "ir_params_sync_among_devices_pass";
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
