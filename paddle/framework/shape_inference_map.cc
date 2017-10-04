/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/shape_inference_map.h"

namespace paddle {
namespace framework {

static VariableNameMap ConvertOpProtoVarsToVarNameMap(
    const google::protobuf::RepeatedPtrField<OpProto::Var>& op_proto_vars) {
  VariableNameMap ret_val;
  for (auto& var : op_proto_vars) {
    ret_val[var.name()] = {};
  }
  return ret_val;
}

static ShapeInferenceMap* g_shape_inference_map = nullptr;

ShapeInferenceMap& ShapeInferenceMap::Instance() {
  if (g_shape_inference_map == nullptr) {
    g_shape_inference_map = new ShapeInferenceMap();
  }
  return *g_shape_inference_map;
}

void ShapeInferenceMap::CreateOpWithKernel(const OpInfo& op_info,
                                           const std::string& op_type) {
  const VariableNameMap inputs =
      ConvertOpProtoVarsToVarNameMap(op_info.Proto().inputs());
  const VariableNameMap outputs =
      ConvertOpProtoVarsToVarNameMap(op_info.Proto().outputs());
  auto* op = op_info.Creator()(op_type, inputs, outputs, {});
  auto* op_with_kernel = dynamic_cast<OperatorWithKernel*>(op);
  auto it = op_shape_inference_map_.find(op_type);
  if (it != op_shape_inference_map_.end()) {
    PADDLE_THROW("OpWithKernel(%s) is already registered for infer_shape",
                 op_type);
  }
  if (op_with_kernel != nullptr) {
    op_shape_inference_map_[op_type] = op_with_kernel;
  }
}

}  // namespace framework
}  // namespace paddle
