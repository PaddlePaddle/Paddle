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

#include "paddle/fluid/inference/analysis/passes/convert_to_mixed_precision.h"

#include <algorithm>
#include <iterator>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/tensor_meta.h"

using namespace paddle::framework;  // NOLINT

namespace paddle {
namespace inference {
namespace analysis {

namespace {

inline std::string SerializeParams(framework::Scope* scope,
                                   const std::vector<std::string>& params) {
  std::ostringstream os;
  phi::CPUContext ctx;
  for (const auto& param : params) {
    VLOG(3) << "Serialize param: " << param;
    PADDLE_ENFORCE_NOT_NULL(
        scope->FindVar(param),
        platform::errors::NotFound("Block should already have a '%s' variable",
                                   param));
    auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
    framework::SerializeToStream(os, *tensor, ctx);
  }
  return os.str();
}

inline void StrToBinary(const std::string& path, const std::string& str) {
  std::ofstream file(path.c_str(), std::ios::binary);
  file.write(str.c_str(), str.size());
  file.close();
}

inline bool NodeVarHasDtype(framework::ir::Node* node) {
  if (node->IsCtrlVar()) return false;

  if (node->IsVar() &&
      (node->Var()->GetType() ==
           paddle::framework::proto::VarType::SELECTED_ROWS ||
       node->Var()->GetType() ==
           paddle::framework::proto::VarType::LOD_TENSOR ||
       node->Var()->GetType() ==
           paddle::framework::proto::VarType::LOD_TENSOR_ARRAY ||
       node->Var()->GetType() == paddle::framework::proto::VarType::STRINGS ||
       node->Var()->GetType() == paddle::framework::proto::VarType::VOCAB)) {
    return true;
  }

  return false;
}

// Return Node* which first appers in block.
framework::ir::Node* GetRealNode(
    const std::vector<framework::ir::Graph*>& graphes,
    int block_idx,
    framework::ir::Node* node,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map) {
  if (vars_in_multi_block_map->count(node->Name())) {
    int var_origin_block_id = vars_in_multi_block_map->at(node->Name()).second;
    if (block_idx != var_origin_block_id) {
      auto graph = graphes[var_origin_block_id];
      for (auto nd : graph->Nodes()) {
        if (nd->Name() == node->Name()) {
          return nd;
        }
      }
    }
  }

  return node;
}

inline bool VarIsMultiOpsOut(
    const std::vector<framework::ir::Graph*>& graphes,
    int block_idx,
    framework::ir::Node* op_node,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map,
    const std::vector<std::set<std::string>>& vars_appear_multi_in_one_block) {
  CHECK_EQ(op_node->IsOp(), true);
  for (auto* out : op_node->outputs) {
    if (out->IsCtrlVar()) continue;
    auto* real_node =
        GetRealNode(graphes, block_idx, out, vars_in_multi_block_map);
    if (!real_node->Var()->Persistable() &&
        vars_appear_multi_in_one_block[block_idx].count(out->Name())) {
      VLOG(2) << out->Name()
              << " is multi op's out, so we skip convert to fp16";
      return true;
    }
  }
  return false;
}

void SaveMixedModel(
    framework::ir::Graph* graph,
    framework::Scope* scope,
    framework::ProgramDesc* mixed_program_desc,
    const std::string& mixed_model_file,
    const std::string& mixed_params_file,
    phi::DataType mixed_precision,
    const std::unordered_map<std::string,
                             std::pair<framework::proto::VarType::Type, int>>&
        vars_in_multi_block_map) {
  paddle::CPUPlace place;
  auto parameters = scope->LocalVarNames();
  std::sort(parameters.begin(), parameters.end());

  std::unordered_set<std::string> weights_should_be_fp32;
  for (auto* node : graph->Nodes()) {
    if (!(node->IsVar() && !node->IsCtrlVar())) continue;
    if (NodeVarHasDtype(node)) {
      if (node->Var()->Persistable() &&
          node->Var()->GetDataType() ==
              paddle::framework::proto::VarType::FP32) {
        VLOG(2) << "weights keep to fp32: " << node->Name();
        weights_should_be_fp32.insert(node->Name());
      }
    }
  }

  for (const auto& param_name : parameters) {
    auto* var = scope->FindLocalVar(param_name);
    if (var->IsType<framework::LoDTensor>() ||
        var->IsType<framework::Tensor>()) {
      auto* t = var->GetMutable<framework::LoDTensor>();
      if (t->dtype() != phi::DataType::FLOAT32) continue;

      framework::Tensor mixed_tensor;
      mixed_tensor.Resize(t->dims());
      auto* data = t->mutable_data<float>(platform::CPUPlace());

      if (mixed_precision == phi::DataType::FLOAT16 &&
          !weights_should_be_fp32.count(param_name)) {
        mixed_tensor.set_type(paddle::experimental::DataType::FLOAT16);
        auto* mixed_data =
            mixed_tensor.mutable_data<float16>(platform::CPUPlace());
        for (int i = 0; i < t->numel(); i++) {
          mixed_data[i] = static_cast<float16>(data[i]);
        }
        t->clear();
        paddle::framework::TensorCopySync(mixed_tensor, place, t);
      } else if (mixed_precision == phi::DataType::BFLOAT16 &&
                 !weights_should_be_fp32.count(param_name)) {
        mixed_tensor.set_type(paddle::experimental::DataType::BFLOAT16);
        auto* mixed_data =
            mixed_tensor.mutable_data<bfloat16>(platform::CPUPlace());
        for (int i = 0; i < t->numel(); i++) {
          mixed_data[i] = static_cast<bfloat16>(data[i]);
        }
        t->clear();
        paddle::framework::TensorCopySync(mixed_tensor, place, t);
      }
    }
  }

  StrToBinary(mixed_model_file,
              mixed_program_desc->Proto()->SerializeAsString());
  StrToBinary(mixed_params_file, SerializeParams(scope, parameters));
}

bool PhiKernelSupportPrecision(
    const std::string& op_type,
    phi::Backend backend,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto kernels = phi::KernelFactory::Instance().kernels();
  if (kernels.find(op_type) == kernels.end()) {
    return false;
  }
  phi::KernelKey kernel_key(backend, layout, data_type);
  return phi::KernelFactory::Instance().HasKernel(op_type, kernel_key);
}

bool GpuKernelSupportPrecision(
    const std::string& op_type,
    phi::DataType data_type,
    phi::DataLayout layout = phi::DataLayout::ALL_LAYOUT) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);
  bool res = PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPU, data_type, layout);
  res |= PhiKernelSupportPrecision(
      phi_op_type, phi::Backend::GPUDNN, data_type, layout);

  if (!res) {
    auto& all_kernels = OperatorWithKernel::AllOpKernels();
    auto it = all_kernels.find(op_type);
    if (it != all_kernels.end()) {
      for (auto& kern_pair : it->second) {
        if (platform::is_gpu_place(kern_pair.first.place_) &&
            kern_pair.first.data_type_ == framework::proto::VarType::FP16) {
          res = true;
        }
      }
    }
  }
  return res;
}

// Just process special cases.
bool OutShouldNotConvert(ir::Node* var_node) {
  auto op_node = var_node->inputs[0];
  auto* op_desc = op_node->Op();

  // batch_norm's input and output (variance and mean) are the same.
  if (op_desc->Type() == "batch_norm") {
    auto vecs = op_desc->Output("MeanOut");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("VarianceOut");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("SavedMean");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
    vecs = op_desc->Output("SavedVariance");
    if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
      return true;
    }
  }

  return false;
}
void ProcessOutputNode(
    const std::vector<framework::ir::Graph*>& graphes,
    int block_idx,
    ir::Node* var_node,
    framework::proto::VarType::Type to_type,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map) {
  auto* real_node =
      GetRealNode(graphes, block_idx, var_node, vars_in_multi_block_map);
  if (!NodeVarHasDtype(real_node)) return;
  auto* out_var = real_node->Var();
  if (out_var->GetDataType() == framework::proto::VarType::FP32) {
    if (OutShouldNotConvert(var_node)) return;
    out_var->SetDataType(to_type);
  }
  VLOG(3) << " out_node name " << var_node->Name() << " data_type "
          << out_var->GetDataType();
}

// Just process special cases for weights conversion.
bool WeightsShouldNotConvert(ir::Node* var_node) {
  auto op_nodes = var_node->outputs;
  for (auto* op_node : op_nodes) {
    auto* op_desc = op_node->Op();
    // batch_norm op's bias, mean, scale and variance just be float32, so we can
    // not convert the dtype.
    if (op_desc->Type() == "batch_norm") {
      auto vecs = op_desc->Input("Bias");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
      vecs = op_desc->Input("Mean");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
      vecs = op_desc->Input("Scale");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
      vecs = op_desc->Input("Variance");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
    } else if (op_desc->Type() == "fused_multi_transformer") {
      auto vecs = op_desc->Input("LnScale");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }

      vecs = op_desc->Input("LnBias");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }

      vecs = op_desc->Input("FFNLnScale");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }

      vecs = op_desc->Input("FFNLnBias");
      if (std::find(vecs.begin(), vecs.end(), var_node->Name()) != vecs.end()) {
        return true;
      }
    }
  }

  return false;
}
inline bool IsFloatVarType(framework::proto::VarType::Type type) {
  if (type == framework::proto::VarType::FP16 ||
      type == framework::proto::VarType::FP32 ||
      type == framework::proto::VarType::BF16)
    return true;
  return false;
}
void ProcessInputNode(
    bool support_precision,
    std::vector<framework::ir::Graph*> graphes,
    ir::Node* in_node,
    ir::Node* op_node,
    int* suffix,
    framework::BlockDesc* block_desc,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* cast_map,
    framework::proto::VarType::Type to_type,
    int block_idx,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map) {
  auto* real_node =
      GetRealNode(graphes, block_idx, in_node, vars_in_multi_block_map);
  if (!NodeVarHasDtype(real_node)) return;
  auto graph = graphes[block_idx];
  bool is_main_block = block_idx == 0;
  auto* in_var = real_node->Var();
  auto in_var_type = in_var->GetDataType();
  bool is_in_multi_block = vars_in_multi_block_map->count(in_var->Name());

  if (!is_main_block && is_in_multi_block) {
    in_var_type = vars_in_multi_block_map->at(in_var->Name()).first;
  }
  if (support_precision) {
    if (in_var->Persistable() &&
        in_var_type == framework::proto::VarType::FP32) {
      if (WeightsShouldNotConvert(in_node)) return;
      in_var->SetDataType(to_type);
      in_var_type = to_type;
    } else if (!in_var->Persistable() && IsFloatVarType(in_var_type) &&
               in_var_type != to_type) {
      AddCastOp(graph,
                in_node,
                op_node,
                in_var_type,
                to_type,
                suffix,
                block_desc,
                cast_map);
    }
  } else {
    if (!in_var->Persistable() && IsFloatVarType(in_var_type) &&
        in_var_type != to_type) {
      AddCastOp(graph,
                in_node,
                op_node,
                in_var_type,
                to_type,
                suffix,
                block_desc,
                cast_map);
    }
  }
  VLOG(3) << " in_node name " << in_var->Name() << " data_type " << in_var_type;
}

void ConvertAllFp64ToFp32(framework::ir::Graph* graph) {
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type == "feed" || op_type == "fetch") continue;

    if (op_type == "fill_constant") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(framework::proto::VarType::FP64))
        op_node->Op()->SetAttr(
            "dtype", static_cast<int>(framework::proto::VarType::FP32));
    } else if (op_type == "assign_value") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(framework::proto::VarType::FP64))
        op_node->Op()->SetAttr(
            "dtype", static_cast<int>(framework::proto::VarType::FP32));
    } else if (op_type == "eye") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(framework::proto::VarType::FP64))
        op_node->Op()->SetAttr(
            "dtype", static_cast<int>(framework::proto::VarType::FP32));
    } else if (op_type == "fill_any_like") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("dtype")) ==
          static_cast<int>(framework::proto::VarType::FP64))
        op_node->Op()->SetAttr(
            "dtype", static_cast<int>(framework::proto::VarType::FP32));
    } else if (op_type == "cast") {
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("in_dtype")) ==
          static_cast<int>(framework::proto::VarType::FP64))
        op_node->Op()->SetAttr(
            "in_dtype", static_cast<int>(framework::proto::VarType::FP32));
      if (PADDLE_GET_CONST(int, op_node->Op()->GetAttr("out_dtype")) ==
          static_cast<int>(framework::proto::VarType::FP64))
        op_node->Op()->SetAttr(
            "out_dtype", static_cast<int>(framework::proto::VarType::FP32));
    }

    auto inputs = op_node->inputs;
    for (auto* in_node : inputs) {
      if (in_node->IsCtrlVar()) continue;
      auto* in_var = in_node->Var();
      if (!in_var->Persistable() &&
          in_var->GetDataType() == framework::proto::VarType::FP64) {
        in_var->SetDataType(framework::proto::VarType::FP32);
      }
    }
  }
}

// Handle special ops which contains dtype attribute. e.g., fill_constant,
// assign_value.
void HandleSpecialOps(framework::OpDesc* op_desc) {
  if (op_desc->Type() == "fill_constant") {
    if (PADDLE_GET_CONST(int, op_desc->GetAttr("dtype")) ==
        static_cast<int>(framework::proto::VarType::FP32))
      op_desc->SetAttr("dtype",
                       static_cast<int>(framework::proto::VarType::FP16));
  } else if (op_desc->Type() == "assign_value") {
    if (PADDLE_GET_CONST(int, op_desc->GetAttr("dtype")) ==
        static_cast<int>(framework::proto::VarType::FP32))
      op_desc->SetAttr("dtype",
                       static_cast<int>(framework::proto::VarType::FP16));
  } else if (op_desc->Type() == "eye") {
    if (PADDLE_GET_CONST(int, op_desc->GetAttr("dtype")) ==
        static_cast<int>(framework::proto::VarType::FP32))
      op_desc->SetAttr("dtype",
                       static_cast<int>(framework::proto::VarType::FP16));
  } else if (op_desc->Type() == "fill_any_like") {
    if (PADDLE_GET_CONST(int, op_desc->GetAttr("dtype")) ==
        static_cast<int>(framework::proto::VarType::FP32))
      op_desc->SetAttr("dtype",
                       static_cast<int>(framework::proto::VarType::FP16));
  } else if (op_desc->Type() == "fill_constant_batch_size_like") {
    if (PADDLE_GET_CONST(int, op_desc->GetAttr("dtype")) ==
        static_cast<int>(framework::proto::VarType::FP32))
      op_desc->SetAttr("dtype",
                       static_cast<int>(framework::proto::VarType::FP16));
  }
}

// We modify op's input output precision, and we need to fix cast op in_dtype
// and out_dtype attribute.
void FixCastAttr(framework::ir::Graph* graph) {
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    if (op_type != "cast") continue;

    auto input = op_node->inputs[0];
    auto output = op_node->outputs[0];
    op_node->Op()->SetAttr("in_dtype",
                           static_cast<int>(input->Var()->GetDataType()));
    op_node->Op()->SetAttr("out_dtype",
                           static_cast<int>(output->Var()->GetDataType()));
  }
}

void FindVarsInMultiBlock(
    framework::ProgramDesc* program_desc,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map,
    std::vector<std::set<std::string>>* vars_appear_multi_in_one_block) {
  std::vector<std::set<std::string>> block_var_names_set(program_desc->Size());
  for (size_t i = 0; i < program_desc->Size(); ++i) {
    for (auto op : program_desc->Block(i).AllOps()) {
      auto in_names = op->InputArgumentNames();
      block_var_names_set[i].insert(in_names.begin(), in_names.end());
      auto out_names = op->OutputArgumentNames();
      if (op->HasAttr("sub_block") == false) {
        for (auto& n : out_names) {
          if (block_var_names_set[i].count(n)) {
            (*vars_appear_multi_in_one_block)[i].insert(n);
          }
        }
      }
      block_var_names_set[i].insert(out_names.begin(), out_names.end());
    }
  }

  for (size_t i = 0; i < program_desc->Size() - 1; ++i) {
    for (size_t j = i + 1; j < program_desc->Size(); ++j) {
      std::set<std::string> vars_in_multi_block;
      std::set_intersection(
          block_var_names_set[i].begin(),
          block_var_names_set[i].end(),
          block_var_names_set[j].begin(),
          block_var_names_set[j].end(),
          std::inserter(vars_in_multi_block, vars_in_multi_block.begin()));

      for (auto name : vars_in_multi_block) {
        vars_in_multi_block_map->emplace(
            name, std::make_pair(framework::proto::VarType::FP32, i));
      }
    }
  }
}

bool OpInOutHasTensorArray(
    std::vector<framework::ir::Graph*> graphes,
    int block_idx,
    framework::ir::Node* op_node,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map) {
  CHECK_EQ(op_node->IsOp(), true);
  for (auto in : op_node->inputs) {
    auto* real_node =
        GetRealNode(graphes, block_idx, in, vars_in_multi_block_map);
    if (!NodeVarHasDtype(real_node)) continue;
    if (real_node->Var()->GetType() ==
        framework::proto::VarType::LOD_TENSOR_ARRAY)
      return true;
  }

  for (auto out : op_node->outputs) {
    auto* real_node =
        GetRealNode(graphes, block_idx, out, vars_in_multi_block_map);
    if (!NodeVarHasDtype(real_node)) continue;

    if (real_node->Var()->GetType() ==
        framework::proto::VarType::LOD_TENSOR_ARRAY)
      return true;
  }
  return false;
}

void ConvertTensorDtype(
    framework::ProgramDesc* program_desc,
    std::vector<framework::ir::Graph*> graphes,
    const std::unordered_set<std::string>& blacklist,
    bool keep_io_types,
    phi::Backend backend,
    phi::DataType tensor_dtype,
    int block_idx,
    std::unordered_map<std::string,
                       std::pair<framework::proto::VarType::Type, int>>*
        vars_in_multi_block_map,
    const std::vector<std::set<std::string>>& vars_appear_multi_in_one_block) {
  auto graph = graphes[block_idx];
  framework::proto::VarType::Type to_type;
  if (tensor_dtype == phi::DataType::FLOAT16) {
    to_type = framework::proto::VarType::FP16;
  } else if (tensor_dtype == phi::DataType::BFLOAT16) {
    to_type = framework::proto::VarType::BF16;
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only "
        "support fp16 and bf16.",
        static_cast<int>(tensor_dtype)));
  }

  auto* block_desc =
      framework::ir::TopologySortOperations(*graph)[0]->Op()->Block();

  int num_low_precision = 0;
  int suffix = 0;
  std::vector<framework::ir::Node*> output_nodes;
  std::unordered_map<framework::ir::Node*, framework::ir::Node*> cast_map;
  auto op_nodes = framework::ir::TopologySortOperations(*graph);
  for (auto* op_node : op_nodes) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    VLOG(3) << "-------------------- op_type " << op_type << ", phi_type "
            << phi::TransToPhiKernelName(op_type);
    // 1. set input dtype.
    if (op_type == "feed") {
      auto feed_var = op_node->outputs[0]->Var();
      if (!keep_io_types &&
          feed_var->GetDataType() == framework::proto::VarType::FP32) {
        feed_var->SetDataType(to_type);
      }
    } else if (op_type == "fetch") {
      auto* fetch_var = op_node->inputs[0];
      output_nodes.push_back(fetch_var);
      continue;
    } else if (op_type == "cast") {
      continue;
    }

    else if (op_node->Op()->HasAttr("sub_block")) {  // NOLINT
      // sub_block op's output dtype should be same as input dtype, if have the
      // same name.
      std::unordered_map<std::string, framework::ir::Node*> in_name_to_node;
      for (auto* in : op_node->inputs) {
        auto* real_node =
            GetRealNode(graphes, block_idx, in, vars_in_multi_block_map);
        if (NodeVarHasDtype(real_node)) {
          in_name_to_node[in->Name()] = in;
        }
      }

      for (auto out : op_node->outputs) {
        auto* real_node =
            GetRealNode(graphes, block_idx, out, vars_in_multi_block_map);
        if (NodeVarHasDtype(real_node)) {
          if (in_name_to_node.count(out->Name()))
            real_node->Var()->SetDataType(
                in_name_to_node[out->Name()]->Var()->GetDataType());
        }
      }

      continue;
    }

    // A strange case found in multi block.
    else if (op_type == "assign" &&  // NOLINT
             op_node->inputs[0]->Name() == op_node->outputs[0]->Name()) {
      VLOG(2) << " in out are same, continue";
      continue;
    }

    // Handle tensor array.
    else if (OpInOutHasTensorArray(  // NOLINT
                 graphes,
                 block_idx,
                 op_node,
                 vars_in_multi_block_map)) {
      VLOG(2) << "  in or out has tensor array, continue";
      continue;
    }

    // 2. if op support fp16/bf16 and not in blacklist.
    //      - cast weight to fp16/bf16.
    //      - add cast op if the input dtype is not fp16/bf16.
    //      - set output dtype.
    //
    // If a var(op's out var) appears multiple times in a block, we should not
    // convert to fp16.
    else if (blacklist.count(op_type) == 0 &&  // NOLINT
             !VarIsMultiOpsOut(graphes,
                               block_idx,
                               op_node,
                               vars_in_multi_block_map,
                               vars_appear_multi_in_one_block)) {
      bool support_precision =
          OpSupportPrecision(op_type, backend, tensor_dtype, blacklist);
      VLOG(2) << " support low precision " << support_precision;

      // if op not has float input, we will not choose the low precision kernel.
      {
        bool has_float_input{false};
        for (auto in_node : op_node->inputs) {
          auto* real_node =
              GetRealNode(graphes, block_idx, in_node, vars_in_multi_block_map);
          if (real_node->Var()->GetDataType() == proto::VarType::FP16 ||
              real_node->Var()->GetDataType() == proto::VarType::FP32 ||
              real_node->Var()->GetDataType() == proto::VarType::FP64 ||
              real_node->Var()->GetDataType() == proto::VarType::BF16) {
            has_float_input = true;
            break;
          }
        }
        if (!has_float_input) {
          support_precision = false;
          VLOG(2) << " op doesn't has float input, just skip.";
        }
      }

      if (support_precision) {
        HandleSpecialOps(op_node->Op());
        ++num_low_precision;
        auto inputs = op_node->inputs;
        // Process inputs.
        for (auto* in_node : inputs) {
          ProcessInputNode(true,
                           graphes,
                           in_node,
                           op_node,
                           &suffix,
                           block_desc,
                           &cast_map,
                           to_type,
                           block_idx,
                           vars_in_multi_block_map);
        }
        // Process outputs.
        for (auto* out_node : op_node->outputs) {
          ProcessOutputNode(
              graphes, block_idx, out_node, to_type, vars_in_multi_block_map);
        }
      } else {
        auto inputs = op_node->inputs;
        for (auto* in_node : inputs) {
          ProcessInputNode(false,
                           graphes,
                           in_node,
                           op_node,
                           &suffix,
                           block_desc,
                           &cast_map,
                           framework::proto::VarType::FP32,
                           block_idx,
                           vars_in_multi_block_map);
        }
      }
    }

    // 3. check op not support fp16/bf16 or in blacklist.
    //      - add cast op if the input dtype is not fp32.
    else {  // NOLINT
      auto ins = op_node->inputs;
      for (auto* in_node : ins) {
        if (in_node->IsCtrlVar()) continue;
        auto* in_var = in_node->Var();
        if (in_var->GetDataType() == to_type) {
          AddCastOp(graph,
                    in_node,
                    op_node,
                    to_type,
                    framework::proto::VarType::FP32,
                    &suffix,
                    block_desc,
                    &cast_map);
        }
      }
    }
  }

  // 4. if output_op's dtype is not compatible to output dtype, then just
  // insert cast.
  for (auto* node : output_nodes) {
    if (node->IsCtrlVar()) continue;
    auto var = node->Var();
    if (keep_io_types && var->GetDataType() == to_type) {
      // fp16/bf16 -> fp32.
      AddCastOp(graph,
                node,
                node->outputs[0],
                to_type,
                framework::proto::VarType::FP32,
                &suffix,
                block_desc,
                &cast_map);
    } else if (!keep_io_types &&
               var->GetDataType() == framework::proto::VarType::FP32) {
      // fp32 -> fp16/bf16
      AddCastOp(graph,
                node,
                node->outputs[0],
                framework::proto::VarType::FP32,
                to_type,
                &suffix,
                block_desc,
                &cast_map);
    }
  }

  for (auto node : graph->Nodes()) {
    auto* real_node =
        GetRealNode(graphes, block_idx, node, vars_in_multi_block_map);
    if (!NodeVarHasDtype(real_node)) continue;

    if (vars_in_multi_block_map->count(real_node->Name()) &&
        vars_in_multi_block_map->at(real_node->Name()).second == block_idx) {
      vars_in_multi_block_map->at(real_node->Name()).first =
          real_node->Var()->GetDataType();
    }
  }

  if (num_low_precision)
    LOG(INFO) << "---  detected " << num_low_precision
              << " low precision ops in " << block_idx << " subgraph";
}
}  // namespace

bool OpSupportPrecision(const std::string& op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& blacklist) {
  auto phi_op_type = phi::TransToPhiKernelName(op_type);
  bool support_precision = false;
  if (blacklist.count(op_type) == 0) {
    if (backend == phi::Backend::GPU)
      support_precision = GpuKernelSupportPrecision(op_type, precision);
    else
      support_precision =
          PhiKernelSupportPrecision(phi_op_type, backend, precision);
  }
  return support_precision;
}

void AddCastOp(
    framework::ir::Graph* graph,
    framework::ir::Node* node,
    framework::ir::Node* next_op,
    framework::proto::VarType::Type from_type,
    framework::proto::VarType::Type to_type,
    int* suffix,
    framework::BlockDesc* block_desc,
    std::unordered_map<framework::ir::Node*, framework::ir::Node*>* map) {
  auto update_cast_desc = [&](framework::OpDesc& desc,
                              const std::string& x_name,
                              const std::string& out_name,
                              const int in_dtype,
                              const int out_dtype) {
    desc.SetType("cast");
    desc.SetInput("X", {x_name});
    desc.SetOutput("Out", {out_name});
    desc.SetAttr("in_dtype", in_dtype);
    desc.SetAttr("out_dtype", out_dtype);
    desc.SetAttr("use_mkldnn", false);
    desc.SetAttr("with_quant_attr", false);
    desc.Flush();
  };

  if (map->count(node) == 0) {
    // insert cast op before node.
    std::string cast_input_name = node->Var()->Name();
    std::string cast_output_name =
        node->Var()->Name() + "_cast.tmp_" + std::to_string((*suffix)++);
    CHECK_NOTNULL(block_desc);
    framework::OpDesc cast_op_desc(block_desc);
    update_cast_desc(cast_op_desc,
                     cast_input_name,
                     cast_output_name,
                     static_cast<int>(from_type),
                     static_cast<int>(to_type));
    auto* cast_op_node = graph->CreateOpNode(&cast_op_desc);
    auto* cast_output_vardesc = block_desc->Var(cast_output_name);
    cast_output_vardesc->SetPersistable(false);
    cast_output_vardesc->SetDataType(to_type);
    cast_output_vardesc->SetShape(node->Var()->GetShape());
    auto* cast_output_node = graph->CreateVarNode(cast_output_vardesc);
    IR_NODE_LINK_TO(cast_op_node, cast_output_node);
    (*map)[node] = cast_output_node;
  }
  next_op->Op()->RenameInput(node->Name(), map->at(node)->Name());
  IR_NODE_LINK_TO(node, map->at(node)->inputs[0]);
  IR_NODE_LINK_TO(map->at(node), next_op);
}

void ConvertToMixedPrecision(const std::string& model_file,
                             const std::string& params_file,
                             const std::string& mixed_model_file,
                             const std::string& mixed_params_file,
                             phi::DataType mixed_precision,
                             phi::Backend backend,
                             bool keep_io_types,
                             std::unordered_set<std::string> black_list) {
  paddle::CPUPlace place;
  framework::Executor executor(place);
  framework::Scope scope;
  auto program_desc =
      inference::Load(&executor, &scope, model_file, params_file);
  auto main_graph = std::unique_ptr<framework::ir::Graph>(
      new framework::ir::Graph(*program_desc));

  std::unordered_map<std::string,
                     std::pair<framework::proto::VarType::Type, int>>
      vars_in_multi_block_map;
  std::vector<std::set<std::string>> vars_appear_multi_in_one_block(
      program_desc->Size());
  FindVarsInMultiBlock(program_desc.get(),
                       &vars_in_multi_block_map,
                       &vars_appear_multi_in_one_block);

  std::vector<framework::ir::Graph*> graphes;
  for (size_t i = 0; i < main_graph->SubGraphsSize(); ++i) {
    auto graph = main_graph->GetSubGraph(i);
    graphes.push_back(graph);
    VLOG(2) << " --------  handle subgraph " << i << ", has "
            << graph->Nodes().size() << " nodes --------";

    ConvertAllFp64ToFp32(graph);
    ConvertTensorDtype(program_desc.get(),
                       graphes,
                       black_list,
                       keep_io_types,
                       backend,
                       mixed_precision,
                       i,
                       &vars_in_multi_block_map,
                       vars_appear_multi_in_one_block);
    FixCastAttr(graph);
  }

  framework::ProgramDesc mixed_program_desc;
  framework::ir::GraphToProgram(*main_graph, &mixed_program_desc);

  SaveMixedModel(main_graph.get(),
                 &scope,
                 &mixed_program_desc,
                 mixed_model_file,
                 mixed_params_file,
                 mixed_precision,
                 vars_in_multi_block_map);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
