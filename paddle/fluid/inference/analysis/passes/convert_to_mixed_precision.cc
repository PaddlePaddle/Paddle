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

#include <unordered_set>

#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/ir/graph.h"
#include "paddle/fluid/framework/ir/graph_helper.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/layout.h"
#include "paddle/phi/core/tensor_meta.h"

using namespace paddle::framework;  // NOLINT

namespace paddle {
namespace inference {
namespace analysis {

namespace {

bool IsKernelSupportPrecision(
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
  bool res =
      IsKernelSupportPrecision(op_type, phi::Backend::GPU, data_type, layout);
  res |= IsKernelSupportPrecision(
      op_type, phi::Backend::GPUDNN, data_type, layout);
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
    }
  }

  return false;
}

void ConvertTensorDtype(framework::ir::Graph* graph,
                        const std::unordered_set<std::string>& blacklist,
                        bool keep_io_types,
                        phi::Backend backend,
                        phi::DataType tensor_dtype) {
  framework::proto::VarType::Type to_type;
  if (tensor_dtype == phi::DataType::FLOAT16) {
    to_type = framework::proto::VarType::FP16;
  } else if (tensor_dtype == phi::DataType::BFLOAT16) {
    to_type = framework::proto::VarType::BF16;
  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "mixed_precision currently not supported dtype %d, we now only support "
        "fp16 and bf16.",
        static_cast<int>(tensor_dtype)));
  }

  int num_low_precision = 0;
  int suffix = 0;
  framework::BlockDesc* block_desc{nullptr};
  std::vector<framework::ir::Node*> output_nodes;
  std::unordered_map<framework::ir::Node*, framework::ir::Node*> cast_map;

  for (auto* op_node : framework::ir::TopologySortOperations(*graph)) {
    if (!op_node->IsOp()) continue;
    auto op_type = op_node->Op()->Type();
    auto phi_op_type = phi::TransToPhiKernelName(op_type);
    // LOG(INFO) << "process op " << op_type << ", corresponding phi type is "
    // << phi_op_type;
    // 1. set input dtype.
    if (op_type == "feed") {
      block_desc = op_node->Op()->Block();
      auto feed_var = op_node->outputs[0]->Var();
      if (!keep_io_types &&
          feed_var->GetDataType() == framework::proto::VarType::FP32) {
        feed_var->SetDataType(to_type);
      }
    } else if (op_type == "fetch") {
      auto* fetch_var = op_node->inputs[0];
      output_nodes.push_back(fetch_var);
      continue;
    }

    // 2. if op support fp16/bf16 and not in blacklist.
    //      - cast weight to fp16/bf16.
    //      - add cast op if the input dtype is not fp16/bf16.
    //      - set output dtype.
    else if (blacklist.count(phi_op_type) == 0) {  // NOLINT
      bool support_precision =
          OpSupportPrecision(phi_op_type, backend, tensor_dtype, blacklist);
      VLOG(2) << "phi_op_type " << phi_op_type << " support low precision "
              << support_precision;
      if (support_precision) {
        ++num_low_precision;
        auto inputs = op_node->inputs;
        for (auto* in_node : inputs) {
          auto* in_var = in_node->Var();
          if (in_var->Persistable() &&
              in_var->GetDataType() == framework::proto::VarType::FP32) {
            if (WeightsShouldNotConvert(in_node)) continue;
            in_var->SetDataType(to_type);
          } else if (!in_var->Persistable() &&
                     in_var->GetDataType() != to_type) {
            AddCastOp(graph,
                      in_node,
                      op_node,
                      in_var->GetDataType(),
                      to_type,
                      &suffix,
                      block_desc,
                      &cast_map);
          }
        }
        for (auto* out_node : op_node->outputs) {
          auto* out_var = out_node->Var();
          if (out_var->GetDataType() == framework::proto::VarType::FP32) {
            if (OutShouldNotConvert(out_node)) continue;
            out_var->SetDataType(to_type);
          }
        }
      } else {
        auto inputs = op_node->inputs;
        for (auto* in_node : inputs) {
          auto* in_var = in_node->Var();
          if (!in_var->Persistable() &&
              in_var->GetDataType() != framework::proto::VarType::FP32) {
            AddCastOp(graph,
                      in_node,
                      op_node,
                      in_var->GetDataType(),
                      framework::proto::VarType::FP32,
                      &suffix,
                      block_desc,
                      &cast_map);
          }
        }
      }
    }

    // 3. check op not support fp16/bf16 or in blacklist.
    //      - add cast op if the input dtype is not fp32.
    else {  // NOLINT
      // trt pass should explicitle add cast op is input is bf16/tf32, etc.
      if (op_node->Name() == "tensorrt_engine") continue;
      for (auto* in_node : op_node->inputs) {
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

  // 4. if output_op's dtype is not compatible to output dtype, then just insert
  // cast.
  for (auto* node : output_nodes) {
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

  if (num_low_precision)
    LOG(INFO) << "---  detected " << num_low_precision << " low precision ops";
}
}  // namespace

bool OpSupportPrecision(const std::string& phi_op_type,
                        phi::Backend backend,
                        phi::DataType precision,
                        const std::unordered_set<std::string>& blacklist) {
  bool support_precision = false;
  if (blacklist.count(phi_op_type) == 0) {
    if (backend == phi::Backend::GPU)
      support_precision = GpuKernelSupportPrecision(phi_op_type, precision);
    else
      support_precision =
          IsKernelSupportPrecision(phi_op_type, backend, precision);
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
  auto graph = std::unique_ptr<framework::ir::Graph>(
      new framework::ir::Graph(*program_desc));

  ConvertTensorDtype(
      graph.get(), black_list, keep_io_types, backend, mixed_precision);

  framework::ProgramDesc mixed_program_desc;
  framework::ir::GraphToProgram(*graph, &mixed_program_desc);

  auto parameters = scope.LocalVarNames();
  std::sort(parameters.begin(), parameters.end());

  auto serialize_params =
      [](framework::Scope* scope,
         const std::vector<std::string>& params) -> std::string {
    std::ostringstream os;
    platform::CPUDeviceContext ctx;
    for (const auto& param : params) {
      VLOG(3) << "Serialize param: " << param;
      PADDLE_ENFORCE_NOT_NULL(
          scope->FindVar(param),
          platform::errors::NotFound(
              "Block should already have a '%s' variable", param));
      auto* tensor = scope->FindVar(param)->GetMutable<framework::LoDTensor>();
      framework::SerializeToStream(os, *tensor, ctx);
    }
    return os.str();
  };

  std::unordered_set<std::string> weights_should_be_fp32;
  for (auto* node : paddle::framework::ir::TopologySortOperations(*graph)) {
    if (!node->IsOp()) continue;
    auto* op_desc = node->Op();
    if (op_desc->Type() == "feed" || op_desc->Type() == "fetch") continue;

    if (op_desc->Type() == "batch_norm") {
      auto vecs = op_desc->Input("Bias");
      for (auto s : vecs) {
        weights_should_be_fp32.insert(s);
      }
      vecs = op_desc->Input("Mean");
      for (auto s : vecs) {
        weights_should_be_fp32.insert(s);
      }
      vecs = op_desc->Input("Scale");
      for (auto s : vecs) {
        weights_should_be_fp32.insert(s);
      }
      vecs = op_desc->Input("Variance");
      for (auto s : vecs) {
        weights_should_be_fp32.insert(s);
      }
    }
  }

  for (const auto& param_name : parameters) {
    auto* var = scope.FindLocalVar(param_name);
    if (var->IsType<framework::LoDTensor>() ||
        var->IsType<framework::Tensor>()) {
      auto* t = var->GetMutable<framework::LoDTensor>();
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

  auto StrToBinary = [](const std::string& path, const std::string& str) {
    std::ofstream file(path.c_str(), std::ios::binary);
    file.write(str.c_str(), str.size());
    file.close();
  };
  StrToBinary(mixed_model_file,
              mixed_program_desc.Proto()->SerializeAsString());
  StrToBinary(mixed_params_file, serialize_params(&scope, parameters));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
