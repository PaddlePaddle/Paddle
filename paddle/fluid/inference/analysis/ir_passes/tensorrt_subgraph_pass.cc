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

#include <algorithm>
#include <map>
#include <set>

#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/subgraph_detector.h"
#include "paddle/fluid/inference/analysis/ir_passes/tensorrt_subgraph_pass.h"
#include "paddle/fluid/inference/tensorrt/convert/op_converter.h"
#include "paddle/fluid/inference/tensorrt/engine.h"
#include "paddle/fluid/inference/tensorrt/op_teller.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

void analysis::TensorRtSubgraphPass::ApplyImpl(
    framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("tensorrt_subgraph_pass", graph);

  auto teller = [](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    return tensorrt::OpTeller::Global().Tell(node->Op()->Type(), *node->Op());
  };

  SubGraphFuser fuser(graph, teller,
                      Get<int>("min_subgraph_size") /*min subgraph size*/);
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in trt, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateTensorRTOp(node, graph, graph_param_names, &repetitive_params);

      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));
}

std::string GenerateEngineKey(const std::set<std::string> &engine_inputs,
                              const std::set<std::string> &engine_outputs,
                              const std::string &predictor_id) {
  std::string engine_hash_key = "";
  for (auto name : engine_inputs) {
    engine_hash_key += name;
  }
  for (auto name : engine_outputs) {
    engine_hash_key += name;
  }
  engine_hash_key += predictor_id;
  auto engine_key = std::to_string(std::hash<std::string>()(engine_hash_key));
  return engine_key;
}

void TensorRtSubgraphPass::CreateTensorRTOp(
    framework::ir::Node *node, Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  auto *op_desc = node->Op();
  auto &subgraph = *Agent(node).subgraph();
  PADDLE_ENFORCE(!subgraph.empty());

  framework::ProgramDesc *program_desc =
      Get<framework::ProgramDesc *>("program");
  // Add new block for TensorRTEngineOP
  const framework::BlockDesc &main_block =
      program_desc->Block(framework::kRootBlockIndex);
  // const framework::BlockDesc& main_block = program_desc->Block(0);
  framework::BlockDesc *new_block = program_desc->AppendBlock(main_block);

  // An fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  string::PrettyLogDetail("---  detect a sub-graph with %d nodes",
                          subgraph.size());

  for (auto *node : subgraph) {
    auto *new_block_op = new_block->AppendOp();
    auto *op = block_desc.AppendOp();
    *new_block_op->Proto() = *node->Op()->Proto();
    *op->Proto() = *node->Op()->Proto();
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the eigine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;

  // The node->inputs containes input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
  }

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  for (auto *x : node->outputs) {
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
  }

  std::unordered_map<std::string, std::string> output_name_map;
  auto &subgraph_nodes = *Agent(node).subgraph();

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to tensorrt OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the corresponding ITensor (trt data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.
  RenameAndGetOutputs(subgraph_nodes, &block_desc, input_names_with_id,
                      &output_names_with_id, &output_names, &output_name_map);

  // When tensorrt engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  for (auto name : output_names) {
    PADDLE_ENFORCE(output_name_map.count(name) != 0);
    output_mapping.push_back(output_name_map[name]);
  }
  PADDLE_ENFORCE(!output_mapping.empty());

  auto *vars = block_desc.Proto()->mutable_vars();
  for (framework::ir::Node *node : graph->Nodes()) {
    if (node->IsVar() && node->Var()) {
      *vars->Add() = *node->Var()->Proto();
    }
  }

  PADDLE_ENFORCE(!block_desc.Proto()->vars().empty(),
                 "the block has no var-desc");

  // Set attrs
  op_desc->SetType("tensorrt_engine");
  op_desc->SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  op_desc->SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));

  op_desc->SetBlockAttr("sub_block", new_block);
  SetAttr(op_desc->Proto(), "subgraph",
          block_desc.Proto()->SerializeAsString());
  SetAttr(op_desc->Proto(), "max_batch_size", Get<int>("max_batch_size"));
  SetAttr(op_desc->Proto(), "workspace_size", Get<int>("workspace_size"));
  SetAttr(op_desc->Proto(), "output_name_mapping", output_mapping);
  SetAttr(op_desc->Proto(), "parameters", params);

  auto enable_int8 = Get<bool>("enable_int8");
  auto use_static_engine = Get<bool>("use_static_engine");
  auto engine_key = GenerateEngineKey(input_names_with_id, output_names_with_id,
                                      std::to_string(0));

  // Get "" when there is no cached calibration table data.
  bool load_from_memory = Get<bool>("model_from_memory");
  std::string calibration_data = "";
  if (enable_int8) {
    calibration_data = GetTrtCalibTableData(
        Get<std::string>("model_opt_cache_dir"), engine_key, enable_int8);
  }
  SetAttr(op_desc->Proto(), "calibration_data", calibration_data);

  SetAttr(op_desc->Proto(), "enable_int8", enable_int8);
  SetAttr(op_desc->Proto(), "engine_key", engine_key);
  std::string trt_engine_serialized_data = "";

  SetAttr(op_desc->Proto(), "engine_serialized_data",
          trt_engine_serialized_data);

  std::unique_ptr<tensorrt::TRTInt8Calibrator> calibrator;
  if (enable_int8 && calibration_data.size() != 0) {
    calibrator.reset(new tensorrt::TRTInt8Calibrator(calibration_data));
  }
  // When in int8 mode and calibration_mode, the program just produce the
  // calibration table data.
  bool calibration_mode = (enable_int8 && calibration_data.size() == 0);
  if (calibration_mode) {
    // calibraion mode means generate int8 calibration table data process.
    return;
  }

  std::copy(params.begin(), params.end(),
            std::back_inserter(*repetitive_params));
  bool need_serialize = (use_static_engine && !load_from_memory);

  if (need_serialize) {
    trt_engine_serialized_data = GetTrtEngineSerializedData(
        Get<std::string>("model_opt_cache_dir"), engine_key);
    // we can load the engine info serialized before from the disk.
    if (!trt_engine_serialized_data.empty()) {
      SetAttr(op_desc->Proto(), "engine_serialized_data",
              trt_engine_serialized_data);
      LOG(INFO) << "Load TRT Optimized Info from "
                << GetTrtEngineSerializedPath(
                       Get<std::string>("model_opt_cache_dir"), engine_key);
      return;
    }
  }

  // the following code will NOT run in following situation:
  // 1. calibraion mode (generate trt int8 calibraiton table data)
  // 2. already load serialized trt engine info.
  LOG(INFO) << "Prepare TRT engine (Optimize model structure, Select OP "
               "kernel etc). This process may cost a lot of time.";
  std::unique_ptr<tensorrt::TensorRTEngine> trt_engine(
      new tensorrt::TensorRTEngine(
          Get<int>("max_batch_size"), Get<int>("workspace_size"), enable_int8,
          calibrator.get(), Get<int>("gpu_device_id")));
  auto *scope = param_scope();
  framework::BlockDesc block_desc_temp(nullptr, block_desc.Proto());
  std::unordered_set<std::string> param_set(params.begin(), params.end());
  inference::Singleton<inference::tensorrt::OpConverter>::Global()
      .ConvertBlockToTRTEngine(
          &block_desc_temp, *scope,
          std::vector<std::string>(input_names.begin(), input_names.end()),
          param_set, output_mapping, trt_engine.get());
  nvinfer1::IHostMemory *serialized_engine_data = trt_engine->Serialize();
  trt_engine_serialized_data =
      std::string((const char *)serialized_engine_data->data(),
                  serialized_engine_data->size());

  if (need_serialize) {
    SaveTrtEngineSerializedDataToFile(
        GetTrtEngineSerializedPath(Get<std::string>("model_opt_cache_dir"),
                                   engine_key),
        trt_engine_serialized_data);
  }
  SetAttr(op_desc->Proto(), "engine_serialized_data",
          trt_engine_serialized_data);
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");
