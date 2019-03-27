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

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes);

void RenameAndGetOutputs(
    const std::vector<framework::ir::Node *> &subgraph_nodes,
    framework::BlockDesc *block_desc,
    const std::set<std::string> &input_names_with_id,
    std::set<std::string> *output_names_with_id,
    std::set<std::string> *output_names,
    std::unordered_map<std::string, std::string> *output_name_map);

std::unique_ptr<framework::ir::Graph> analysis::TensorRtSubgraphPass::ApplyImpl(
    std::unique_ptr<framework::ir::Graph> graph) const {
  framework::ir::FusePassBase::Init("tensorrt_subgraph_pass", graph.get());

  auto teller = [](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) return false;
    return tensorrt::OpTeller::Global().Tell(node->Op()->Type(), *node->Op());
  };

  SubGraphFuser fuser(graph.get(), teller,
                      Get<int>("min_subgraph_size") /*min subgraph size*/);
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in trt, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !Agent(node).subgraph()->empty()) {
      CreateTensorRTOp(node, graph.get(), graph_param_names,
                       &repetitive_params);

      std::unordered_set<const Node *> nodes2remove(
          Agent(node).subgraph()->begin(), Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph.get(), nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph.get(), nodes2remove);
  graph->Set(framework::ir::kRepetitiveParamAttr,
             new std::vector<std::string>(repetitive_params));

  return graph;
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
  SetAttr(op_desc->Proto(), "gpu_id", Get<int>("gpu_device_id"));
  SetAttr(op_desc->Proto(), "output_name_mapping", output_mapping);
  SetAttr(op_desc->Proto(), "parameters", params);

  auto enable_int8 = Get<bool>("enable_int8");
  auto use_static_engine = Get<bool>("use_static_engine");
  auto engine_key = GenerateEngineKey(input_names_with_id, output_names_with_id,
                                      std::to_string(0));

  // Get "" when there is no cached calibration table data.
  bool load_from_memory = Get<bool>("model_from_memory");
  std::string calibration_data = "";
  if (!load_from_memory && use_static_engine) {
    calibration_data = GetTrtCalibTableData(
        Get<std::string>("model_opt_cache_dir"), engine_key, enable_int8);
  }
  SetAttr(op_desc->Proto(), "calibration_data", calibration_data);

  SetAttr(op_desc->Proto(), "enable_int8", enable_int8);
  SetAttr(op_desc->Proto(), "engine_key", engine_key);
  std::string trt_engine_serialized_data = "";
  if (load_from_memory) {
    std::map<std::string, std::string> engine_opt_info =
        Get<std::map<std::string, std::string>>("engine_opt_info");
    if (engine_opt_info.count(engine_key)) {
      trt_engine_serialized_data = engine_opt_info[engine_key];
    }
  }
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

std::vector<std::string> ExtractParameters(
    const std::unordered_set<Node *> &nodes) {
  // We can judge whether a variable is a parameter by
  // its presistable property, but sometimes the presistable
  // of the feed op output is true, so we have to identify it.
  std::vector<std::string> feed_outputs;
  for (const auto &node : nodes) {
    if (!node->IsOp()) continue;
    std::string op_type = node->Op()->Type();
    if (op_type == "feed" || op_type == "fetch") {
      std::vector<std::string> output_names = node->Op()->OutputArgumentNames();
      std::copy(output_names.begin(), output_names.end(),
                std::back_inserter(feed_outputs));
    }
  }

  std::vector<std::string> parameters;
  for (const auto &node : nodes) {
    if (!node->IsVar()) continue;
    if (node->Var()->Persistable() &&
        std::find(feed_outputs.begin(), feed_outputs.end(), node->Name()) ==
            feed_outputs.end()) {
      parameters.push_back(node->Name());
    }
  }
  return parameters;
}

void RenameAndGetOutputs(
    const std::vector<framework::ir::Node *> &subgraph_nodes,
    framework::BlockDesc *block_desc,
    const std::set<std::string> &input_names_with_id,
    std::set<std::string> *output_names_with_id,
    std::set<std::string> *output_names,
    std::unordered_map<std::string, std::string> *output_name_map) {
  //// In the normal case, the paddle-trt exists bug when runing the googlenet.
  // When there are more than two convolutions of 1 * 1 with the same input, the
  // paddle-tensorrt will do the merging optimization, which fuse those conv
  // into one conv, and then trigger bug. So,  We should use strategy to avoid
  // this optimization for the time being. This bug will be fixed in the future.
  std::unordered_map<std::string /*name*/, int /*ITensor_quote_num*/>
      same_hierarchy_conv2d_num_map;

  for (size_t index = 0; index < block_desc->OpSize(); ++index) {
    framework::proto::OpDesc *op = block_desc->Op(index)->Proto();
    framework::OpDesc op_desc(*op, nullptr);
    auto correspond_node = subgraph_nodes[index];
    PADDLE_ENFORCE_EQ(correspond_node->Name(), op->type());

    std::unordered_map<std::string, size_t> var2id;
    std::unordered_map<std::string, framework::ir::Node *> in_vars;
    for (auto *in_var : correspond_node->inputs) {
      var2id[in_var->Name()] = in_var->id();
      in_vars[in_var->Name()] = in_var;
    }
    // rename for the input variables of op inside subgraph
    for (int i = 0; i < op->inputs_size(); i++) {
      // one input
      auto *in_var = op->mutable_inputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < in_var->arguments_size(); k++) {  // all the arguments
        std::string arg_value = in_var->arguments(k);
        std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);
        if (input_names_with_id.count(arg_value_with_id)) {
          replaced_names.push_back(arg_value);
        } else {
          replaced_names.push_back(arg_value_with_id);
        }
      }
      in_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        in_var->add_arguments(replaced_names[k]);
      }
    }
    var2id.clear();
    for (auto out_var : correspond_node->outputs) {
      var2id[out_var->Name()] = out_var->id();
    }

    if (op_desc.Type() == "conv2d") {
      auto input_var_name = op_desc.Input("Input").front();
      auto filter_var_name = op_desc.Input("Filter").front();
      auto out_var_name = op_desc.Output("Output").front();
      auto filter_shape = in_vars[filter_var_name]->Var()->GetShape();
      const std::vector<int> strides =
          boost::get<std::vector<int>>(op_desc.GetAttr("strides"));
      const std::vector<int> paddings =
          boost::get<std::vector<int>>(op_desc.GetAttr("paddings"));
      if (same_hierarchy_conv2d_num_map[input_var_name] > 0) {
        (*output_names_with_id)
            .insert(out_var_name + std::to_string(var2id[out_var_name]));
        (*output_names).insert(out_var_name);
      } else if (filter_shape[2] == 1 && filter_shape[3] == 1 &&
                 strides[0] == 1 && strides[1] == 1 && paddings[0] == 0 &&
                 paddings[1] == 0) {
        same_hierarchy_conv2d_num_map[input_var_name] += 1;
      }
    }

    // rename for the output variables of op inside subgraph
    for (int i = 0; i < op->outputs_size(); i++) {
      framework::proto::OpDesc_Var *out_var = op->mutable_outputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < out_var->arguments_size(); k++) {
        std::string arg_value = out_var->arguments(k);
        std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);
        if (output_names_with_id->count(arg_value_with_id)) {
          (*output_name_map)[arg_value] = arg_value_with_id;
        }
        replaced_names.push_back(arg_value_with_id);
      }
      out_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        out_var->add_arguments(replaced_names[k]);
      }
    }
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(tensorrt_subgraph_pass,
              paddle::inference::analysis::TensorRtSubgraphPass)
    .RequirePassAttr("max_batch_size")
    .RequirePassAttr("workspace_size")
    .RequirePassAttr("min_subgraph_size");
