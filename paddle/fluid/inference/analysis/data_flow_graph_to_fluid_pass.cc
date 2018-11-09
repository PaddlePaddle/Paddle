//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/inference/analysis/data_flow_graph_to_fluid_pass.h"
#include <vector>
#include "paddle/fluid/framework/block_desc.h"
#include "paddle/fluid/framework/ir/fuse_pass_base.h"
#include "paddle/fluid/framework/op_desc.h"
#include "paddle/fluid/framework/proto_desc.h"
#include "paddle/fluid/inference/analysis/analyzer.h"
#include "paddle/fluid/inference/analysis/dfg_graphviz_draw_pass.h"
#include "paddle/fluid/inference/io.h"

namespace paddle {
namespace inference {

namespace analysis {

using framework::proto::ProgramDesc;

std::vector<std::string> ExtractParameters(
    const std::vector<std::unique_ptr<Node>> &nodes);

bool DataFlowGraphToFluidPass::Initialize(Argument *argument) {
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument)
  ANALYSIS_ARGUMENT_CHECK_FIELD(argument->origin_program_desc)
  // The transformed_program_desc should inherit all the VarDesc and BlockDesc
  // from the original program desc. The operators of the main block(the first
  // block) should rewritten by data flow graph.
  argument->transformed_program_desc.reset(
      new ProgramDesc(*argument->origin_program_desc));
  argument->transformed_program_desc->mutable_blocks(framework::kRootBlockIndex)
      ->clear_ops();
  desc_ = argument->transformed_program_desc.get();
  argument_ = argument;
  return true;
}

bool DataFlowGraphToFluidPass::Finalize() { return true; }

void DataFlowGraphToFluidPass::Run(DataFlowGraph *graph) {
  // FilterRedundantOutputOfSubGraph(graph);
  for (auto &node : GraphTraits<DataFlowGraph>(*graph).nodes_in_TS()) {
    if (node.deleted()) continue;

    switch (node.type()) {
      case Node::Type::kFunction: {
        AddFluidOp(&node);
      } break;
      case Node::Type::kFunctionBlock: {
        AddEngineOp(&node);
      } break;
      default:
        continue;
    }
  }

  if (argument_->Has(framework::ir::kParamScopeAttr)) {
    LOG(WARNING) << "parameter changes in the scope takes effect";
  }

  PADDLE_ENFORCE(argument_->transformed_program_desc.get());
}

void DataFlowGraphToFluidPass::AddFluidOp(Node *node) {
  PADDLE_ENFORCE(node);
  PADDLE_ENFORCE(node->IsFunction());
  PADDLE_ENFORCE(node->pb_desc() || !node->pb_msg().empty(),
                 "node has invalid protobuf repr.");

  // currently only the main block is analyzed.
  PADDLE_ENFORCE(desc_);
  auto *main_block = desc_->mutable_blocks(framework::kRootBlockIndex);
  auto *op = main_block->add_ops();

  if (node->pb_desc()) {
    auto *ori_op = static_cast<framework::proto::OpDesc *>(node->pb_desc());
    *op =
        *ori_op;  // copy the attributes, by default, these will not be changed
    // by analysis phrase.
    // The inputs and outputs of the existing ops are not changed by tensorrt
    // subgraph pass.
    // NOTE It might be changed by other passes in the long run.
  } else {
    op->ParseFromString(node->pb_msg());
  }
}

void CreateTrtEngineOp(Node *node, Argument *argument,
                       framework::proto::BlockDesc *block) {
  PADDLE_ENFORCE(argument->main_dfg.get());
  const DataFlowGraph &graph = *(argument->main_dfg);
  static int counter{0};
  PADDLE_ENFORCE(node->IsFunctionBlock());
  framework::OpDesc desc;
  auto *func = static_cast<FunctionBlock *>(node);

  // collect inputs
  std::unordered_set<std::string> input_names;
  std::unordered_set<std::string> input_names_with_id;
  for (auto *x : func->inlinks) {
    input_names.insert(x->name());
    input_names_with_id.insert(x->name() + std::to_string(x->id()));
  }
  desc.SetInput(
      "Xs", std::vector<std::string>(input_names.begin(), input_names.end()));

  std::unordered_set<std::string> output_names;
  std::unordered_set<std::string> output_names_with_id;
  for (auto *x : func->outlinks) {
    output_names.insert(x->name());
    output_names_with_id.insert(x->name() + std::to_string(x->id()));
  }

  desc.SetOutput(
      "Ys", std::vector<std::string>(output_names.begin(), output_names.end()));
  desc.SetType("tensorrt_engine");

  std::unordered_map<std::string, std::string> output_name_map;

  // The following procedure is used to rename all the intermediate
  // variables and the output variables of the subgraph.
  // Why we do this?
  // During the transition from fluid OP to tensorrt OP, we map
  // the input and output Tensor(fluid data structure) of fluid OP
  // to the correspondin ITensor (trt data structure) through the
  // Tensor name. When we set up ITensor for an variable, we must
  // ensure that it has not been set before.
  // If there is variable in the fluid graph, which is not only the
  // input of a OP, but also the output of a Op, there will be problems.
  // So we have to rename the variable in the subgraph to make sure
  // it is either an OP's input or an OP's output.

  auto subgraph_nodes = func->subgraph;
  for (int index = 0; index < block->ops_size(); index++) {
    framework::proto::OpDesc *op = block->mutable_ops(index);
    auto correspond_node = subgraph_nodes[index];
    PADDLE_ENFORCE_EQ(correspond_node->name(), op->type());

    std::unordered_map<std::string, size_t> var2id;
    for (auto *in_var : correspond_node->inlinks) {
      var2id[in_var->name()] = in_var->id();
    }
    // rename for the input variables of op inside subgraph
    for (int i = 0; i < op->inputs_size(); i++) {
      framework::proto::OpDesc_Var *in_var = op->mutable_inputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < in_var->arguments_size(); k++) {
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
    for (auto out_var : correspond_node->outlinks) {
      var2id[out_var->name()] = out_var->id();
    }

    // rename for the output variables of op inside subgraph
    for (int i = 0; i < op->outputs_size(); i++) {
      framework::proto::OpDesc_Var *out_var = op->mutable_outputs(i);
      std::vector<std::string> replaced_names;
      for (int k = 0; k < out_var->arguments_size(); k++) {
        std::string arg_value = out_var->arguments(k);
        std::string arg_value_with_id =
            arg_value + std::to_string(var2id[arg_value]);
        if (output_names_with_id.count(arg_value_with_id)) {
          output_name_map[arg_value] = arg_value_with_id;
        }
        replaced_names.push_back(arg_value_with_id);
      }
      out_var->clear_arguments();
      for (size_t k = 0; k < replaced_names.size(); k++) {
        out_var->add_arguments(replaced_names[k]);
      }
    }
  }
  // When tensorrt engine runs at the end of the operation,
  // output_mapping help us copy the data from the renamed ITensor
  // to Tensor.
  std::vector<std::string> output_mapping;
  for (auto name : output_names) {
    PADDLE_ENFORCE(output_name_map.count(name) != 0);
    output_mapping.push_back(output_name_map[name]);
  }

  PADDLE_ENFORCE(!block->vars().empty(), "the block has no var-desc");
  // Set attrs

  SetAttr(desc.Proto(), "subgraph", block->SerializeAsString());
  SetAttr(desc.Proto(), "max_batch_size", argument->Get<int>("max_batch_size"));
  SetAttr(desc.Proto(), "workspace_size", argument->Get<int>("workspace_size"));
  SetAttr(desc.Proto(), "engine_uniq_key", "trt-" + std::to_string(counter++));
  SetAttr(desc.Proto(), "parameters", ExtractParameters(graph.nodes.nodes()));
  SetAttr(desc.Proto(), "output_name_mapping", output_mapping);
  node->SetPbMsg(desc.Proto()->SerializeAsString());
}

std::vector<std::string> ExtractParameters(
    const std::vector<std::unique_ptr<Node>> &nodes) {
  std::vector<std::string> parameters;
  for (const auto &node : nodes) {
    if (!node->IsValue()) continue;
    PADDLE_ENFORCE(!node->pb_msg().empty(), "pb_msg should be set first");
    framework::proto::VarDesc var;
    var.ParseFromString(node->pb_msg());
    if (var.persistable()) {
      parameters.push_back(var.name());
    }
  }
  return parameters;
}

void DataFlowGraphToFluidPass::AddEngineOp(Node *node) {
  // TODO(Superjomn) Here need to expose some arguments for default setting.
  PADDLE_ENFORCE(node->IsFunctionBlock());
  auto *block_node = static_cast<FunctionBlock *>(node);
  framework::proto::BlockDesc proto;
  framework::BlockDesc block_desc(nullptr, &proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  VLOG(4) << "origin variable size: "
          << argument_->origin_program_desc->blocks(0).vars().size();
  VLOG(4) << "transformed variable size: " << block_desc.Proto()->vars().size();
  // copy ops.

  for (auto *node : block_node->subgraph) {
    auto *op = block_desc.AppendOp();
    PADDLE_ENFORCE(!node->pb_msg().empty());
    op->Proto()->ParseFromString(node->pb_msg());
  }

  *block_desc.Proto()->mutable_vars() =
      argument_->origin_program_desc->blocks(0).vars();
  PADDLE_ENFORCE(!block_desc.Proto()->vars().empty());
  CreateTrtEngineOp(node, argument_, block_desc.Proto());
  auto *main_block = desc_->mutable_blocks(framework::kRootBlockIndex);
  auto *op = main_block->add_ops();
  PADDLE_ENFORCE(!node->pb_msg().empty(), "failed to set desc for block");
  op->ParseFromString(node->pb_msg());
}

namespace {
class DFG_DebuggerPass : public DFG_GraphvizDrawPass {
 public:
  using Config = DFG_GraphvizDrawPass::Config;
  explicit DFG_DebuggerPass(const Config &config)
      : DFG_GraphvizDrawPass(config) {}

  std::string repr() const override { return "dfg-to-fluid-debuger-pass"; }

  bool Finalize() override { return true; }
};
}  // namespace

AnalysisPass *DataFlowGraphToFluidPass::CreateGraphvizDebugerPass() const {
  return new DFG_DebuggerPass(DFG_GraphvizDrawPass::Config(
      FLAGS_IA_graphviz_log_root,
      "data_flow_graph_to_fluid_graphviz_debugger"));
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle
