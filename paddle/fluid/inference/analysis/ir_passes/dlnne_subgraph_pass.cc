// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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
#include <fstream>
#include <iostream>
#include <map>
#include <set>

#include "paddle/fluid/framework/attribute.h"
#include "paddle/fluid/framework/ir/graph_pattern_detector.h"
#include "paddle/fluid/framework/ir/subgraph_detector.h"
#include "paddle/fluid/framework/op_version_registry.h"
#include "paddle/fluid/framework/type_defs.h"
#include "paddle/fluid/inference/analysis/helper.h"
#include "paddle/fluid/inference/analysis/ir_passes/dlnne_subgraph_pass.h"
#include "paddle/fluid/string/pretty_log.h"

namespace paddle {
namespace inference {
namespace analysis {

using framework::ir::Node;

void analysis::DlnneSubgraphPass::InferShapeForDlnneMainGraph() const {
  // copy from paddle2onnx
  static std::unordered_set<std::string> OP_WITHOUT_KERNEL_SET = {
      "feed",
      "fetch",
      "recurrent",
      "go",
      "rnn_memory_helper_grad",
      "conditional_block",
      "while",
      "send",
      "recv",
      "listen_and_serv",
      "fl_listen_and_serv",
      "ncclInit",
      "select",
      "checkpoint_notify",
      "gen_bkcl_id",
      "c_gen_bkcl_id",
      "gen_nccl_id",
      "c_gen_nccl_id",
      "c_comm_init",
      "c_sync_calc_stream",
      "c_sync_comm_stream",
      "queue_generator",
      "dequeue",
      "enqueue",
      "heter_listen_and_serv",
      "c_wait_comm",
      "c_wait_compute"};

  std::string bilinear_interp_v2_type = "bilinear_interp_v2";
  auto input_dict =
      Get<std::map<std::string, std::vector<int64_t>>>("input_shape_dict");

  framework::ProgramDesc *global_program =
      Get<framework::ProgramDesc *>("program");
  auto block = global_program->MutableBlock(framework::kRootBlockIndex);
  for (auto kv : input_dict) {
    auto var = block->FindVar(kv.first);
    if (var != nullptr) {
      var->SetShape(kv.second);
    } else {
      VLOG(4) << "input_name:" << kv.first << " not find in all input vars";
    }
  }

  std::vector<framework::OpDesc *> all_ops = block->AllOps();

  for (size_t i = 0; i < block->OpSize(); i++) {
    // the output_shape of bilinear_interp_v2 cannot be inferd by input shape,
    // it also need the value of input tensor, so when call OpDesc->InferShape,
    // the output_shape of bilinear_interp_v2 is still dynamic, here we try to
    // infer the output_shape of bilinear_interp_v2 infer shape for
    // bilinear_interp_v2
    if (block->Op(i)->Type() == bilinear_interp_v2_type) {
      framework::VariableNameMap input_name_map = block->Op(i)->Inputs();
      std::vector<std::string> input_name_vec = input_name_map["OutSize"];
      PADDLE_ENFORCE_EQ(
          input_name_vec.size(),
          1,
          platform::errors::PreconditionNotMet(
              "The 'bilinear_interp_v2 op' input 'OutSize' size must be 1 "));

      // find shape->slice->bilinear_interp_v2 pattern
      int start_id = 0;
      int end_id = 0;
      std::vector<std::string> slice_input_name_vec;
      for (auto *i_op : all_ops) {
        if (i_op->HasOutput("Out")) {
          auto it = find(i_op->Output("Out").begin(),
                         i_op->Output("Out").end(),
                         input_name_vec[0]);
          if (it != i_op->Output("Out").end()) {
            slice_input_name_vec = i_op->Input("Input");
            PADDLE_ENFORCE_EQ(
                slice_input_name_vec.size(),
                1,
                platform::errors::PreconditionNotMet(
                    "The 'slice op' input 'Input' size must be 1 "));

            auto start_vec = i_op->GetAttrIfExists<std::vector<int>>("starts");
            start_id = start_vec[0];
            auto end_vec = i_op->GetAttrIfExists<std::vector<int>>("ends");
            end_id = end_vec[0];
            break;
          }
        }
      }

      std::vector<std::string> shape_input_name_vec;
      for (auto *i_op : all_ops) {
        if (i_op->HasOutput("Out")) {
          auto it = find(i_op->Output("Out").begin(),
                         i_op->Output("Out").end(),
                         slice_input_name_vec[0]);
          if (it != i_op->Output("Out").end()) {
            shape_input_name_vec = i_op->Input("Input");
            PADDLE_ENFORCE_EQ(
                slice_input_name_vec.size(),
                1,
                platform::errors::PreconditionNotMet(
                    "The 'shape op' input 'Input' size must be 1 "));
            break;
          }
        }
      }
      auto target_var = block->FindVarRecursive(shape_input_name_vec[0]);
      std::vector<int64_t> target_shape = target_var->GetShape();
      size_t target_shape_len = target_shape.size();
      if (start_id < 0) {
        start_id = target_shape_len + start_id;
      } else if (start_id > static_cast<int>(target_shape_len)) {
        start_id = target_shape_len;
      }

      if (end_id < 0) {
        end_id = target_shape_len + end_id;
      } else if (end_id > static_cast<int>(target_shape_len)) {
        end_id = target_shape_len;
      }

      if (start_id < end_id) {
        std::vector<int64_t> OutSize_dims(target_shape.begin() + start_id,
                                          target_shape.begin() + end_id);

        framework::VariableNameMap output_name_map = block->Op(i)->Outputs();
        std::vector<std::string> output_name_vec = output_name_map["Out"];
        auto out_var = block->FindVarRecursive(output_name_vec[0]);
        PADDLE_ENFORCE_NOT_NULL(
            out_var,
            platform::errors::NotFound(
                "bilinear_interp_v2 op's output %s is not found in the block.",
                output_name_vec[0]));
        std::vector<int64_t> ori_shape = out_var->GetShape();
        std::string data_layout =
            block->Op(i)->GetAttrIfExists<std::string>("data_layout");
        size_t start_dim = 0;
        size_t end_dim = 0;

        if (data_layout == "NCHW") {
          start_dim = 2;
          end_dim = ori_shape.size();
        } else {
          start_dim = 1;
          end_dim = ori_shape.size() - 1;
        }
        for (size_t i_dim = start_dim; i_dim < end_dim; i_dim++) {
          ori_shape[i_dim] = OutSize_dims[i_dim - start_dim];
        }

        VLOG(4) << "Set bilinear_interp_v2 shape: " << ori_shape[2] << ", "
                << ori_shape[3];
        out_var->SetShape(ori_shape);
      }

    } else {
      if (OP_WITHOUT_KERNEL_SET.find(block->Op(i)->Type()) ==
          OP_WITHOUT_KERNEL_SET.end())
        block->Op(i)->InferShape(*block);
    }
  }
}

bool analysis::DlnneSubgraphPass::IsDynamicOp(std::string var_name,
                                              bool use_static_batch) const {
  framework::ProgramDesc *global_program =
      Get<framework::ProgramDesc *>("program");
  auto block = global_program->MutableBlock(framework::kRootBlockIndex);
  auto var = block->FindVar(var_name);

  if (var != nullptr) {
    std::vector<int64_t> var_shape = var->GetShape();
    size_t start_idx = use_static_batch ? 1 : 0;
    for (; start_idx < var_shape.size(); start_idx++) {
      if (var_shape[start_idx] < 1) {
        return false;
      }
    }
  }
  return true;
}

void analysis::DlnneSubgraphPass::ApplyImpl(framework::ir::Graph *graph) const {
  framework::ir::FusePassBase::Init("dlnne_subgraph_pass", graph);

  InferShapeForDlnneMainGraph();

  static std::unordered_set<std::string> teller_set{
      "nearest_interp_v2",
      "mul",
      "matmul",
      "matmul_v2",
      "flatten_contiguous_range",
      "conv2d",
      "pool2d",
      "relu",
      "softmax",
      "sigmoid",
      "softplus",
      "hard_swish",
      "hard_sigmoid",
      "depthwise_conv2d",
      "batch_norm",
      "exp",
      "concat",
      "clip",
      "cast",
      "tanh",
      "pad",
      "elementwise_add",
      "elementwise_mul",
      "elementwise_sub",
      "elementwise_div",
      "elementwise_pow",
      "dropout",
      // "deformable_conv",

      "prelu",
      "conv2d_transpose",
      "leaky_relu",
      "log",
      "fc",
      "shuffle_channel",
      "swish",
      "split",
      "instance_norm",
      "gelu",
      "layer_norm",
      "scale",
      "slice",
      "stack",
      "relu6",
      "reshape2",
      "transpose2",
      "concat",
      "slice",
      "fill_constant",
      "fill_constant_batch_size_like",
      "shape",
      "unsqueeze2",
      "pad3d",
      "squeeze2",
      "bilinear_interp_v2"
      // "yolo_box"
  };

  // the op which output is special, need special process
  static std::unordered_set<std::string> special_output_op_set{
      "transpose2",
      "fill_constant_batch_size_like",
      "flatten_contiguous_range",
      "batch_norm",
      "unsqueeze2",
  };

  // the op when it's shape is dynamic still can be fused by
  // dlnne_engine_op
  static std::unordered_set<std::string> dynamic_pass_op_set{
      "reshape2",
  };
  auto disable_nodes_by_outputs =
      Get<std::unordered_set<std::string>>("disable_nodes_by_outputs");
  bool use_static_batch = Get<bool>("use_static_batch");

  auto teller = [&](const framework::ir::Node *node) {
    if (!node->IsOp() || !node->Op()) {
      return false;
    }
    if (teller_set.find(node->Op()->Type()) == teller_set.end()) {
      VLOG(4) << "don't support op:" << node->Op()->Type();
      return false;
    } else {
      bool flag = true;
      // check node output
      if (dynamic_pass_op_set.find(node->Op()->Type()) !=
          dynamic_pass_op_set.end()) {
        flag = true;
      } else if (special_output_op_set.find(node->Op()->Type()) ==
                 special_output_op_set.end()) {
        for (auto *x : node->outputs) {
          std::string var_name = x->Var()->Name();
          flag = IsDynamicOp(var_name, use_static_batch);
          if (!flag) break;
        }
      } else {
        std::string var_name = node->outputs[0]->Var()->Name();
        flag = IsDynamicOp(var_name, use_static_batch);
      }
      // check node input
      if (flag) {
        for (auto *x : node->inputs) {
          std::string var_name = x->Var()->Name();
          flag = IsDynamicOp(var_name, use_static_batch);
          if (!flag) break;
        }
      }
      if (!flag) {
        VLOG(4) << "don't support dynamic shape:" << node->Op()->Type();
      }
      bool flag2 = true;
      for (auto *x : node->outputs) {
        if (disable_nodes_by_outputs.find(x->Name()) !=
            disable_nodes_by_outputs.end()) {
          flag2 = false;
        }
      }
      if (!flag2) {
        VLOG(4) << "user don't use " << node->Name() << "...";
      }
      return flag && flag2;
    }
  };

  framework::ir::SubGraphFuser fuser(
      graph,
      teller,
      Get<int>("min_subgraph_size") /*min subgraph size*/,
      "dlnne_engine");
  fuser();

  std::vector<std::string> graph_param_names =
      ExtractParameters(graph->Nodes());
  // those parameter already exist in dlnne, and should not have another copy in
  // fluid.
  std::vector<std::string> repetitive_params;

  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && !framework::ir::Agent(node).subgraph()->empty()) {
      CreateDlnneOp(node, graph, graph_param_names, &repetitive_params);

      std::unordered_set<const Node *> nodes2remove(
          framework::ir::Agent(node).subgraph()->begin(),
          framework::ir::Agent(node).subgraph()->end());
      framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
    }
  }

  std::unordered_set<const Node *> nodes2remove;
  for (auto *node : graph->Nodes()) {
    if (node->IsOp() && framework::ir::Agent(node).deleted()) {
      nodes2remove.insert(node);
    }
  }
  framework::ir::GraphSafeRemoveNodes(graph, nodes2remove);
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
std::string replace_name(std::string name,
                         const char *raw,
                         const char *new_char) {
  std::string r_name = name;
  int pos = r_name.find(raw);
  while (pos >= 0) {
    r_name = r_name.replace(pos, 1, new_char);
    pos = r_name.find(raw);
  }
  return r_name;
}

auto fix_batch_as_one(
    std::unordered_map<std::string, framework::VarDesc *> *name_var_desc,
    std::set<std::string> *valid_input_names,
    bool use_static_batch = false) {
  std::unordered_map<std::string, std::vector<int64_t>> name_var_shape;

  if (use_static_batch) {
    std::set<std::string> names;
    names.insert(valid_input_names->begin(), valid_input_names->end());

    for (auto name : names) {
      if (name_var_desc->find(name) != name_var_desc->end()) {
        auto var_desc = (*name_var_desc)[name];
        auto sp = var_desc->GetShape();
        if (sp[0] == -1) {
          sp[0] = 1;
          name_var_shape[name] = sp;
          std::stringstream sp_str;
          copy(sp.begin(),
               sp.end(),
               std::ostream_iterator<int64_t>(sp_str, ","));

          LOG(INFO)
              << "Warning: fix var:" << name << " batch,shape is ["
              << sp_str.str()
              << "],we assume subgraph's inputs/outputs first dim is batch,"
              << "but when the first dim is not mean batch "
              << "we suggest you use fix shape model....";
        }
      }
    }
  }
  return name_var_shape;
}
/*
there are two ProgramDesc in the function, global_program is used for generate a
Dlnne op, dump_program is used for dump the subgraph to onnx subgraph which is
loaded by Dlnne op
*/
void DlnneSubgraphPass::CreateDlnneOp(
    framework::ir::Node *node,
    framework::ir::Graph *graph,
    const std::vector<std::string> &graph_params,
    std::vector<std::string> *repetitive_params) const {
  auto *op_desc = node->Op();
  auto &subgraph = *framework::ir::Agent(node).subgraph();
  PADDLE_ENFORCE_EQ(subgraph.empty(),
                    false,
                    platform::errors::PreconditionNotMet(
                        "The subgraph should not be empty."));

  // A fake block desc.
  framework::proto::BlockDesc block_proto;
  framework::BlockDesc block_desc(nullptr, &block_proto);
  block_desc.Proto()->set_parent_idx(-1);
  block_desc.Proto()->set_idx(0);
  LOG(INFO) << "---  detect a sub-graph with " << subgraph.size() << " nodes";
  // for debug
  framework::ProgramDesc *global_program =
      Get<framework::ProgramDesc *>("program");
  const framework::BlockDesc &main_block =
      global_program->Block(framework::kRootBlockIndex);

  std::set<std::string> input_names;
  std::set<std::string> input_names_with_id;
  std::vector<std::string> params;
  std::set<std::string> valid_input_names;
  // if we delete fluid copy of params shared by more than 1 ops, there will be
  // problem, so we filter them out.

  // The node->inputs contains input tensors and parameters.
  for (auto *x : node->inputs) {
    input_names.insert(x->Name());
    input_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::count(graph_params.begin(), graph_params.end(), x->Name()) > 0) {
      params.push_back(x->Name());
    }
    if (std::find(graph_params.begin(), graph_params.end(), x->Name()) ==
        graph_params.end()) {
      valid_input_names.insert(x->Name());
    }
  }

  std::set<std::string> output_names;
  std::set<std::string> output_names_with_id;
  std::vector<int> origin_output_dims;
  std::set<std::string> valid_output_names;
  for (auto *x : node->outputs) {
    origin_output_dims.push_back(x->Var()->GetShape().size());
    output_names.insert(x->Name());
    output_names_with_id.insert(x->Name() + std::to_string(x->id()));
    if (std::find(graph_params.begin(), graph_params.end(), x->Name()) ==
        graph_params.end()) {
      valid_output_names.insert(x->Name());
    }
  }

  auto *child_block = global_program->AppendBlock(main_block);
  framework::ProgramDesc dump_program;
  auto *export_block = dump_program.MutableBlock(framework::kRootBlockIndex);

  std::unordered_map<std::string, framework::VarDesc *> name_var_desc;
  for (auto *node : subgraph) {
    auto *op = block_desc.AppendOp();
    *op->Proto() = *node->Op()->Proto();
    auto *child_op = child_block->AppendOp();
    *child_op->Proto() = *node->Op()->Proto();
    // generate op by node to append on block
    {
      auto *export_op = export_block->AppendOp();

      framework::OpDesc op_desc;
      op_desc.CopyFrom(*node->Op());

      for (auto argument_name : op_desc.InputArgumentNames()) {
        if (std::count(
                graph_params.begin(), graph_params.end(), argument_name) > 0) {
          op_desc.Rename(argument_name, replace_name(argument_name, "/", "."));
        }
      }
      for (auto argument_name : op_desc.OutputArgumentNames()) {
        if (std::count(
                graph_params.begin(), graph_params.end(), argument_name) > 0) {
          op_desc.Rename(argument_name, replace_name(argument_name, "/", "."));
        }
      }
      *export_op->Proto() = *op_desc.Proto();

      for (auto *x : node->inputs) {
        if (x->IsVar()) {
          auto var_desc_infer = main_block.FindVarRecursive(x->Name());
          if (var_desc_infer != nullptr) {
            name_var_desc[x->Name()] = var_desc_infer;
          } else {
            name_var_desc[x->Name()] = x->Var();
          }
        }
      }

      for (auto *x : node->outputs) {
        if (x->IsVar()) {
          auto var_desc_infer = main_block.FindVarRecursive(x->Name());
          if (var_desc_infer != nullptr) {
            name_var_desc[x->Name()] = var_desc_infer;
          } else {
            name_var_desc[x->Name()] = x->Var();
          }
        }
      }
    }
  }

  // starting fix bath as one
  bool use_static_batch = Get<bool>("use_static_batch");
  auto name_shape_table =
      fix_batch_as_one(*name_var_desc, *valid_input_names, use_static_batch);

  for (const auto &name_shape : name_shape_table) {
    VLOG(4) << "Fix batch shape as one var name: " << name_shape.first;
  }

  // Then, we will use the input_names_with_id and output_names_with_id to
  // generate the engine key.
  // So, We use set instead of unordered_set here to ensure that the engine key
  // is unique.
  auto engine_key = GenerateEngineKey(
      input_names_with_id, output_names_with_id, std::to_string(0));
  auto precision_mode = Get<AnalysisConfig::Precision>("precision_mode");
  bool enable_int8 = false;
  if (precision_mode == AnalysisConfig::Precision::kInt8) {
    enable_int8 = true;
  }
  auto use_calib_mode = Get<bool>("use_calib_mode");

  std::string calibration_data_path = "./calibration/dlnne_calib_" + engine_key;
  bool calibration_mode = false;
  if (enable_int8 && use_calib_mode && !PathExists(calibration_data_path)) {
    calibration_mode = true;
    MKDIR("./calibration");
    MKDIR(calibration_data_path.c_str());
  }
  VLOG(4) << "calibration_mode: " << calibration_mode;
  std::stringstream ss;
  ss << "engine_key:" << engine_key << " outputs:[";
  for (auto name : valid_output_names) {
    ss << name << ",";
  }
  ss << "]";
  VLOG(4) << ss.str();

  // Set attrs
  op_desc->SetType("dlnne_engine");
  op_desc->SetInput("Xs",
                    std::vector<std::string>(valid_input_names.begin(),
                                             valid_input_names.end()));

  op_desc->SetOutput("Ys",
                     std::vector<std::string>(valid_output_names.begin(),
                                              valid_output_names.end()));
  op_desc->SetBlockAttr("sub_block", child_block);

  op_desc->SetAttr("parameters", params);
  op_desc->SetAttr("engine_key", engine_key);
  op_desc->SetAttr("max_batch_size", Get<int>("max_batch_size"));
  op_desc->SetAttr("use_static_batch", Get<bool>("use_static_batch"));
  op_desc->SetAttr("weight_share_mode", Get<std::string>("weight_share_mode"));
  op_desc->SetAttr("enable_int8", enable_int8);
  op_desc->SetAttr("use_calib_mode", use_calib_mode);
  op_desc->SetAttr("calibration_mode", calibration_mode);
  op_desc->SetAttr("calibration_data_path", calibration_data_path);

  std::string subgraph_root_path = "./dump/" + engine_key;
  op_desc->SetAttr("subgraph_root_path", subgraph_root_path);

  std::stringstream ins_stream;
  for (auto name : valid_input_names) {
    ins_stream << "," << name;
  }
  op_desc->SetAttr("valid_input_names", ins_stream.str().substr(1));

  std::stringstream outs_stream;
  for (auto name : valid_output_names) {
    outs_stream << "," << name;
  }
  op_desc->SetAttr("valid_output_names", outs_stream.str().substr(1));

  auto *scope = param_scope();
  {
    // add feed to subgraph:
    int input_idx = 0;
    for (auto input_name : valid_input_names) {
      auto *feed1 = export_block->AppendOp();
      feed1->SetType("feed");
      feed1->SetInput("X", {"feed"});
      feed1->SetOutput("Out", {input_name});
      feed1->SetAttr("col", input_idx);
      input_idx++;
    }
    // add fetch to subgraph:
    int output_idx = 0;
    for (auto output_name : valid_output_names) {
      auto *fetch1 = export_block->AppendOp();
      fetch1->SetType("fetch");
      fetch1->SetInput("X", {output_name});
      fetch1->SetOutput("Out", {"out"});
      fetch1->SetAttr("col", output_idx);
      output_idx++;
    }

    VLOG(4) << "name_var_desc size:" << name_var_desc.size();

    for (auto &kv : name_var_desc) {
      auto *new_add_var1 = export_block->Proto()->add_vars();
      paddle::framework::VarDesc copy_var_desc(*(kv.second->Proto()));

      if (name_shape_table.find(kv.first) != name_shape_table.end()) {
        copy_var_desc.SetShape(name_shape_table[kv.first]);
      }
      *new_add_var1 = *(copy_var_desc.Proto());

      auto *variable_tmp1 = scope->FindVar(kv.first);
      if (variable_tmp1 != nullptr) {
        *new_add_var1->mutable_name() = replace_name(kv.first, "/", ".");
        new_add_var1->set_persistable(true);
      } else {
        new_add_var1->set_persistable(false);
      }
    }

    std::string model_str;
    dump_program.Proto()->SerializeToString(&model_str);
    op_desc->SetAttr("subgraph", model_str);
    op_desc->Flush();

    if (calibration_mode) {
      return;
    }

    MKDIR("./dump");
    MKDIR(subgraph_root_path.c_str());
    std::ofstream m_stream;
    m_stream.open(subgraph_root_path + "/__model__", std::ios::out);

    for (auto param_name : params) {
      auto *var = scope->FindVar(param_name);
      if (var != nullptr) {
        auto *var_t = var->GetMutable<framework::LoDTensor>();
        std::ofstream p_stream;
        p_stream.open(
            subgraph_root_path + "/" + replace_name(param_name, "/", "."),
            std::ios::out);
        platform::DeviceContextPool &pool =
            platform::DeviceContextPool::Instance();
        auto &dev_ctx = *pool.Get(var_t->place());
        framework::SerializeToStream(p_stream, *var_t, dev_ctx);
        p_stream.close();
      }
    }

    m_stream << model_str;
    m_stream.close();
  }
}

}  // namespace analysis
}  // namespace inference
}  // namespace paddle

REGISTER_PASS(dlnne_subgraph_pass,
              paddle::inference::analysis::DlnneSubgraphPass);
