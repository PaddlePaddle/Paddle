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

#pragma once

#include "paddle/fluid/eager/api/utils/global_utils.h"
#include "paddle/fluid/eager/grad_node_info.h"
#include "paddle/fluid/eager/tensor_wrapper.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/run_program_op.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/profiler/event_tracing.h"

#ifdef PADDLE_WITH_GCU
#include "paddle/fluid/framework/ir/gcu/constant_compute_utils.h"
#include "paddle/fluid/framework/ir/pass_tester_helper.h"
#include "paddle/fluid/platform/device/gcu/compiler/single_op_compiler.h"
#include "paddle/fluid/platform/device/gcu/executor/single_op_executor.h"
#include "paddle/fluid/platform/device/gcu/layout/gcu_layout_interface.h"
#include "paddle/fluid/platform/device/gcu/register/register.h"
#include "paddle/fluid/platform/device/gcu/runtime/gcu_rt_interface.h"
#include "paddle/fluid/platform/device/gcu/utils/layout.h"
#include "paddle/fluid/platform/device/gcu/utils/types.h"

PHI_DECLARE_bool(check_nan_inf);

namespace paddle {
namespace gcu {
const char *const use_gcu_cache_executor = "USE_GCU_CACHE_EXECUTOR";
const char *const kRunningModeSerial = "serial";

using LoDTensor = phi::DenseTensor;
using paddle::framework::Scope;
using paddle::framework::details::VariableInfo;
using paddle::framework::ir::Graph;
using paddle::platform::gcu::EquivalenceTransformer;
using paddle::platform::gcu::GcuBuilder;
using paddle::platform::gcu::GcuBuilderPtr;
using paddle::platform::gcu::GcuOp;
using paddle::platform::gcu::GcuOpPtr;
using paddle::platform::gcu::INSENSITIVE;
using paddle::platform::gcu::kAttrOpOutVarName;
using paddle::platform::gcu::kUnusedArchetype;
using paddle::platform::gcu::SingleOpGcuCompiler;
using paddle::platform::gcu::SingleOpGcuExecutor;
using paddle::platform::gcu::SingleOpGcuExecutorManager;
using paddle::platform::gcu::TransformUtil;
using Node = paddle::framework::ir::Node;
using GcuRunTimeInfo = paddle::platform::gcu::runtime::GcuRunTimeInfo;
using ScopePtr = std::shared_ptr<paddle::framework::Scope>;

static std::map<size_t, std::set<std::string>> all_tmp_node_names_;
static std::map<size_t, std::set<std::string>> forward_skip_eager_vars_;

static void SaveLodTensor(const platform::Place &place,
                          const framework::Variable *var,
                          const std::string &file_name) {
  auto &tensor = var->Get<LoDTensor>();
  if (!tensor.IsInitialized()) {
    VLOG(0) << "dump tensor to file failed by tensor is not initialized; file "
               "name is :"
            << file_name;
    return;
  }

  // get device context from pool
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  std::ofstream fout(file_name, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    platform::errors::Unavailable(
                        "Cannot open %s to save variables.", file_name));

  // auto in_dtype = framework::TransToProtoVarType(tensor.dtype());
  framework::SerializeToStream(fout, tensor, dev_ctx);

  fout.close();
}

static void SaveSelectedRows(const platform::Place &place,
                             const framework::Variable *var,
                             const std::string &file_name) {
  auto &selectedRows = var->Get<phi::SelectedRows>();
  // get device context from pool
  platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
  auto &dev_ctx = *pool.Get(place);

  std::ofstream fout(file_name, std::ios::binary);
  PADDLE_ENFORCE_EQ(static_cast<bool>(fout),
                    true,
                    platform::errors::Unavailable(
                        "Cannot open %s to save variables.", file_name));
  framework::SerializeToStream(fout, selectedRows, dev_ctx);
  fout.close();
}

static void VarToFile(const platform::Place &place,
                      const std::string file_name,
                      framework::Variable *var) {
  if (var->IsType<LoDTensor>()) {
    SaveLodTensor(place, var, file_name);
  } else if (var->IsType<phi::SelectedRows>()) {
    SaveSelectedRows(place, var, file_name);
  } else {
    PADDLE_THROW(platform::errors::InvalidArgument(
        "Save operator only supports saving LoDTensor and SelectedRows "
        "variable, %s has wrong type",
        file_name));
  }
}

static void VarsToFile(const platform::Place &place,
                       const std::string &dir,
                       const std::vector<std::string> &var_names,
                       std::vector<framework::Variable *> vars) {
  for (size_t i = 0; i < var_names.size(); i++) {
    if (var_names[i] == framework::kEmptyVarName ||
        var_names[i] == "Fake_var") {
      continue;
    }
    // TensorToFile(dir + var_names[i], vars[i]->GetMutable<LoDTensor>());
    VarToFile(place, dir + var_names[i], vars[i]);
  }
}

static void VarsToFile(
    const platform::Place &place,
    const std::string &dir,
    const std::map<std::string, framework::VarDesc *> &graph_var_nodes,
    const framework::Scope &scope) {
  std::vector<framework::Variable *> vars;
  std::vector<std::string> tensor_names;

  for (auto iter = graph_var_nodes.begin(); iter != graph_var_nodes.end();
       iter++) {
    auto &var_name = iter->first;
    if (var_name == framework::kEmptyVarName || var_name == "Fake_var") {
      continue;
    }

    auto numel = [](std::vector<int64_t> &shape) -> int64_t {
      int64_t res = 1;
      for (auto dim : shape) {
        res *= dim;
      }
      return res;
    };

    auto *var = scope.FindVar(var_name);

    if ((iter->second != nullptr) &&
        ((iter->second->GetType() == framework::proto::VarType::LOD_TENSOR) ||
         (iter->second->GetType() ==
          framework::proto::VarType::SELECTED_ROWS))) {
      std::vector<int64_t> shape = iter->second->GetShape();
      if (numel(shape) != 0) {
        vars.emplace_back(var);
        tensor_names.emplace_back(var_name);
      }
    } else {
      VLOG(0) << "dump var to file failed by var is nullptr; var name is : "
              << var_name;
    }
  }

  for (size_t i = 0; i < tensor_names.size(); i++) {
    VarToFile(place, dir + tensor_names[i], vars[i]);
  }
}

static void GraphToFile(std::string dir, bool forward, Graph *graph) {
  auto nodes = framework::ir::TopologySortOperations(*graph);
  std::map<std::string, framework::VarDesc *> graph_var_nodes;
  std::ostringstream os;
  os << "Output Var: {\n";
  for (Node *node : nodes) {
    for (auto output : node->outputs) {
      os << "    " << output->Name() << "\n";
    }
  }
  os << "}\n";

  std::string var_str = os.str();
  std::string graph_str = var_str + framework::ir::DebugString(graph);
  std::string file_name = dir + (forward ? "forward_" : "backward_");

  std::ofstream graph_file((file_name + "graph.txt").c_str(), std::ios::binary);
  graph_file.write(graph_str.c_str(), graph_str.size());
  graph_file.close();

  std::ofstream var_file((file_name + "var.txt").c_str(), std::ios::binary);
  var_file.write(var_str.c_str(), var_str.size());
  var_file.close();
}

static std::string GenerateDumpTensorDir(std::string device_type,
                                         bool forward,
                                         bool before_train,
                                         int64_t program_id,
                                         size_t program_count,
                                         size_t step_count) {
  std::ostringstream os;
  os << "./dump_tensor/" << device_type << "/" << program_count << "_"
     << program_id << "/" << step_count << "/"
     << (forward ? "forward_" : "backward_")
     << (before_train ? "before_train/" : "after_train/");
  return os.str();
}

static std::string GenerateDumpGraphDir(std::string device_type,
                                        int64_t program_id,
                                        size_t program_count) {
  std::ostringstream os;
  os << "./dump_tensor/" << device_type << "/" << program_count << "_"
     << program_id << "/";
  return os.str();
}

static void RefreshDynamicInputVarShape(
    const Scope &scope,
    std::map<std::string, framework::VarDesc *> input_var_nodes) {
  for (auto var_node : input_var_nodes) {
    auto *var = scope.FindVar(var_node.first);

    if ((var != nullptr) &&
        (var_node.second->GetType() == framework::proto::VarType::LOD_TENSOR)) {
      auto tensor_dims = var->GetMutable<LoDTensor>()->dims();
      auto tensor_shape = phi::vectorize(tensor_dims);
      auto var_shape = var_node.second->GetShape();
      if ((platform::gcu::TransformUtil::IsDyn(var_shape) &&
           !platform::gcu::TransformUtil::IsDyn(tensor_shape)) ||
          ((var_shape != tensor_shape) && var_shape.size() <= 0)) {
        VLOG(6) << "input var_name:" << var_node.first << " "
                << "tensor shape:" << TransformUtil::GetShapeStr(tensor_shape)
                << " "
                << "var shape:" << TransformUtil::GetShapeStr(var_shape) << " "
                << "[WARN]use tensor shape to flush var shape!";
        var_node.second->SetShape(tensor_shape);
      }
    }
  }
}

static std::vector<std::string> ParseAttr(std::string attr_value) {
  std::vector<std::string> out_var_names;
  if (attr_value == "") return out_var_names;
  const char *divided_symbol = ";";
  size_t pos = attr_value.find(divided_symbol);
  if (pos == attr_value.npos) {
    out_var_names.emplace_back(attr_value);
  }
  while (pos != attr_value.npos) {
    std::string sub_str = attr_value.substr(0, pos);
    out_var_names.emplace_back(sub_str);
    attr_value = attr_value.substr(pos + 1, attr_value.size());
    pos = attr_value.find(divided_symbol);
  }
  if (attr_value.length() != 0) {
    out_var_names.emplace_back(attr_value);
  }
  return out_var_names;
}

static GcuOpPtr AddGteOp(const Node *node, const GcuOpPtr &input) {
  if (!node->IsVar()) {
    VLOG(3) << "ERROR! op name:" << node->Name()
            << " should be var type when convert to Gcu gte node";
    return nullptr;
  }
  auto attr_out_var_names = input->GetAttribute(kAttrOpOutVarName);
  PADDLE_ENFORCE_NE(attr_out_var_names == builder::Attribute(""),
                    true,
                    platform::errors::NotFound(
                        "lack of attr [%s] for gcu tuple op, please check.",
                        kAttrOpOutVarName));
  auto out_var_names = attr_out_var_names.GetValueAsString();
  auto list_out_var_names = ParseAttr(out_var_names);
  auto var_name = node->Name();
  int32_t idx = 0;
  VLOG(3) << out_var_names;
  for (const auto &name : list_out_var_names) {
    if (name != var_name) {
      idx++;
    } else {
      break;
    }
  }
  auto shape = node->Var()->GetShape();
  auto dtype =
      paddle::framework::TransToPhiDataType(node->Var()->GetDataType());
  auto ptype = TransformUtil::ConvertDataType(dtype);
  builder::Type input_type(shape, ptype);
  return std::make_shared<GcuOp>(builder::GetTupleElement(*input, idx));
}

class DoInferByHlirBuilder {
 private:
  GcuBuilderPtr builder_ = nullptr;
  std::map<std::string, GcuOpPtr> gcu_node_cache_;
  std::map<std::string, Node *> var_node_cache_;

 public:
  DoInferByHlirBuilder() {
    builder_ = std::make_shared<GcuBuilder>();
    builder_->SetShapeInference(true);
  }
  void Run(Scope &scope, Node *node) {  // NOLINT
    auto op_desc = node->Op();
    auto op_type = op_desc->Type();
    auto op_name = node->Name();
    VLOG(10) << "OpType " << op_type << " strart infer shape by hlir builder. ";
    if (!node->IsOp()) {
      VLOG(10) << "OpType " << op_type << " is not operator.";
      return;
    }
    auto func = EquivalenceTransformer::GetInstance().Get(op_type, INSENSITIVE);
    if (func == nullptr) {
      VLOG(10) << "OpType " << op_type
               << " is not register gcu op convert func, please check.";
      return;
    }

    bool dyn_input = false;
    for (auto input : node->inputs) {
      if (input->IsCtrlVar()) {
        continue;
      }
      auto var_name = input->Name();
      if (gcu_node_cache_.count(var_name) > 0) {
        var_node_cache_[var_name] = input;
        continue;
      }
      auto shape = input->Var()->GetShape();
      if (TransformUtil::IsDyn(shape)) {
        dyn_input = true;
        break;
      }
      auto dtype =
          paddle::framework::TransToPhiDataType(input->Var()->GetDataType());
      auto ptype = TransformUtil::ConvertDataType(dtype);
      builder::Type input_type(shape, ptype);
      auto gcu_op = std::make_shared<GcuOp>(builder_->CreateInput(input_type));
      gcu_node_cache_[var_name] = gcu_op;
      var_node_cache_[var_name] = input;
    }
    if (dyn_input) {
      VLOG(6)
          << "OpType " << op_type
          << " has input with dynamic shape and is about to exit the process";
      return;
    }

    for (auto output : node->outputs) {
      if (output->IsCtrlVar()) {
        continue;
      }
      auto var_name = output->Name();
      var_node_cache_[var_name] = output;
    }

    std::map<std::string, std::vector<GcuOpPtr>> input_ops;
    for (const auto &e : op_desc->Inputs()) {
      if (e.second.empty()) {
        continue;
      }
      std::vector<GcuOpPtr> v;
      for (std::string n : e.second) {
        auto gcu_op = gcu_node_cache_[n];
        if (gcu_op == nullptr) {
          VLOG(2) << "[WARN]Can not find transfered gcu op by"
                     "input name "
                  << n;
        }
        auto gcu_shape_str =
            TransformUtil::GetShapeStr(gcu_op->GetType().GetShape());
        VLOG(3) << "Input Archetype name: " << e.first << " in name:" << n
                << " shape:" << gcu_shape_str;
        v.push_back(gcu_op);
      }
      input_ops[e.first] = v;
    }
    GcuOpPtr op = func(builder_, node, input_ops, kRunningModeSerial);
    VLOG(10) << "Transfered to gcu node end, op name:" << op_name
             << ", type:" << op_type;

    PADDLE_ENFORCE_NE(
        op,
        nullptr,
        platform::errors::NotFound(
            "op type:%s transfered gcu node should not be nullptr!",
            op_name.c_str(),
            op_type.c_str()));
    gcu_node_cache_[op_name] = op;
    bool is_tuple_out = op->GetType().IsTuple();
    // check tuple condition same with pd
    if (is_tuple_out) {
      size_t gcu_output_num = op->GetType().GetTupleSize();
      size_t valid_output_counter = 0;
      for (const auto &e : op_desc->Outputs()) {
        if (!e.second.empty()) {
          VLOG(6) << "Out Archetype name:" << e.first;
          for (const auto &p : e.second) {
            VLOG(6) << "    correspond var name:" << p;
            valid_output_counter++;
          }
        }
      }
      PADDLE_ENFORCE_EQ(
          valid_output_counter,
          gcu_output_num,
          platform::errors::NotFound(
              "op type:%s paddle valid output size is %u, but gcu is %u",
              op_type.c_str(),
              valid_output_counter,
              gcu_output_num));
    }
    if (!is_tuple_out) {
      for (const auto &e : op_desc->Outputs()) {
        if (e.second.empty()) {
          continue;
        }
        std::string weight_name = "";
        for (std::string n : e.second) {
          VLOG(3) << "Output Archetype name: " << e.first << " out name:" << n;
          auto out_name = n;
          gcu_node_cache_[out_name] = op;
          // for shape infer check
          auto gcu_shape = op->GetType().GetShape();
          auto var_op = var_node_cache_[out_name];
          auto paddle_shape = var_op->Var()->GetShape();
          // normalize scalar shape process, [] -> [1]
          if (gcu_shape.empty()) {
            gcu_shape = {1};
          }
          if (paddle_shape.empty()) {
            paddle_shape = {1};
          }
          PADDLE_ENFORCE_EQ(
              gcu_shape.size(),
              paddle_shape.size(),
              platform::errors::NotFound(
                  "op_name:%s, op type:%s transfered gcu node "
                  "should have same rank!"
                  "but origin rank is %u now is %u, paddld "
                  "shape:%s, gcu shape:%s",
                  op_name.c_str(),
                  op_type.c_str(),
                  paddle_shape.size(),
                  gcu_shape.size(),
                  TransformUtil::GetShapeStr(paddle_shape).c_str(),
                  TransformUtil::GetShapeStr(gcu_shape).c_str()));
          auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
          auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);

          if (TransformUtil::IsDyn(paddle_shape) &&
              !TransformUtil::IsDyn(gcu_shape)) {
            auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
            auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);
            VLOG(6) << "out var_name:" << out_name.c_str() << " "
                    << "op_type:" << op_type.c_str() << " "
                    << "shape_pd:" << paddle_shape_str << " "
                    << "shape_gcu:" << gcu_shape_str << " "
                    << "[WARN]use gcu shape to flush paddle shape!";
            var_op->Var()->SetShape(gcu_shape);
            auto *var = scope.FindVar(var_op->Name());
            if (var != nullptr) {
              var->GetMutable<LoDTensor>()->Resize(phi::make_ddim(gcu_shape));
            }
            continue;
          }
          PADDLE_ENFORCE_EQ(gcu_shape_str,
                            paddle_shape_str,
                            platform::errors::NotFound(
                                "op_name:%s, op type:%s"
                                " transfered gcu node should have same shape!"
                                "but origin shape is %s now is %s",
                                op_name.c_str(),
                                op_type.c_str(),
                                paddle_shape_str.c_str(),
                                gcu_shape_str.c_str()));
        }
      }
    } else {
      std::set<std::string> names_in;
      for (const auto &e : op_desc->Inputs()) {
        if (e.second.empty()) {
          continue;
        }
        for (std::string n : e.second) {
          names_in.insert(n);
        }
      }
      for (const auto &e : op_desc->Outputs()) {
        if (e.second.empty()) continue;
        for (const auto &out_name : e.second) {
          auto out_var_op = var_node_cache_[out_name];
          PADDLE_ENFORCE_NE(
              out_var_op,
              nullptr,
              platform::errors::NotFound(
                  "op name:%s op type:%s out name:%s not found var op!",
                  op_name.c_str(),
                  op_type.c_str(),
                  out_name.c_str()));
          VLOG(6) << "  out var name:" << out_name
                  << " var op name:" << out_var_op->Name();
          GcuOpPtr gte = AddGteOp(out_var_op, op);
          PADDLE_ENFORCE_NE(
              gte,
              nullptr,
              platform::errors::NotFound(
                  "op name:%s op type:%s transfer to gcu gte node failed!",
                  op_name.c_str(),
                  op_type.c_str()));
          gcu_node_cache_[out_name] = gte;

          // for shape infer check
          auto gcu_shape = gte->GetType().GetShape();
          auto var_op = var_node_cache_[out_name];
          auto paddle_shape = var_op->Var()->GetShape();
          // normalize scalar shape process, [] -> [1]
          if (gcu_shape.empty()) {
            gcu_shape = {1};
          }
          if (paddle_shape.empty()) {
            paddle_shape = {1};
          }
          if (TransformUtil::IsDyn(paddle_shape) &&
              !TransformUtil::IsDyn(gcu_shape)) {
            auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
            auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);
            VLOG(6) << "out var_name:" << out_name.c_str() << " "
                    << "op_type:" << op_type.c_str() << " "
                    << "shape_pd:" << paddle_shape_str << " "
                    << "shape_gcu:" << gcu_shape_str << " "
                    << "[WARN]use gcu shape to flush paddle shape!";
            var_op->Var()->SetShape(gcu_shape);
            auto *var = scope.FindVar(var_op->Name());
            if (var != nullptr) {
              var->GetMutable<LoDTensor>()->Resize(phi::make_ddim(gcu_shape));
            }
            continue;
          }
          PADDLE_ENFORCE_EQ(
              gcu_shape.size(),
              paddle_shape.size(),
              platform::errors::NotFound(
                  "op_name:%s, op type:%s"
                  " transfered gcu node should have same rank!"
                  "but origin rank is %u now is %u, out name is %s",
                  op_name.c_str(),
                  op_type.c_str(),
                  paddle_shape.size(),
                  gcu_shape.size(),
                  out_name));
          auto gcu_shape_str = TransformUtil::GetShapeStr(gcu_shape);
          auto paddle_shape_str = TransformUtil::GetShapeStr(paddle_shape);
          PADDLE_ENFORCE_EQ(
              gcu_shape_str,
              paddle_shape_str,
              platform::errors::NotFound(
                  "op_name:%s, op type:%s"
                  " transfered gcu node should have same shape!"
                  "but origin shape is %s now is %s, out name is %s",
                  op_name.c_str(),
                  op_type.c_str(),
                  paddle_shape_str.c_str(),
                  gcu_shape_str.c_str(),
                  out_name));
        }
      }
    }
  }
};

static void GcuInferShape(framework::ir::Graph *graph) {
  VLOG(10) << "enter GcuInferShape";
  if (VLOG_IS_ON(10)) {
    std::cout << "Before GcuInferShape Graph: \n" << DebugString(graph);
  }

  ScopePtr scope(new paddle::framework::Scope());
  std::unordered_set<std::string> var_names;
  for (auto node : graph->Nodes()) {
    if (!node->IsVar() || !node->Var() || var_names.count(node->Name())) {
      continue;
    }
    auto var_desc = node->Var();
    auto *ptr = scope->Var(var_desc->Name());
    paddle::framework::InitializeVariable(ptr, var_desc->GetType());

    auto tensor = ptr->GetMutable<LoDTensor>();
    tensor->Resize(phi::make_ddim(var_desc->GetShape()));
    tensor->set_type(
        paddle::framework::TransToPhiDataType(var_desc->GetDataType()));
    if ((node->inputs.empty()) && !(node->outputs.empty())) {
      var_names.emplace(node->Name());
    }
  }

  // infer shape
  std::set<std::string> use_hlir_do_infer_ops = {
      "shape", "slice", "bilinear_interp_v2"};
  std::set<std::string> need_run_ops = {"fill_constant"};
  DoInferByHlirBuilder do_infer_by_hlir_builder;
  auto nodes = framework::ir::TopologySortOperations(*graph);
  for (auto node : nodes) {
    VLOG(10) << "InferShapePass: Infer shape for Op (" << node->Name() << ")";
    auto op_desc = node->Op();

    auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);
    paddle::framework::RuntimeContext rt_ctx(
        op->Inputs(), op->Outputs(), *scope);
    for (auto input : node->inputs) {
      if (input->IsCtrlVar()) {
        continue;
      }
    }
    op->RuntimeInferShape(*scope, paddle::platform::CPUPlace(), rt_ctx);

    for (auto it = rt_ctx.outputs.begin(); it != rt_ctx.outputs.end(); it++) {
      for (size_t i = 0; i < it->second.size(); i++) {
        auto output_name = op_desc->Output(it->first)[i];
        auto dim = it->second[i]->GetMutable<LoDTensor>()->dims();
        auto new_shape = phi::vectorize(dim);
        for (auto output_node : node->outputs) {
          if (output_node->Name() == output_name) {
            output_node->Var()->SetShape(new_shape);
            if (VLOG_IS_ON(10)) {
              std::ostringstream sout;
              sout << "InferShapePass: output[" << output_node->Name()
                   << "], infer shape:[";
              for (auto s : new_shape) {
                sout << std::to_string(s) << ", ";
              }
              sout << "]";
              VLOG(10) << sout.str();
            }
          }
        }
      }
    }
    VLOG(10) << "InferShapePass: Infer shape for Op (" << node->Name()
             << ") finished";
    framework::ir::ConstantComputation::ConstComputeIfNecessary(
        op, scope, node);
    std::set<std::string> refresh_shape_after_compute_ops = {"fill_constant"};
    if (refresh_shape_after_compute_ops.count(node->Name()) > 0) {
      for (auto output : node->outputs) {
        if (output->IsCtrlVar()) {
          continue;
        }
        auto var = scope->GetVar(output->Name());
        auto tensor = var->GetMutable<LoDTensor>();
        if (tensor->IsInitialized()) {
          auto dim = var->GetMutable<LoDTensor>()->dims();
          auto new_shape = phi::vectorize(dim);
          output->Var()->SetShape(new_shape);
          op_desc->SetAttr("shape", new_shape);
          if (VLOG_IS_ON(10)) {
            std::ostringstream sout;
            sout << "InferShapePass: output[" << output->Name()
                 << "], infer shape: " << dim.to_str();
            VLOG(10) << sout.str();
          }
        }
      }
    }
    if (use_hlir_do_infer_ops.count(node->Name()) > 0) {
      do_infer_by_hlir_builder.Run(*scope, node);
      continue;
    }
  }
  scope.reset();

  if (VLOG_IS_ON(10)) {
    std::cout << "After GcuInferShape Graph: \n" << DebugString(graph);
  }
  VLOG(10) << "leave GcuInferShape";
}

static bool CheckTensorHasNanOrInf(const std::string &program_key,
                                   const std::string &tensor_name,
                                   LoDTensor *tensor) {
  if (tensor->dtype() != DataType::FLOAT32) {
    VLOG(6) << "Skip check tensor " << tensor_name << ", its dtype is "
            << tensor->dtype();
    return false;
  }

  LoDTensor cpu_tensor;
  cpu_tensor.Resize(tensor->dims());
  float *cpu_data = static_cast<float *>(
      cpu_tensor.mutable_data(platform::CPUPlace(), tensor->dtype()));

  framework::TensorCopySync(*tensor, platform::CPUPlace(), &cpu_tensor);
  bool has_nan_inf = false;
  for (int i = 0; i < cpu_tensor.numel(); i++) {
    if (std::isnan(cpu_data[i]) || std::isinf(cpu_data[i])) {
      has_nan_inf = true;
      break;
    }
  }
  if (has_nan_inf) {
    VLOG(0) << "Program " << program_key << " output tensor " << tensor_name
            << " contains Inf or Nan.";
  }
  return has_nan_inf;
}

static void CompileAndRunProgram(
    const platform::Place &ctx_place,
    const int64_t &program_id,
    const std::string &program_key,
    std::vector<std::string> &input_names,   // NOLINT
    std::vector<std::string> &output_names,  // NOLINT
    const std::vector<LoDTensor *> &inputs,
    std::vector<LoDTensor *> &outputs,  // NOLINT
    framework::Scope &scope,            // NOLINT
    const std::shared_ptr<framework::ir::Graph> &graph,
    const int train_flag = 1) {
  VLOG(3) << "start CompileAndRunProgram program_key: " << program_key;

  if (VLOG_IS_ON(3)) {
    std::cout << "CompileAndRunProgram graph: \n"
              << framework::ir::DebugString(graph.get()) << std::endl;
  }
  int device_id = ctx_place.GetDeviceId();
  if (!platform::gcu::runtime::GcuHasRuntimeInfo(device_id)) {
    int rank_id = 0;
    auto rt_info = std::make_shared<GcuRunTimeInfo>(device_id, false, 0, 0, 0);
    PADDLE_ENFORCE_NE(
        rt_info,
        nullptr,
        platform::errors::PreconditionNotMet(
            "Failed to create rt_info on GCU rank:%d, device_id:%d",
            rank_id,
            device_id));
    platform::gcu::runtime::GcuSetCurrentDevice(device_id);
    platform::gcu::runtime::GcuSetRuntimeInfo(device_id, rt_info);
  }

  auto executables =
      platform::gcu::TransformUtil::GetGcuExecutable(program_key);
  if (executables.size() <= 0) {
    auto tmp_gcu_compiler = std::make_shared<SingleOpGcuCompiler>(&scope);
    auto var_fixed_map = tmp_gcu_compiler->GetVarFixedMap();
    std::map<std::string, framework::VarDesc *> input_var_nodes;
    for (Node *node : graph.get()->Nodes()) {
      if (node->IsOp()) continue;
      auto op_name = node->Name();
      if ((node->inputs.size() <= 0) && (node->outputs.size() > 0)) {
        input_var_nodes[op_name] = node->Var();
      }
      // fix shape about dyn such as interpolate
      if (var_fixed_map.count(op_name) != 0) {
        node->Var()->SetShape(var_fixed_map[op_name]);
      }
    }
    RefreshDynamicInputVarShape(scope, input_var_nodes);
    GcuInferShape(graph.get());

    auto gcu_compiler_ = std::make_shared<SingleOpGcuCompiler>(&scope);
    gcu_compiler_->Compile({graph.get()},
                           {input_names},
                           {output_names},
                           inputs,
                           all_tmp_node_names_[program_id],
                           program_key);
  }
  // run exec
  std::vector<const LoDTensor *> real_inputs;
  real_inputs.reserve(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    VLOG(3) << "real input " << i << " name: " << input_names[i]
            << " initialized: " << inputs[i]->initialized()
            << " dims: " << inputs[i]->dims();
    if (inputs[i]->initialized()) {
      real_inputs.emplace_back(inputs[i]);
    }
  }

  VLOG(3) << "input_names: " << input_names.size()
          << " output_names: " << output_names.size()
          << " real_inputs: " << real_inputs.size()
          << " outputs: " << outputs.size();

  VLOG(3) << "== target == scope ptr:" << (int64_t)(&scope);
  std::shared_ptr<SingleOpGcuExecutor> gcu_exec = nullptr;
  static const char *use_cache = std::getenv(use_gcu_cache_executor);
  static bool use_new_executor =
      (use_cache == nullptr || std::string(use_cache) != "true");
  if (use_new_executor) {
    VLOG(3) << "CompileAndRunProgram use_new_executor. ";
    gcu_exec = std::make_shared<SingleOpGcuExecutor>(&scope);
    gcu_exec->RunGcuOp(
        real_inputs, outputs, ctx_place, program_key, train_flag);
    gcu_exec->ReleaseMemory();
  } else {
    auto manager = SingleOpGcuExecutorManager::GetInstance();
    gcu_exec = manager->Find(program_key);
    if (gcu_exec == nullptr) {
      gcu_exec = std::make_shared<SingleOpGcuExecutor>(&scope);
      manager->Add(program_key, gcu_exec);
    }
    gcu_exec->RunGcuOp(
        real_inputs, outputs, ctx_place, program_key, train_flag, &scope);
  }

  if (FLAGS_check_nan_inf) {
    VLOG(3) << "Check nan or inf for program " << program_key;
    bool has_nan_inf = false;
    for (size_t i = 0; i < outputs.size(); ++i) {
      auto *out = outputs[i];
      has_nan_inf |= CheckTensorHasNanOrInf(program_key, output_names[i], out);
    }
    PADDLE_ENFORCE_NE(has_nan_inf, true);
  }

  VLOG(3) << "end CompileAndRunProgram program_key: " << program_key;
}

static void GetTensors(
    const paddle::framework::BlockDesc &global_block,
    framework::Scope &scope,                      // NOLINT
    std::map<std::string, LoDTensor *> &tensors,  // NOLINT
    const std::vector<std::string> &var_names,
    const std::map<std::string, framework::VarDesc *> graph_var_nodes,
    bool skip_zero_memory = true) {
  for (size_t i = 0; i < var_names.size(); ++i) {
    if (var_names[i] == framework::kEmptyVarName ||
        kUnusedArchetype.count(var_names[i]) > 0 ||
        var_names[i] == "Fake_var" || !global_block.HasVar(var_names[i])) {
      continue;
    }

    auto numel = [](std::vector<int64_t> &shape) -> int64_t {
      int64_t res = 1;
      for (auto dim : shape) {
        res *= dim;
      }
      return res;
    };

    auto *var = scope.FindVar(var_names[i]);
    VLOG(6) << "GetTensors var name: " << var_names[i]
            << ", is nullptr: " << (var == nullptr);
    std::vector<int64_t> shape = graph_var_nodes.at(var_names[i])->GetShape();
    if (graph_var_nodes.count(var_names[i])) {
      std::vector<int64_t> shape = graph_var_nodes.at(var_names[i])->GetShape();
      VLOG(6) << "GetTensors var name:" << var_names[i]
              << ", shape:" << phi::make_ddim(shape).to_str()
              << ", numel:" << numel(shape);
      if (numel(shape) != 0 || !skip_zero_memory) {
        tensors[var_names[i]] = var->GetMutable<LoDTensor>();
      }
    } else {
      VLOG(6) << "GetTensors not find var in graph, var name:" << var_names[i]
              << ", shape:" << phi::make_ddim(shape).to_str()
              << ", numel:" << numel(shape);
    }
  }
}

inline void GcuRunProgramAPI(const std::shared_ptr<framework::ir::Graph> &graph,
                             paddle::framework::Scope &scope,  // NOLINT
                             const paddle::framework::BlockDesc &global_block,
                             bool retain_tmp_var,
                             int64_t program_id,
                             std::vector<std::string> input_var_names,
                             std::vector<std::string> param_names,
                             std::vector<std::string> output_var_names,
                             std::vector<std::string> dout_var_names,
                             const platform::Place &place) {
  auto tmp_gcu_compiler = std::make_shared<SingleOpGcuCompiler>(&scope);
  auto var_fixed_map = tmp_gcu_compiler->GetVarFixedMap();
  std::map<std::string, framework::VarDesc *> graph_var_nodes;
  std::map<std::string, framework::VarDesc *> no_memory_nodes;
  std::set<std::string> no_useful_nodes;
  for (Node *node : graph.get()->Nodes()) {
    if (node->IsVar()) continue;
    auto op_name = node->Name();
    for (const auto &out : node->Op()->Outputs()) {
      if (kUnusedArchetype.count(out.first) == 0) {
        continue;
      }
      for (const auto &name : out.second) {
        no_useful_nodes.insert(name);
      }
    }
  }
  for (Node *node : graph.get()->Nodes()) {
    if (node->IsOp()) {
      auto op = node->Op();
      if (op->Type() != "reshape2" && op->Type() != "transpose2") {
        continue;
      }
      auto no_memory_node_name = op->Output("XShape");
      if (no_memory_node_name.size() <= 0) {
        continue;
      }
      for (auto *out : node->outputs) {
        if (out->IsOp()) {
          continue;
        }
        if (out->Name() == no_memory_node_name[0]) {
          VLOG(6) << "no_memory_node_name: " << no_memory_node_name[0];
          no_memory_nodes[no_memory_node_name[0]] = out->Var();
          break;
        }
      }
      continue;
    }
    auto op_name = node->Name();
    if (no_useful_nodes.count(op_name) == 0) {
      graph_var_nodes[op_name] = node->Var();
    }
    // fix shape about dyn such as interpolate
    if (var_fixed_map.count(op_name) != 0) {
      node->Var()->SetShape(var_fixed_map[op_name]);
    }
  }

  for (auto iter : graph_var_nodes) {
    scope.Var(iter.first);
  }

  // prepare inputs and outputs
  std::map<std::string, LoDTensor *> map_inputs;
  std::map<std::string, LoDTensor *> map_outputs;

  // inputs
  GetTensors(
      global_block, scope, map_inputs, input_var_names, graph_var_nodes, false);

  // outputs
  GetTensors(
      global_block, scope, map_outputs, output_var_names, graph_var_nodes);
  GetTensors(global_block, scope, map_outputs, dout_var_names, graph_var_nodes);

  if (retain_tmp_var) {
    // tmp tensor
    std::set<std::string> skip_eager_delete_vars =
        forward_skip_eager_vars_.at(program_id);
    std::vector<std::string> tmp_tensor_names;
    // tmp_tensor_names.reserve(skip_eager_delete_vars.size());
    for (auto iter : graph_var_nodes) {
      if ((map_inputs.count(iter.first) <= 0) &&
          std::find(param_names.begin(), param_names.end(), iter.first) ==
              param_names.end() &&
          (map_outputs.count(iter.first) <= 0) &&
          //   (skip_eager_delete_vars.count(iter.first) <= 0) &&
          (no_memory_nodes.count(iter.first) <= 0)) {
        tmp_tensor_names.emplace_back(iter.first);
      }
    }
    GetTensors(
        global_block, scope, map_outputs, tmp_tensor_names, graph_var_nodes);

    all_tmp_node_names_[program_id] =
        std::set<std::string>(tmp_tensor_names.begin(), tmp_tensor_names.end());
  }

  std::vector<LoDTensor *> inputs;
  std::vector<LoDTensor *> outputs;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  size_t io_cnt = 0;
  for (auto tensor_info : map_inputs) {
    input_names.push_back(tensor_info.first);
    inputs.push_back(tensor_info.second);
    VLOG(6) << "GcuRunProgramOpKernel inputs[" << (io_cnt++)
            << "] = " << tensor_info.first;
  }
  io_cnt = 0;
  for (auto tensor_info : map_outputs) {
    output_names.push_back(tensor_info.first);
    outputs.push_back(tensor_info.second);
    VLOG(6) << "GcuRunProgramOpKernel outputs[" << (io_cnt++)
            << "] = " << tensor_info.first;
  }

  std::hash<std::string> hasher;
  std::string key_str = std::to_string(program_id) + "GcuRunProgramOpKernel";
  for (size_t i = 0; i < inputs.size(); ++i) {
    key_str += input_names[i] + inputs[i]->dims().to_str() + "; ";
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    key_str += output_names[i] + "; ";
  }

  auto program_key = std::to_string(hasher(key_str));
  VLOG(3) << "GcuRunProgramOpKernel program_id: " << program_id
          << " program_key: " << program_key;

  // run program
  int train_flag = (retain_tmp_var) ? 1 : 2;
  CompileAndRunProgram(place,
                       program_id,
                       program_key,
                       input_names,
                       output_names,
                       inputs,
                       outputs,
                       scope,
                       graph,
                       train_flag);

  // refresh no_memory_nodes shape
  for (auto var_node : no_memory_nodes) {
    auto *var = scope.FindVar(var_node.first);
    (var_node.second->GetType() == framework::proto::VarType::LOD_TENSOR);

    if ((var != nullptr) &&
        (var_node.second->GetType() == framework::proto::VarType::LOD_TENSOR)) {
      auto tensor = var->GetMutable<LoDTensor>();
      auto tensor_dims = var->GetMutable<LoDTensor>()->dims();
      auto tensor_shape = phi::vectorize(tensor_dims);
      auto var_shape = var_node.second->GetShape();
      VLOG(6) << "input var_name:" << var_node.first << " "
              << "var shape:" << TransformUtil::GetShapeStr(var_shape)
              << " tensor shape:" << tensor_dims.to_str() << " "
              << " "
              << "[WARN]use var shape to flush tensor shape!";
      tensor->Resize(phi::make_ddim(var_shape));
    }
  }
}

inline void GcuRunProgramGradAPI(
    const std::shared_ptr<framework::ir::Graph> &graph,
    paddle::framework::Scope &scope,  // NOLINT
    const paddle::framework::BlockDesc &global_block,
    int64_t program_id,
    std::vector<std::string> output_grad_var_names,
    std::vector<std::string> input_grad_var_names,
    std::vector<std::string> param_grad_names,
    const platform::Place &place) {
  auto tmp_gcu_compiler = std::make_shared<SingleOpGcuCompiler>(&scope);
  auto var_fixed_map = tmp_gcu_compiler->GetVarFixedMap();
  std::map<std::string, framework::VarDesc *> graph_var_nodes;
  for (Node *node : graph.get()->Nodes()) {
    if (node->IsOp()) continue;
    auto op_name = node->Name();
    graph_var_nodes[op_name] = node->Var();
    // fix shape about dyn such as interpolate
    if (var_fixed_map.count(op_name) != 0) {
      node->Var()->SetShape(var_fixed_map[op_name]);
    }
  }

  for (auto iter : graph_var_nodes) {
    scope.Var(iter.first);
  }

  // prepare inputs and outputs
  std::map<std::string, LoDTensor *> map_inputs;
  std::map<std::string, LoDTensor *> map_outputs;

  GetTensors(global_block,
             scope,
             map_inputs,
             output_grad_var_names,
             graph_var_nodes,
             false);

  // set tmp tensor as inputs
  auto &tmp_node_names = all_tmp_node_names_.at(program_id);
  std::vector<std::string> tmp_var_names(tmp_node_names.begin(),
                                         tmp_node_names.end());

  GetTensors(global_block, scope, map_inputs, tmp_var_names, graph_var_nodes);

  // outputs
  GetTensors(
      global_block, scope, map_outputs, input_grad_var_names, graph_var_nodes);
  GetTensors(
      global_block, scope, map_outputs, param_grad_names, graph_var_nodes);

  std::vector<LoDTensor *> inputs;
  std::vector<LoDTensor *> outputs;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;

  size_t io_cnt = 0;
  for (auto tensor_info : map_inputs) {
    input_names.push_back(tensor_info.first);
    inputs.push_back(tensor_info.second);
    VLOG(6) << "GcuRunProgramGradOpKernel inputs[" << (io_cnt++)
            << "] = " << tensor_info.first;
  }
  io_cnt = 0;
  for (auto tensor_info : map_outputs) {
    output_names.push_back(tensor_info.first);
    outputs.push_back(tensor_info.second);
    VLOG(6) << "GcuRunProgramGradOpKernel outputs[" << (io_cnt++)
            << "] = " << tensor_info.first;
  }

  std::hash<std::string> hasher;
  std::string key_str =
      std::to_string(program_id) + "GcuRunProgramGradOpKernel";
  for (size_t i = 0; i < inputs.size(); ++i) {
    key_str += input_names[i] + inputs[i]->dims().to_str() + "; ";
  }
  for (size_t i = 0; i < output_names.size(); ++i) {
    key_str += output_names[i] + "; ";
  }

  auto program_key = std::to_string(hasher(key_str));
  VLOG(3) << "GcuRunProgramGradOpKernel program_id: " << program_id
          << " program_key: " << program_key;

  // run program
  CompileAndRunProgram(place,
                       program_id,
                       program_key,
                       input_names,
                       output_names,
                       inputs,
                       outputs,
                       scope,
                       graph,
                       1);
  all_tmp_node_names_.erase(program_id);
}
}  // namespace gcu
}  // namespace paddle
#endif

namespace details {
using Tensor = paddle::Tensor;

static std::vector<Tensor> DereferenceTensors(
    const std::vector<Tensor *> &tensor_ptr) {
  std::vector<Tensor> res;
  for (auto *t : tensor_ptr) {
    res.emplace_back(*t);
  }
  return res;
}

static std::vector<std::string> GetTensorsName(const std::vector<Tensor> &ins) {
  std::vector<std::string> in_names;
  for (auto &in_t : ins) {
    in_names.emplace_back(in_t.name());
  }
  return in_names;
}

static std::vector<std::string> GetTensorsName(
    const std::vector<Tensor *> &ins) {
  std::vector<std::string> in_names;
  for (auto *in_t : ins) {
    in_names.emplace_back(in_t->name());
  }
  return in_names;
}

static void CheckInputVarStatus(const Tensor &tensor) {
  PADDLE_ENFORCE_EQ(tensor.defined() && tensor.is_dense_tensor(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "The input tensor %s of "
                        "RunProgram(Grad)Op holds "
                        "wrong type. Expect type is DenseTensor.",
                        tensor.name()));

  PADDLE_ENFORCE_EQ(
      static_cast<phi::DenseTensor *>(tensor.impl().get())->IsInitialized(),
      true,
      paddle::platform::errors::InvalidArgument(
          "The tensor in input tensor %s of "
          "RunProgram(Grad)Op "
          "is not initialized.",
          tensor.name()));
}

static void CheckOutputVarStatus(const paddle::framework::Variable &src_var,
                                 const Tensor &dst_tensor) {
  auto name = dst_tensor.name();
  PADDLE_ENFORCE_EQ(dst_tensor.defined(),
                    true,
                    paddle::platform::errors::InvalidArgument(
                        "dst_tensor `%s` shall be defined.", name));

  if (dst_tensor.is_dense_tensor()) {
    auto &src_tensor = src_var.Get<phi::DenseTensor>();
    PADDLE_ENFORCE_EQ(phi::DenseTensor::classof(&src_tensor),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The output tensor %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is DenseTensor",
                          name));
    PADDLE_ENFORCE_EQ(src_tensor.IsInitialized(),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The tensor in output tensor %s get from "
                          "RunProgram(Grad)Op's internal "
                          "scope is not initialized.",
                          name));
  } else if (dst_tensor.is_selected_rows()) {
    auto &src_tensor = src_var.Get<phi::SelectedRows>();
    PADDLE_ENFORCE_EQ(phi::SelectedRows::classof(&src_tensor),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The output tensodfr %s get from "
                          "RunProgram(Grad)Op's internal scope holds "
                          "wrong type. Expect type is SelectedRows",
                          name));
    PADDLE_ENFORCE_EQ(src_tensor.initialized(),
                      true,
                      paddle::platform::errors::InvalidArgument(
                          "The tensor in output tensor %s get from "
                          "RunProgram(Grad)Op's "
                          "internal scope is not initialized.",
                          name));

  } else {
    PADDLE_THROW(paddle::platform::errors::InvalidArgument(
        "The RunProgram(Grad)Op only support output "
        "variable of type LoDTensor or SelectedRows",
        name));
  }
}

static void ShareTensorsIntoScope(const std::vector<Tensor> &tensors,
                                  paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto name = tensors[i].name();
    if (name == "Fake_var") {
      continue;
    }
    auto *var = scope->Var(name);
    CheckInputVarStatus(tensors[i]);
    // share tensor
    auto tensor_base = tensors[i].impl();
    if (phi::DenseTensor::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::DenseTensor>();
      auto t = std::dynamic_pointer_cast<phi::DenseTensor>(tensor_base);
      *dst_tensor = *t;
    } else if (phi::SelectedRows::classof(tensor_base.get())) {
      auto *dst_tensor = var->GetMutable<phi::SelectedRows>();
      auto t = std::dynamic_pointer_cast<phi::SelectedRows>(tensor_base);
      *dst_tensor = *t;
    }
  }
}

static void ShareTensorsFromScope(
    const std::vector<Tensor *> &tensors,
    const paddle::framework::BlockDesc &global_block,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    // NOTE: In case of setting out_tmp.stop_gradient = True in model code, all
    // parameters before generating out_tmp have no @GRAD, it will raise error
    // because we can't find them in scope. So we skip sharing these vars or
    // var@GRAD if they don't appear in global block.
    auto &name = tensors[i]->name();
    if (name == paddle::framework::kEmptyVarName || name == "Fake_var" ||
        !global_block.HasVar(name)) {
      VLOG(2) << "find tensor name is " << name << ", skip it!";
      continue;
    }
    // NOTE: Here skip not found var is dangerous, if a bug is caused here,
    // the result is grad calculation error, which will be very hidden!
    auto *var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        paddle::platform::errors::NotFound("The output tensor %s is not in "
                                           "RunProgram(Grad)Op'"
                                           "s internal scope.",
                                           name));
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    if (var->IsType<phi::DenseTensor>()) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
      VLOG(2) << "share " << name << " from scope";
      *dst_tensor = src_tensor;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    }
  }
}

static void ShareTensorsFromScopeWithPartialBlock(
    const std::vector<Tensor *> &tensors,
    const paddle::framework::BlockDesc &forward_global_block,
    const paddle::framework::BlockDesc &backward_global_block,
    paddle::framework::Scope *scope) {
  for (size_t i = 0; i < tensors.size(); ++i) {
    auto &name = tensors[i]->name();
    if (name == paddle::framework::kEmptyVarName || name == "Fake_var" ||
        (!forward_global_block.HasVar(name) &&
         !backward_global_block.HasVar(name))) {
      VLOG(2) << "find tensor name is " << name << ", skip it!";
      continue;
    }
    auto *var = scope->FindVar(name);
    PADDLE_ENFORCE_NOT_NULL(
        var,
        paddle::platform::errors::NotFound("The output tensor %s is not in "
                                           "RunProgram(Grad)Op'"
                                           "s internal scope.",
                                           name));
    CheckOutputVarStatus(*var, *tensors[i]);
    // share tensor
    if (var->IsType<phi::DenseTensor>()) {
      auto &src_tensor = var->Get<phi::DenseTensor>();
      auto *dst_tensor = const_cast<phi::DenseTensor *>(
          dynamic_cast<const phi::DenseTensor *>(tensors[i]->impl().get()));
      VLOG(2) << "share " << name << " from scope";
      *dst_tensor = src_tensor;
    } else if (var->IsType<phi::SelectedRows>()) {
      auto &src_tensor = var->Get<phi::SelectedRows>();
      auto *dst_tensor = const_cast<phi::SelectedRows *>(
          dynamic_cast<const phi::SelectedRows *>(tensors[i]->impl().get()));
      *dst_tensor = src_tensor;
    }
  }
}

static void BuildScopeByBlock(
    const paddle::framework::InterpreterCore &interpreter_core,
    const paddle::framework::BlockDesc &block,
    paddle::framework::Scope *scope) {
  for (auto &var_desc : block.AllVars()) {
    auto var_name = var_desc->Name();
    if (var_name == paddle::framework::kEmptyVarName) {
      continue;
    }
    if (!scope->FindLocalVar(var_name)) {
      auto *ptr = scope->Var(var_name);
      InitializeVariable(ptr, var_desc->GetType());
      VLOG(2) << "Initialize Block Variable " << var_name;
    }
  }
  auto &data_transfer_added_vars =
      interpreter_core.GetVariableScope()->DataTransferAddedVars();
  for (size_t i = 0; i < data_transfer_added_vars.size(); i++) {
    auto *ptr = scope->Var(data_transfer_added_vars[i].first);
    InitializeVariable(ptr,
                       static_cast<paddle::framework::proto::VarType::Type>(
                           data_transfer_added_vars[i].second));
    VLOG(2) << "Initialize Transfer Added Variable "
            << data_transfer_added_vars[i].first;
  }
}

static void GcScope(paddle::framework::Scope *scope) {
  std::deque<std::shared_ptr<paddle::memory::Allocation>> *garbages =
      new std::deque<std::shared_ptr<paddle::memory::Allocation>>();

  for (auto &var : scope->LocalVars()) {
    if (var != nullptr) {
      if (var->IsType<phi::DenseTensor>()) {
        garbages->emplace_back(
            var->GetMutable<phi::DenseTensor>()->MoveMemoryHolder());
      }
      if (var->IsType<phi::SelectedRows>()) {
        garbages->emplace_back(var->GetMutable<phi::SelectedRows>()
                                   ->mutable_value()
                                   ->MoveMemoryHolder());
      }
      if (var->IsType<paddle::framework::LoDTensorArray>()) {
        auto *lod_tensor_arr =
            var->GetMutable<paddle::framework::LoDTensorArray>();
        for (auto &t : *lod_tensor_arr) {
          garbages->emplace_back(t.MoveMemoryHolder());
        }
        lod_tensor_arr->clear();
      }
    }
  }
  delete garbages;  // free mem
}

}  // namespace details

inline void RunProgramAPI(
    const std::vector<paddle::Tensor> &x,
    const std::vector<paddle::Tensor> &params,
    std::vector<paddle::Tensor *> &out,                   // NOLINT
    std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    std::vector<paddle::Tensor *> &dout,                  // NOLINT
    bool require_any_grad,
    const paddle::framework::AttributeMap &attrs) {
  VLOG(2) << "RunProgramOpKernel Compute";
  // In the original run_program OP, the default value of the is_test
  // attribute is false, we should check if there is is_test parameter
  // in attrs
  auto is_test = false;
  if (attrs.count("is_test")) {
    is_test = PADDLE_GET_CONST(bool, attrs.at("is_test"));
  }
  auto program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));
  auto place = egr::Controller::Instance().GetExpectedPlace();

  // NOTE(chenweihang): In order not to add new variable type, use vector
  // here. Originally, here can use scope directly.
  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      paddle::platform::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  VLOG(2) << "RunProgramOp use interpretercore to execute program.";

  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();

  VLOG(4) << "global_inner_scope:" << global_inner_scope;

  auto input_names = details::GetTensorsName(x);
  auto output_names = details::GetTensorsName(out);
  auto param_names = details::GetTensorsName(params);
  auto dout_names = details::GetTensorsName(dout);

  if (VLOG_IS_ON(6)) {
    std::stringstream s;
    s << "input_names: ";
    for (auto name : input_names) {
      s << name << " ";
    }
    s << std::endl;
    s << "param_names: ";
    for (auto name : param_names) {
      s << name << " ";
    }
    s << std::endl;
    s << "output_names: ";
    for (auto name : output_names) {
      s << name << " ";
    }
    s << std::endl;
    s << "dout_names: ";
    for (auto name : dout_names) {
      s << name << " ";
    }
    s << std::endl;
    VLOG(6) << s.str();
  }

  auto *forward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("forward_global_block"));
  auto *backward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("backward_global_block"));
  auto *forward_program = forward_global_block->Program();
  auto *backward_program = backward_global_block->Program();

  auto &interpretercore_info_cache =
      paddle::framework::InterpreterCoreInfoCache::Instance();
  std::shared_ptr<paddle::framework::InterpreterCore> interpreter_core =
      nullptr;
  if (!interpretercore_info_cache.Has(program_id, /*is_grad=*/false)) {
    paddle::platform::RecordEvent record_event(
        "create_new_interpretercore",
        paddle::platform::TracerEventType::UserDefined,
        1);
    VLOG(2) << "No interpretercore cahce, so create a new interpretercore "
               "for program: "
            << program_id;
    // Step 1. share input_vars & parameters into scope
    details::ShareTensorsIntoScope(x, global_inner_scope);
    details::ShareTensorsIntoScope(params, global_inner_scope);
    // Step 2. create new interpretercore
    interpreter_core =
        paddle::framework::CreateInterpreterCoreInfoToCache(*forward_program,
                                                            place,
                                                            /*is_grad=*/false,
                                                            program_id,
                                                            global_inner_scope);
    // Step 3. get all eager gc vars
    std::set<std::string> skip_eager_delete_vars =
        paddle::framework::details::ParseSafeEagerDeletionSkipVarsSet(
            *backward_program);
    // all out_vars are skip_eager_var
    skip_eager_delete_vars.insert(output_names.begin(), output_names.end());
    skip_eager_delete_vars.insert(dout_names.begin(), dout_names.end());
    // update interpretercore skip_gc_var
    interpreter_core->SetSkipGcVars(skip_eager_delete_vars);

    std::set<std::string> input_vars;
    input_vars.insert(input_names.begin(), input_names.end());
    interpreter_core->SetJitInputVars(input_vars);

    if (VLOG_IS_ON(6)) {
      std::stringstream s;
      s << "skip_eager_delete_vars: ";
      for (auto name : skip_eager_delete_vars) {
        s << name << " ";
      }
      VLOG(6) << s.str();
    }

#ifdef PADDLE_WITH_GCU
    paddle::gcu::forward_skip_eager_vars_[program_id] = skip_eager_delete_vars;
#endif

    interpretercore_info_cache.UpdateSkipEagerDeleteVars(
        program_id, false, skip_eager_delete_vars);
    VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
  } else {
    paddle::platform::RecordEvent record_event(
        "get_interpretercore_cahce",
        paddle::platform::TracerEventType::UserDefined,
        1);
    VLOG(2) << "Get interpretercore cahce by program:" << program_id;
    // Step 1. get cache interpretercore
    auto &cached_value =
        interpretercore_info_cache.GetMutable(program_id, /*is_grad=*/false);
    interpreter_core = cached_value.core_;
    // Step 2. update scope for cache interpretercore
    details::ShareTensorsIntoScope(x, global_inner_scope);
    details::ShareTensorsIntoScope(params, global_inner_scope);
    if (interpreter_core->GetVariableScope()->GetMutableScope() !=
        global_inner_scope) {
      details::BuildScopeByBlock(
          *interpreter_core.get(), *forward_global_block, global_inner_scope);
      interpreter_core->reset_scope(global_inner_scope);
    }
  }

  // interpretercore run
  if (forward_global_block->OpSize() > 0) {
    if (place.GetDeviceType() == "gcu") {
      int64_t start_op_index = 0;
      int64_t end_op_index = forward_global_block->OpSize();
      auto graph = std::make_shared<paddle::framework::ir::Graph>(
          *forward_program, start_op_index, end_op_index);
      graph->Set<int>(paddle::platform::gcu::kGraphType,
                      new int(paddle::platform::gcu::FP));
      if (VLOG_IS_ON(6)) {
        std::cout << "forward_global_block\n"
                  << paddle::framework::ir::DebugString(graph.get()) + "\n\n";
      }

      paddle::gcu::GcuRunProgramAPI(graph,
                                    *global_inner_scope,
                                    *forward_global_block,
                                    (!is_test || require_any_grad),
                                    program_id,
                                    input_names,
                                    param_names,
                                    output_names,
                                    dout_names,
                                    place);
    } else {
      paddle::platform::RecordEvent record_event(
          "interpreter_core_run",
          paddle::platform::TracerEventType::UserDefined,
          1);
      interpreter_core->Run({});
    }
  }

  {
    paddle::platform::RecordEvent record_event(
        "fetch_and_gc", paddle::platform::TracerEventType::UserDefined, 1);
    // Get Output
    details::ShareTensorsFromScopeWithPartialBlock(
        out, *forward_global_block, *backward_global_block, global_inner_scope);
    details::ShareTensorsFromScopeWithPartialBlock(dout,
                                                   *forward_global_block,
                                                   *backward_global_block,
                                                   global_inner_scope);

    VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(out_scope_vec->front());

    if (is_test || !require_any_grad) {
      VLOG(4) << "don't require any grad, set this scope can reused";
      VLOG(4) << "is_test: " << is_test
              << ", require_any_grad: " << require_any_grad;
      global_inner_scope->SetCanReused(true);
      details::GcScope(global_inner_scope);
    } else {
      VLOG(4) << "not test, set this scope can not reused";
      global_inner_scope->SetCanReused(false);
    }
  }

#ifdef PADDLE_WITH_MKLDNN
  if (FLAGS_use_mkldnn) paddle::platform::DontClearMKLDNNCache(place);
#endif
}

inline void RunProgramGradAPI(
    const std::vector<paddle::Tensor> &x UNUSED,
    const std::vector<paddle::Tensor> &params UNUSED,
    const std::vector<paddle::Tensor> &out_grad,
    const std::vector<paddle::framework::Scope *> &step_scope,  // NOLINT
    const paddle::framework::AttributeMap &attrs,
    std::vector<paddle::Tensor *> &x_grad,      // NOLINT
    std::vector<paddle::Tensor *> &params_grad  // NOLINT
) {
  // if all output vars are set to stop_gradient, grad op no need to executed
  if (x_grad.empty() && params_grad.empty()) return;

  auto program_id = PADDLE_GET_CONST(int64_t, attrs.at("program_id"));

  auto *out_scope_vec = &step_scope;
  PADDLE_ENFORCE_EQ(
      out_scope_vec->size(),
      1,
      paddle::platform::errors::InvalidArgument(
          "The OutScope of RunProgramGradOp should only hold one scope."));

  auto place = egr::Controller::Instance().GetExpectedPlace();
  VLOG(2) << "RunProgramGradOp use interpretercore to execute program.";

  paddle::framework::Scope *global_inner_scope = out_scope_vec->front();
  VLOG(4) << "global_inner_scope:" << global_inner_scope;

  auto *forward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("forward_global_block"));
  auto *backward_global_block = PADDLE_GET_CONST(
      paddle::framework::BlockDesc *, attrs.at("backward_global_block"));
  auto *backward_program = backward_global_block->Program();

  auto out_grad_names = details::GetTensorsName(out_grad);
  auto &interpretercore_info_cache =
      paddle::framework::InterpreterCoreInfoCache::Instance();
  std::shared_ptr<paddle::framework::InterpreterCore> interpreter_core =
      nullptr;
  if (!interpretercore_info_cache.Has(program_id, /*is_grad=*/true)) {
    paddle::platform::RecordEvent record_event(
        "create_new_interpretercore",
        paddle::platform::TracerEventType::UserDefined,
        1);
    VLOG(2) << "No interpretercore cahce, so create a new interpretercore";
    details::ShareTensorsIntoScope(out_grad, global_inner_scope);
    interpreter_core =
        paddle::framework::CreateInterpreterCoreInfoToCache(*backward_program,
                                                            place,
                                                            /*is_grad=*/true,
                                                            program_id,
                                                            global_inner_scope);

    // share threadpool
    // NOTE(zhiqiu): this only works interpreter_core is executed strictly
    // after the related fwd_interpreter_core.
    if (interpretercore_info_cache.Has(program_id, false)) {
      auto fwd_interpreter_core =
          interpretercore_info_cache.GetMutable(program_id, /*is_grad=*/false)
              .core_;
      interpreter_core->ShareWorkQueueFrom(fwd_interpreter_core);
      VLOG(4) << "Share workqueue from " << fwd_interpreter_core.get() << " to "
              << interpreter_core.get();
    }

    std::vector<std::string> x_grad_names;
    std::vector<std::string> param_grad_names;
    if (!x_grad.empty()) {
      x_grad_names = details::GetTensorsName(x_grad);
    }
    if (!params_grad.empty()) {
      param_grad_names = details::GetTensorsName(params_grad);
    }
    // get all eager gc vars
    std::set<std::string> skip_eager_delete_vars;
    // all out_vars are skip_eager_var
    skip_eager_delete_vars.insert(x_grad_names.begin(), x_grad_names.end());
    // initialize skip gc vars by forward_program and backward_program
    paddle::framework::details::AppendSkipDeletionVars(param_grad_names,
                                                       &skip_eager_delete_vars);
    interpreter_core->SetSkipGcVars(skip_eager_delete_vars);
    interpretercore_info_cache.UpdateSkipEagerDeleteVars(
        program_id, /*is_grad=*/true, skip_eager_delete_vars);
    VLOG(2) << "Get skip GC vars size is: " << skip_eager_delete_vars.size();
  } else {
    paddle::platform::RecordEvent record_event(
        "get_interpretercore_cahce",
        paddle::platform::TracerEventType::UserDefined,
        1);
    VLOG(2) << "Get interpretercore cahce by program:" << program_id;
    auto &cached_value =
        interpretercore_info_cache.GetMutable(program_id, /*is_grad=*/true);
    interpreter_core = cached_value.core_;

    // update scope
    details::ShareTensorsIntoScope(out_grad, global_inner_scope);
    if (interpreter_core->GetVariableScope()->GetMutableScope() !=
        global_inner_scope) {
      details::BuildScopeByBlock(
          *interpreter_core.get(), *backward_global_block, global_inner_scope);
      interpreter_core->reset_scope(global_inner_scope);
    }
  }

  if (backward_global_block->OpSize() > 0) {
    if (place.GetDeviceType() == "gcu") {
      int64_t start_op_index = 0;
      int64_t end_op_index = backward_global_block->OpSize();
      auto graph = std::make_shared<paddle::framework::ir::Graph>(
          *backward_program, start_op_index, end_op_index);
      graph->Set<int>(paddle::platform::gcu::kGraphType,
                      new int(paddle::platform::gcu::BP));
      if (VLOG_IS_ON(6)) {
        std::cout << "backward_global_block\n"
                  << paddle::framework::ir::DebugString(graph.get()) + "\n\n";
      }

      std::vector<std::string> x_grad_names;
      std::vector<std::string> param_grad_names;
      if (!x_grad.empty()) {
        x_grad_names = details::GetTensorsName(x_grad);
      }
      if (!params_grad.empty()) {
        param_grad_names = details::GetTensorsName(params_grad);
      }

      paddle::gcu::GcuRunProgramGradAPI(graph,
                                        *global_inner_scope,
                                        *backward_global_block,
                                        program_id,
                                        out_grad_names,
                                        x_grad_names,
                                        param_grad_names,
                                        place);
    } else {
      paddle::platform::RecordEvent record_event(
          "interpreter_core_run",
          paddle::platform::TracerEventType::UserDefined,
          1);
      // Debug info: scope info when run end
      VLOG(3) << paddle::framework::GenScopeTreeDebugInfo(
          out_scope_vec->front());
      interpreter_core->Run({});
    }
  }

  {
    paddle::platform::RecordEvent record_event(
        "fetch_and_gc", paddle::platform::TracerEventType::UserDefined, 1);
    // Step 4. get outputs
    details::ShareTensorsFromScopeWithPartialBlock(x_grad,
                                                   *forward_global_block,
                                                   *backward_global_block,
                                                   global_inner_scope);
    details::ShareTensorsFromScopeWithPartialBlock(params_grad,
                                                   *forward_global_block,
                                                   *backward_global_block,
                                                   global_inner_scope);
    VLOG(4) << "after backward gc all vars";
    global_inner_scope->SetCanReused(true);
    details::GcScope(global_inner_scope);
  }
}

class GradNodeRunProgram : public egr::GradNodeBase {
 public:
  GradNodeRunProgram(size_t bwd_in_slot_num, size_t bwd_out_slot_num)
      : egr::GradNodeBase(bwd_in_slot_num, bwd_out_slot_num) {}

  ~GradNodeRunProgram() {
    if (!executed_) {
      auto *out_scope_vec = &step_scope_;
      VLOG(4) << "~GradNodeRunProgram";
      // Normally out_scope_vec.size() == 1. for safty, we add for-loop here.
      for (size_t i = 0; i < out_scope_vec->size(); ++i) {
        paddle::framework::Scope *global_inner_scope = out_scope_vec->at(i);
        global_inner_scope->SetCanReused(true);
        details::GcScope(global_inner_scope);
        VLOG(4) << "global_inner_scope SetCanReused";
      }
    }
  }
  // Functor: perform backward computations
  virtual paddle::small_vector<std::vector<paddle::Tensor>,
                               egr::kSlotSmallVectorSize>
  operator()(paddle::small_vector<std::vector<paddle::Tensor>,
                                  egr::kSlotSmallVectorSize> &grads,  // NOLINT
             bool create_graph UNUSED,
             bool is_new_grad UNUSED) override {
    VLOG(3) << "Running Eager Backward Node: GradNodeRunProgram";
    paddle::small_vector<std::vector<paddle::Tensor>, egr::kSlotSmallVectorSize>
        hooked_grads = GradNodeRunProgram::ApplyGradientHooks(grads);
    PADDLE_ENFORCE_EQ(hooked_grads.size(),
                      1,
                      paddle::platform::errors::InvalidArgument(
                          "The hooked_grads.size() of RunProgramGradOp should "
                          "be equal to 1."));

    std::vector<paddle::Tensor> x_grad;
    std::vector<paddle::Tensor> params_grad;
    std::vector<paddle::Tensor *> x_grad_ptr;
    std::vector<paddle::Tensor *> params_grad_ptr;
    {
      paddle::platform::RecordEvent record_event(
          "construct_grad_tensor",
          paddle::platform::TracerEventType::UserDefined,
          1);

      egr::EagerUtils::FillZeroForEmptyOptionalGradInput(&hooked_grads[0],
                                                         this->InputMeta()[0]);
      VLOG(3) << "hooked_grads[0].size() : " << hooked_grads[0].size();
      ConstructXGradTensors(x_, &x_grad);
      ConstructParamGradTensors(params_, &params_grad);
      for (auto &i : x_grad) {
        x_grad_ptr.emplace_back(&i);
      }
      for (auto &i : params_grad) {
        if (i.defined()) {
          params_grad_ptr.emplace_back(&i);
        }
      }
    }

    auto out_grad_names =
        PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("out_grad_names"));
    PADDLE_ENFORCE_EQ(hooked_grads[0].size(),
                      out_grad_names.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The hooked_grads[0].size() and "
                          "out_grad_names.size() should be equal."));
    for (size_t i = 0; i < out_grad_names.size(); ++i) {
      hooked_grads[0][i].set_name(out_grad_names[i]);
    }
    RunProgramGradAPI(x_,
                      params_,
                      hooked_grads[0],
                      step_scope_,
                      attrs_,
                      x_grad_ptr,
                      params_grad_ptr);
    VLOG(3) << "End Eager Backward Node: GradNodeRunProgram";

    executed_ = true;
    return {x_grad, params_grad};
  }

  void ClearTensorWrappers() override { VLOG(6) << "Do nothing here now"; }

  // SetAttrMap
  void SetAttrMap(const paddle::framework::AttributeMap &attrs) {
    attrs_ = attrs;
  }

  void SetFwdX(const std::vector<paddle::Tensor> &tensors) { x_ = tensors; }

  void SetFwdParams(const std::vector<paddle::Tensor> &tensors) {
    params_ = tensors;
  }

  void SetStepScope(const std::vector<paddle::framework::Scope *> &scopes) {
    step_scope_ = scopes;
  }

 protected:
  void ConstructXGradTensors(const std::vector<paddle::Tensor> &x,
                             std::vector<paddle::Tensor> *x_grad) {
    auto x_grad_names =
        PADDLE_GET_CONST(std::vector<std::string>, attrs_.at("x_grad_names"));
    PADDLE_ENFORCE_EQ(
        x.size(),
        x_grad_names.size(),
        paddle::platform::errors::InvalidArgument(
            "The x.size() and x_grad_names.size() should be equal. "
            "But received x.size() = %d, x_grad_names.size() = %d",
            x.size(),
            x_grad_names.size()));

    // TODO(dev): Need an elegant way to determine inforamtion of grad_tensor,
    // such as: name, tensor type(DenseTensor or SelectedRows).
    for (size_t i = 0; i < x.size(); i++) {
      if (x[i].is_dense_tensor()) {
        x_grad->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (x[i].is_selected_rows()) {
        x_grad->emplace_back(std::make_shared<phi::SelectedRows>());
      }
      x_grad->back().set_name(x_grad_names[i]);
    }
  }

  void ConstructParamGradTensors(const std::vector<paddle::Tensor> &params,
                                 std::vector<paddle::Tensor> *param_grads) {
    auto param_grad_names = PADDLE_GET_CONST(std::vector<std::string>,
                                             attrs_.at("param_grad_names"));
    PADDLE_ENFORCE_EQ(params.size(),
                      param_grad_names.size(),
                      paddle::platform::errors::InvalidArgument(
                          "The param.size() and "
                          "param_grad_names.size() should be equal."));

    for (size_t i = 0; i < params.size(); ++i) {
      auto &p = params[i];
      auto &p_grad = egr::EagerUtils::unsafe_autograd_meta(p)->Grad();
      // In eager mode, the number of param_grad should be the same as
      // param, so here an empty Tensor is added for the param with
      // stop_gradient=True
      if (!p_grad.defined()) {
        param_grads->emplace_back();
      } else if (p_grad.is_dense_tensor()) {
        param_grads->emplace_back(std::make_shared<phi::DenseTensor>());
      } else if (p_grad.is_selected_rows()) {
        param_grads->emplace_back(std::make_shared<phi::SelectedRows>());
      }
      param_grads->back().set_name(param_grad_names[i]);
    }
  }

  std::shared_ptr<GradNodeBase> Copy() const override {
    auto copied_node =
        std::shared_ptr<GradNodeRunProgram>(new GradNodeRunProgram(*this));
    return copied_node;
  }

 private:
  // TensorWrappers
  std::vector<paddle::Tensor> x_;
  std::vector<paddle::Tensor> params_;
  std::vector<paddle::framework::Scope *> step_scope_;

  // Attribute Map
  paddle::framework::AttributeMap attrs_;

  bool executed_{false};
};
