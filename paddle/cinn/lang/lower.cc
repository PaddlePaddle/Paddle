// Copyright (c) 2021 CINN Authors. All Rights Reserved.
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

#include "paddle/cinn/lang/lower.h"

#include <iostream>
#include <map>
#include <set>
#include <stack>
#include <unordered_set>
#include <utility>

#include "paddle/cinn/common/integer_set.h"
#include "paddle/cinn/ir/buffer.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/lang/lower_impl.h"
#include "paddle/cinn/lang/lower_tensor_group.h"
#include "paddle/cinn/optim/optimize.h"
#include "paddle/cinn/utils/string.h"

namespace cinn {
namespace lang {

using ast_gen_ius::TensorGroup;
using ir::Tensor;
using poly::Stage;

std::vector<ir::Argument> GetArgs(
    const Expr& func_body, const std::vector<std::string>& input_output_nodes) {
  std::vector<ir::Argument> res;
  std::map<std::string, std::set<const ir::Load*>> name2loads;
  std::map<std::string, std::set<const ir::Store*>> name2stores;
  auto load_or_store_nodes = ir::ir_utils::CollectIRNodesWithoutTensor(
      func_body,
      [&](const Expr* x) { return x->As<ir::Store>() || x->As<ir::Load>(); });

  for (auto&& e : load_or_store_nodes) {
    if (e.As<ir::Load>()) {
      auto&& tensor_name = e.As<ir::Load>()->tensor.as_tensor()->name;
      name2loads[tensor_name].insert(e.As<ir::Load>());
    } else {  // Store node
      auto&& tensor_name = e.As<ir::Store>()->tensor.as_tensor()->name;
      name2stores[tensor_name].insert(e.As<ir::Store>());
    }
  }

  for (auto&& node_name : input_output_nodes) {
    auto load_it = name2loads.find(node_name);
    auto store_it = name2stores.find(node_name);
    // if a node is ir::Load and also ir::Store, then process it as a ir::Store
    // in priority.
    if (store_it != name2stores.end()) {  //
      for (auto&& node : store_it->second) {
        const auto* tensor = node->tensor.as_tensor();
        if (tensor->buffer.defined()) {
          res.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
          break;
        }
      }
    } else if (load_it != name2loads.end()) {
      for (auto&& node : load_it->second) {
        const auto* tensor = node->tensor.as_tensor();
        if (tensor->buffer.defined()) {
          res.emplace_back(tensor->buffer, ir::Argument::IO::kInput);
          break;
        }
      }
    }
  }

  if (VLOG_IS_ON(3)) {
    for (auto& i : input_output_nodes)
      VLOG(3) << "In input_output_nodes, arg has : " << i;
    for (auto& i : res) VLOG(3) << "In res, arg has : " << i.name();
  }
  return res;
}

bool CanProveBufferNumelLT(const ir::Buffer& lhs, const ir::Buffer& rhs) {
  common::cas_intervals_t var_intervals;
  common::SymbolicExprAnalyzer analyzer(var_intervals);
  std::optional<bool> prove_lt =
      analyzer.ProveLT(lhs->SymbolicNumel(), rhs->SymbolicNumel());
  return prove_lt.value_or(false);
}

// Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(
    const std::vector<cinn::ir::Tensor>& tensor_args, Expr body) {
  std::unordered_set<std::string> tensor_arg_names;
  std::unordered_set<std::string> buffer_arg_names;
  for (auto& tensor : tensor_args) {
    tensor_arg_names.insert(tensor->name);
    if (tensor->buffer.defined()) {
      buffer_arg_names.insert(tensor->buffer->name);
    }
  }
  std::map<std::string, ir::Buffer>
      name_to_buffer;  // used to avoid duplication.

  auto all_temp_tensors =
      ir::ir_utils::CollectIRNodesWithoutTensor(body, [&](const Expr* x) {
        return x->as_tensor() && x->as_tensor()->buffer.defined() &&
               ((!buffer_arg_names.count(x->as_tensor()->buffer->name) &&
                 !tensor_arg_names.count(x->as_tensor()->name)) ||
                utils::EndsWith(x->as_tensor()->buffer->name, "temp_buffer"));
      });
  for (auto& e : all_temp_tensors) {
    auto buffer_name = e.as_tensor()->buffer->name;
    if (!name_to_buffer.count(buffer_name)) {
      name_to_buffer[buffer_name] = e.as_tensor()->buffer;
    } else {
      // TODO(phlrain): why update
      if (CanProveBufferNumelLT(e.as_tensor()->buffer,
                                name_to_buffer[buffer_name])) {
        name_to_buffer[buffer_name] = e.as_tensor()->buffer;
      }
    }
  }

  std::vector<ir::Buffer> temp_buffers;
  for (auto& i : name_to_buffer) {
    temp_buffers.push_back(i.second);
  }

  return temp_buffers;
}

//! Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(const std::vector<Tensor>& tensor_args,
                                       const TensorGroup& tensor_group,
                                       Expr body) {
  std::unordered_set<std::string> tensor_arg_names;
  std::unordered_set<std::string> buffer_arg_names;
  for (auto& tensor : tensor_args) {
    tensor_arg_names.insert(tensor->name);
    if (tensor->buffer.defined()) {
      buffer_arg_names.insert(tensor->buffer->name);
    }
  }
  std::map<std::string, ir::Buffer>
      name_to_buffer;  // used to avoid duplication.

  auto all_temp_tensors =
      ir::ir_utils::CollectIRNodesWithoutTensor(body, [&](const Expr* x) {
        return x->as_tensor() && x->as_tensor()->buffer.defined() &&
               (!tensor_group.Contain(x->as_tensor()->name) ||
                ((!buffer_arg_names.count(x->as_tensor()->buffer->name) &&
                  !tensor_arg_names.count(x->as_tensor()->name)) ||
                 utils::EndsWith(x->as_tensor()->buffer->name, "temp_buffer")));
      });
  for (auto& e : all_temp_tensors) {
    auto buffer_name = e.as_tensor()->buffer->name;
    if (!name_to_buffer.count(buffer_name)) {
      name_to_buffer[buffer_name] = e.as_tensor()->buffer;
    } else {
      // Just copy from old code, but why?
      if (e.as_tensor()->buffer->numel() <
          name_to_buffer[buffer_name]->numel()) {
        name_to_buffer[buffer_name] = e.as_tensor()->buffer;
      }
    }
  }

  std::vector<ir::Buffer> temp_buffers;
  for (auto& i : name_to_buffer) {
    temp_buffers.push_back(i.second);
  }
  return temp_buffers;
}

//! Collect the temporary tensors from a computational graph.
std::vector<ir::Buffer> GetTempBuffers(const std::vector<ir::Argument>& args,
                                       Expr body) {
  std::unordered_set<std::string> buffer_arg_names;
  for (auto& a : args) {
    if (a.is_buffer()) {
      buffer_arg_names.insert(a.name());
    }
  }
  std::map<std::string, ir::Buffer>
      name_to_buffer;  // used to avoid duplication.

  auto all_temp_tensors =
      ir::ir_utils::CollectIRNodesWithoutTensor(body, [&](const Expr* x) {
        return x->as_tensor() && x->as_tensor()->buffer.defined() &&
               (!buffer_arg_names.count(x->as_tensor()->buffer->name) ||
                utils::EndsWith(x->as_tensor()->buffer->name, "temp_buffer"));
      });
  for (auto& e : all_temp_tensors) {
    auto buffer_name = e.as_tensor()->buffer->name;
    if (!name_to_buffer.count(buffer_name)) {
      name_to_buffer[buffer_name] = e.as_tensor()->buffer;
    } else {
      if (e.as_tensor()->buffer->numel() <
          name_to_buffer[buffer_name]->numel()) {
        name_to_buffer[buffer_name] = e.as_tensor()->buffer;
      }
    }
  }
  // visit the ir body and update the map of name_to_buffer
  auto update_map =
      ir::ir_utils::CollectIRNodesWithoutTensor(body, [&](const Expr* x) {
        if (x->as_tensor() && x->as_tensor()->buffer.defined()) {
          auto buffer_name = x->as_tensor()->buffer->name;
          if (name_to_buffer.count(buffer_name) &&
              x->as_tensor()->buffer->numel() <
                  name_to_buffer[buffer_name]->numel()) {
            name_to_buffer[buffer_name] = x->as_tensor()->buffer;
          }
        }
        return x->as_tensor() && x->as_tensor()->buffer.defined();
      });

  std::vector<ir::Buffer> temp_buffers;
  for (auto& i : name_to_buffer) temp_buffers.push_back(i.second);
  return temp_buffers;
}

std::set<ir::Tensor> CollectTempTensorsFromCtrlDepends(
    ast_gen_ius::TensorGroup* tensor_group,
    const std::vector<Tensor>& tensor_args) {
  std::set<ir::Tensor> res;
  for (const ir::Tensor& a : tensor_group->GetAllTensors()) {
    for (const ir::Tensor& t : tensor_group->GetCtrlDepTensors(a->name)) {
      res.emplace(t);
    }
  }
  for (const ir::Tensor& t : tensor_args) {
    if (res.count(t)) {
      res.erase(t);
    }
  }
  return res;
}

ir::LoweredFunc LowerToAst(const std::string& name,
                           const std::vector<Tensor>& tensor_args,
                           ast_gen_ius::TensorGroup* tensor_group,
                           const Target& target) {
  std::vector<ir::LoweredFunc> result =
      LowerToAstVec(name, tensor_args, tensor_group, target);
  PADDLE_ENFORCE_EQ(result.size(),
                    1UL,
                    ::common::errors::InvalidArgument(
                        "LowerToAst contains not only 1 LoweredFunc, "
                        "use LowerToAstVec instead."));
  return result[0];
}

std::vector<ir::LoweredFunc> LowerToAstVec(
    const std::string& name,
    const std::vector<Tensor>& tensor_args,
    ast_gen_ius::TensorGroup* tensor_group,
    const Target& target) {
  std::set<ir::Tensor> ctrl_deps =
      CollectTempTensorsFromCtrlDepends(tensor_group, tensor_args);
  auto lower_instance = detail::LowerTensorGroup(
      name,
      tensor_args,
      {},
      tensor_group,
      std::vector<Tensor>(ctrl_deps.begin(), ctrl_deps.end()),
      target);
  std::vector<ir::LoweredFunc> result = lower_instance();
  for (auto& res : result) {
    target.arch.Match(
        [&](common::NVGPUArch) { res->device_api = ir::DeviceAPI::GPU; },
        [&](std::variant<common::HygonDCUArchHIP, common::HygonDCUArchSYCL>) {
          res->device_api = ir::DeviceAPI::GPU;
        },
        [&](std::variant<common::UnknownArch,
                         common::X86Arch,
                         common::ARMArch>) {});
  }
  return result;
}

}  // namespace lang
}  // namespace cinn
