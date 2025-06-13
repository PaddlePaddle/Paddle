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

#include "paddle/cinn/lang/lower_impl.h"

#include <algorithm>
#include <queue>
#include <string>
#include <unordered_set>

#include "paddle/cinn/common/common.h"
#include "paddle/cinn/common/context.h"
#include "paddle/cinn/common/ir_util.h"
#include "paddle/cinn/ir/ir_base.h"
#include "paddle/cinn/ir/ir_printer.h"
#include "paddle/cinn/ir/tensor.h"
#include "paddle/cinn/optim/ir_simplify.h"
#include "paddle/cinn/optim/replace_var_with_expr.h"
#include "paddle/cinn/optim/transform_polyfor_to_for.h"
#include "paddle/common/enforce.h"

PD_DECLARE_bool(cinn_runtime_display_debug_info);

namespace cinn {
namespace lang {
namespace detail {

void CheckNoIslCallRemains(Expr* expr) {
  auto isl_calls = ir::ir_utils::CollectIRNodes(*expr, [](const Expr* expr) {
    return expr->As<ir::Call>() && expr->As<ir::Call>()->is_isl_call();
  });
#ifdef CINN_DEBUG
  for (auto& item : isl_calls) {
    LOG(ERROR) << "ISL call: " << item;
  }
#endif
  if (!isl_calls.empty()) {
    LOG(WARNING) << "Some ISL call nodes remained, get " << isl_calls.size()
                 << " isl_calls, the first one is " << *isl_calls.begin();
  }
}

const char* CompuGraphNode::__type_info__ = "ComputeGraphNode";
const char* CompuGraphNode::type_info() const { return __type_info__; }
std::string CompuGraphNode::id() const {
  PADDLE_ENFORCE_EQ(
      tensor.defined(),
      true,
      ::common::errors::InvalidArgument("Tensor is not defined. Please ensure "
                                        "tensor is properly initialized."));
  return tensor->name;
}

void LowerImpl::CheckArgsUnique() {
  for (auto& tensor : tensor_args_) {
    if (!tensor->buffer.defined()) {
      LOG(ERROR) << "tensor [" << tensor->name << "] buffer is null";
      continue;
    }
  }
}

std::vector<ir::Argument> LowerImpl::GenerateFunctionArgumentList(
    Expr fn_body) {
  CheckArgsUnique();

  std::vector<ir::Argument> args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(&fn_body);

  std::set<std::string> arg_names;

  for (auto& scalar : scalar_args_) {
    PADDLE_ENFORCE_EQ(
        arg_names.count(scalar->name),
        0,
        ::common::errors::InvalidArgument(
            "Argument name '%s' already exists in the argument names set.",
            scalar->name));
    auto* scalar_node = scalar.As<ir::_Var_>();
    PADDLE_ENFORCE_EQ(
        scalar_node->type().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The type of scalar node '%s' is not valid.", scalar->name));
    arg_names.insert(scalar->name);

    args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  for (auto& tensor : tensor_args_) {
    auto* tensor_node = tensor.As<ir::_Tensor_>();
    bool is_output = teller.count(tensor->name);
    VLOG(1) << "tensor argument " << tensor->name << " buffer "
            << tensor->buffer->name;

    // avoid duplicate
    if (!tensor_node->buffer.defined()) continue;
    // if a argument is already marked as kInput, mark it as kOutput and move it
    // to the back.
    if (arg_names.count(tensor_node->buffer->name)) {
      auto it =
          std::find_if(args.begin(), args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor_node->buffer->name;
          });
      PADDLE_ENFORCE_EQ(
          it != args.end(),
          true,
          ::common::errors::InvalidArgument(
              "Argument with name '%s' not found in the argument list.",
              tensor_node->buffer->name));
      if (it->is_input()) {
        args.erase(it);
      } else if (it->is_output()) {
        continue;
      }
    }

    arg_names.insert(tensor_node->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    VLOG(3) << "Collect " << (is_output ? "W" : "R") << " argument "
            << tensor->buffer->name;
    args.emplace_back(tensor_node->buffer, io);
  }

  return args;
}
// Generate Function Arguments for split kernel.
std::vector<ir::Argument> LowerImpl::GenFuncArgForSplitKernel(
    Expr func_iterator, std::vector<ir::Tensor> temp_tensors) {
  CheckArgsUnique();

  std::vector<ir::Argument> in_args;
  std::vector<ir::Argument> out_args;
  auto teller = ir::ir_utils::CollectTensorNeedsWrite(&func_iterator);
  std::set<std::string> arg_names;
  std::set<std::string> all_tensor_names;

  for (auto& scalar : scalar_args_) {
    PADDLE_ENFORCE_EQ(
        arg_names.count(scalar->name),
        0,
        ::common::errors::InvalidArgument(
            "Argument name '%s' already exists in the argument names set.",
            scalar->name));
    auto* scalar_node = scalar.As<ir::_Var_>();
    PADDLE_ENFORCE_EQ(
        scalar_node->type().valid(),
        true,
        ::common::errors::InvalidArgument(
            "The type of scalar node '%s' is not valid.", scalar->name));
    arg_names.insert(scalar->name);

    in_args.emplace_back(scalar, ir::Argument::IO::kInput);
  }

  auto all_tensors = ir::ir_utils::CollectIRNodes(
      func_iterator, [&](const Expr* x) { return x->as_tensor(); });

  auto all_vars = ir::ir_utils::CollectIRNodes(
      func_iterator, [&](const Expr* x) { return x->as_var(); });

  for (auto& i : all_tensors) {
    auto* tensor = i.as_tensor();
    all_tensor_names.insert(tensor->name);
    VLOG(3) << "In all_tensors, it has : " << tensor->name;
  }
  for (auto& i : all_vars) {
    auto* var = i.as_var();
    VLOG(3) << "In all_vars, it has : " << var->name;
  }

  for (auto& i : scalar_args_) {
    VLOG(3) << "In scalar_args_, var has : " << i->name;
  }

  std::set<std::string> temp_tensor_names;

  for (auto& i : temp_tensors) {
    VLOG(3) << "In temp_tensors, it has : " << i->name;
    temp_tensor_names.insert(i->name);
  }

  for (auto& tensor : tensor_args_) {
    VLOG(3) << "In tensor_args_, it has : " << tensor->name;
    if (temp_tensor_names.count(tensor->name) > 0) continue;
    if (all_tensor_names.count(tensor->name) == 0) continue;
    bool is_output = teller.count(tensor->name);
    VLOG(3) << "tensor argument " << tensor->name << " buffer "
            << tensor->buffer->name;

    // avoid duplicate
    if (!tensor->buffer.defined()) {
      VLOG(3) << "tensor->buffer is not defined";
      continue;
    }
    // if a argument is already marked as kInput, mark it as kOutput and move it
    // to the back.
    if (arg_names.count(tensor->buffer->name)) {
      auto it = std::find_if(
          in_args.begin(), in_args.end(), [&](const ir::Argument& x) {
            return x.name() == tensor->buffer->name;
          });
      if (it != in_args.end()) {
        in_args.erase(it);
      } else {
        continue;
      }
    }

    arg_names.insert(tensor->buffer->name);

    auto io = is_output ? ir::Argument::IO::kOutput : ir::Argument::IO::kInput;
    if (io == ir::Argument::IO::kInput)
      in_args.emplace_back(tensor->buffer, io);
    else
      out_args.emplace_back(tensor->buffer, io);
  }
  if (out_args.empty()) {
    for (auto& i : all_tensors) {
      auto* tensor = i.as_tensor();
      VLOG(3) << "Tensor " << tensor->name;
      if (tensor->buffer.defined() && !arg_names.count(tensor->buffer->name)) {
        bool is_output =
            teller.count(tensor->name) && teller.count(tensor->name);
        if (is_output)
          out_args.emplace_back(tensor->buffer, ir::Argument::IO::kOutput);
      }
    }
  }

  std::vector<ir::Argument> args(in_args.begin(), in_args.end());
  args.insert(std::end(args), out_args.begin(), out_args.end());
  return args;
}

std::vector<Tensor> LowerImpl::CollectTemporaryTensors() {
  // a temporary should be in the comp_graph but not contained in the
  // tensor_args.
  paddle::flat_hash_map<std::string, Tensor> tensor_arg_map = GenTensorArgMap();
  paddle::flat_hash_map<std::string, Tensor> temp_tensor_map;

  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    PADDLE_ENFORCE_NOT_NULL(
        cnode,
        ::common::errors::InvalidArgument(
            "Node could not be safely cast to CompuGraphNode."));
    if (!tensor_arg_map.count(cnode->tensor->name)) {
      temp_tensor_map[cnode->tensor->name] = cnode->tensor;
    }
  }

  std::vector<Tensor> temp_tensors;
  std::transform(
      temp_tensor_map.begin(),
      temp_tensor_map.end(),
      std::back_inserter(temp_tensors),
      [&](const decltype(temp_tensor_map)::value_type& x) { return x.second; });
  return temp_tensors;
}

paddle::flat_hash_map<std::string, Tensor> LowerImpl::GenTensorArgMap() {
  paddle::flat_hash_map<std::string, Tensor> map;
  for (auto& t : tensor_args_) {
    map[t->name] = t;
  }
  return map;
}

paddle::flat_hash_map<std::string, Tensor> LowerImpl::GenAllTensorMap() {
  paddle::flat_hash_map<std::string, Tensor> map;
  for (auto& t : CollectAllTensors()) {
    map[t->name] = t;
  }
  return map;
}

std::vector<Tensor> LowerImpl::CollectAllTensors() {
  std::vector<Tensor> tensors;
  auto topo_order = compu_graph_->topological_order();  // NOLINT
  auto& nodes = std::get<0>(topo_order);
  auto& edges = std::get<1>(topo_order);
  for (auto* node : nodes) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    PADDLE_ENFORCE_NOT_NULL(
        cnode,
        ::common::errors::InvalidArgument(
            "Node could not be safely cast to CompuGraphNode."));
    tensors.push_back(cnode->tensor);
  }
  return tensors;
}

std::set<std::pair<std::string, std::string>>
LowerImpl::CollectExtraDependencies() const {
  std::set<std::pair<std::string, std::string>> deps;
  for (auto* node : compu_graph_->nodes()) {
    auto* cnode = node->safe_as<CompuGraphNode>();
    PADDLE_ENFORCE_NOT_NULL(
        cnode,
        ::common::errors::InvalidArgument(
            "Node could not be safely cast to CompuGraphNode."));
  }
  return deps;
}

}  // namespace detail
}  // namespace lang
}  // namespace cinn
