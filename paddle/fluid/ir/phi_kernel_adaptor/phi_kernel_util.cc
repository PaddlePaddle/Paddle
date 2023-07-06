// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"

#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/phi/core/kernel_context.h"

#include "paddle/fluid/framework/string_array.h"
#include "paddle/fluid/framework/tensor_ref_array.h"
#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/kernel_type.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
#include "paddle/phi/core/enforce.h"

#include "glog/logging.h"

namespace ir {

paddle::framework::Variable* CreateVar(ir::Value value,
                                       std::string name,
                                       paddle::framework::Scope* scope,
                                       paddle::framework::Scope* local_scope) {
  Operation* def_op = value.GetDefiningOp();
  bool is_persisable = false;
  if (def_op->attributes().count("is_persisable")) {
    is_persisable = def_op->attributes()
                        .at("is_persisable")
                        .dyn_cast<ir::BoolAttribute>()
                        .data();
  }
  if (is_persisable) {
    const paddle::framework::Scope* ancestor_scope = scope;
    while (ancestor_scope->parent()) {
      ancestor_scope = ancestor_scope->parent();
    }
    VLOG(6) << "Create var: " << name << " in scope " << ancestor_scope;
    return const_cast<paddle::framework::Scope*>(ancestor_scope)->Var(name);
  } else {
    VLOG(6) << "Create var: " << name << " in scope " << local_scope;
    return local_scope->Var(name);
  }
}

void HandleForSpecialOp(ir::Operation* op,
                        paddle::framework::Scope* scope,
                        paddle::framework::Scope* local_scope,
                        std::unordered_map<ir::Value, std::string>* name_map,
                        int& count) {  // NOLINT
  std::string op_name = op->name();
  if (op->attributes().count("op_name")) {
    op_name =
        op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data();
  }
  size_t input_num = op->num_operands();

  if (op_name == "pd.fetch") {
    // fetch is a very special op, with no output
    VLOG(6) << "Handle for pd.fetch:";
    for (size_t i = 0; i < input_num; ++i) {
      auto var = scope->Var("fetch");
      VLOG(6) << "Create var: fetch in scope " << scope;
      auto fetch_list = var->GetMutable<paddle::framework::FetchList>();
      int index =
          op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
      fetch_list->resize(index + 1);
    }
  }

  if (op_name == "pd.feed") {
    VLOG(6) << "Handle for pd.feed:";
    auto ptr = op->result(0);
    std::string name = "inner_var_" + std::to_string(count++);
    name_map->emplace(ptr, name);
    auto var = CreateVar(ptr, name, scope, local_scope);
    // TODO(phlrain): need to update here, support StringTensor
    auto out_tensor = var->GetMutable<phi::DenseTensor>();

    auto feed_var = scope->Var("feed");
    VLOG(6) << "Create var: feed in scope " << scope;
    int index =
        op->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
    auto feed_list = feed_var->Get<paddle::framework::FeedList>();
    auto& in_tensor = (PADDLE_GET(phi::DenseTensor, feed_list.at(index)));
    std::cerr << "~~~~~~~~~~~~~~~~~~~~~~~~ share here  " << in_tensor.dims()
              << std::endl;
    out_tensor->ShareDataWith(in_tensor);
  }

  if (op_name == "builtin.combine") {
    VLOG(6) << "Handle for builtin.combine:";
    auto out_value = op->result(0);
    std::string name;
    if (name_map->find(out_value) != name_map->end()) {
      name = name_map->at(out_value);
    } else {
      name = "inner_var_" + std::to_string(count++);
      name_map->emplace(out_value, name);
    }

    auto var = CreateVar(out_value, name, scope, local_scope);
    auto tensor_array = var->GetMutable<paddle::framework::TensorRefArray>();
    // clear tensor array
    tensor_array->clear();

    for (size_t i = 0; i < input_num; ++i) {
      auto ptr = op->operand(i);

      PADDLE_ENFORCE_EQ(
          name_map->count(ptr),
          true,
          phi::errors::PreconditionNotMet("can not found input of combine op"));
      tensor_array->emplace_back(
          &(CreateVar(ptr, name_map->at(ptr), scope, local_scope)
                ->Get<phi::DenseTensor>()));
    }
  }

  if (op_name == "builtin.set_parameter") {
    VLOG(6) << "Handle for builtin.set_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();

    auto in_ptr = op->operand(0);
    // change opreand name to param_name

    auto orig_name = name_map->at(in_ptr);
    if (scope->FindVar(param_name) == nullptr) {
      (*name_map)[in_ptr] = param_name;
      scope->Rename(orig_name, param_name);
    }
  }

  if (op_name == "builtin.get_parameter") {
    VLOG(6) << "Handle for builtin.get_parameter:";
    auto param_name = op->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();
    auto out_ptr = op->result(0);
    name_map->emplace(out_ptr, param_name);
  }
}

void BuildScope(ir::Block* block,
                paddle::framework::Scope* scope,
                paddle::framework::Scope* local_scope,
                std::unordered_map<ir::Value, std::string>* name_map) {
  // NOTE(zhiqiu): if use local_scope (local_scope != nullptr), the persistable
  // is created in scope , and other is created in local_scope.
  auto inner_local_scope = local_scope != nullptr ? local_scope : scope;
  VLOG(6) << "Build: scope [" << scope << "] inner_local_scope ["
          << inner_local_scope << "]";

  // int count = name_map->size();
  int count = name_map->size();
  for (auto it = block->begin(); it != block->end(); ++it) {
    ir::Operation* op = *it;

    auto attr_map = op->attributes();
    std::string op_name = op->name();
    if (attr_map.count("op_name")) {
      op_name = attr_map.at("op_name").dyn_cast<ir::StrAttribute>().data();
    }

    if (op_name == "pd.feed" || op_name == "pd.fetch" ||
        op_name == "builtin.combine" || op_name == "builtin.set_parameter" ||
        op_name == "builtin.get_parameter") {
      VLOG(6) << "HandleForSpecialOp: " << op_name;
      HandleForSpecialOp(op, scope, inner_local_scope, name_map, count);
      continue;
    }

    size_t input_num = op->num_operands();
    if (input_num > 0) {
      for (size_t i = 0; i < input_num; ++i) {
        auto ptr = op->operand(i);
        if (ptr) {
          PADDLE_ENFORCE_NE(
              name_map->find(ptr),
              name_map->end(),
              phi::errors::PreconditionNotMet(
                  "input should in name map, [%d] 'th input of [%s] op",
                  i,
                  op_name));
        }
      }
    }

    int out_num = op->num_results();
    if (out_num > 0) {
      for (int i = 0; i < out_num; ++i) {
        ir::Value ptr = op->result(i);
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          name = "inner_var_" + std::to_string(count++);
          name_map->emplace(ptr, name);
        }
        auto var = CreateVar(ptr, name, scope, inner_local_scope);
        // Only support DenseTensor or Vector<DenseTensor>
        if (!ptr.type()) {
          var->GetMutable<phi::DenseTensor>();
        } else if (ptr.type()
                       .isa<paddle::dialect::AllocatedDenseTensorType>()) {
          var->GetMutable<phi::DenseTensor>();
        } else if (ptr.type().isa<ir::VectorType>()) {
          auto tensor_array =
              var->GetMutable<paddle::framework::TensorRefArray>();
          for (size_t i = 0; i < ptr.type().dyn_cast<ir::VectorType>().size();
               i++) {
            PADDLE_ENFORCE(
                ptr.type()
                    .dyn_cast<ir::VectorType>()[i]
                    .isa<paddle::dialect::AllocatedDenseTensorType>(),
                paddle::platform::errors::Fatal(
                    "Element of VectorType output only support "
                    "DenseTensorType"));
            std::string name_i = "inner_var_" + std::to_string(count++);
            auto var_i = CreateVar(ptr, name_i, scope, inner_local_scope);
            tensor_array->emplace_back(var_i->GetMutable<phi::DenseTensor>());
          }
        } else {
          PADDLE_THROW(phi::errors::PreconditionNotMet(
              "Output only support DenseTensorType or VectorType"));
        }
      }
    }
  }
}

}  // namespace ir
