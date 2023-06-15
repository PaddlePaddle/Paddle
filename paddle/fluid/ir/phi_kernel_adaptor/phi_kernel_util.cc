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

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"

#include "glog/logging.h"

namespace ir {

void BuildScope(ir::Block* block,
                paddle::framework::Scope* scope,
                std::unordered_map<ir::Value, std::string>* name_map) {
  std::unordered_map<ir::Value, int> map_test;

  // int count = name_map->size();
  int count = 0;
  for (auto it = block->begin(); it != block->end(); ++it) {
    size_t input_num = (*it)->num_operands();
    auto attr_map = (*it)->attributes();
    std::string op_name = (*it)->name();
    if (attr_map.count("op_name")) {
      op_name = attr_map.at("op_name").dyn_cast<ir::StrAttribute>().data();
    }
    if (op_name == "pd.fetch") {
      // fetch is a very special op, with no output
      for (size_t i = 0; i < input_num; ++i) {
        auto ptr = (*it)->operand(i).source();
        auto var_name = attr_map.at("name").dyn_cast<ir::StrAttribute>().data();

        PADDLE_ENFORCE_EQ(
            name_map->count(ptr),
            true,
            phi::errors::PreconditionNotMet(
                "input of fetch op should in name mape, var_name is [%s]",
                var_name));

        scope->Rename(name_map->at(ptr), var_name);
        (*name_map)[ptr] = var_name;
      }
      continue;
    }

    if (input_num > 0) {
      for (size_t i = 0; i < input_num; ++i) {
        auto ptr = (*it)->operand(i).source();
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          PADDLE_THROW(phi::errors::PreconditionNotMet(
              "input should in name map, [%d] 'th input of [%s] op",
              i,
              op_name));
        }
      }
    }

    int out_num = (*it)->num_results();

    if (out_num > 0) {
      for (int i = 0; i < out_num; ++i) {
        ir::Value ptr = (*it)->result(i);
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          name = "inner_var_" + std::to_string(count++);
          name_map->emplace(ptr, name);
        }
        auto var = scope->Var(name);

        // need to update here, only support DenseTensor
        var->GetMutable<phi::DenseTensor>();
      }
    }
  }
}

void BuildInferMetaContext(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    const OpInfoTuple& op_yaml_info,
    phi::InferMetaContext* ctx) {
  // inputs include input and mutable attributes
  auto input_info = std::get<0>(op_yaml_info);
  std::map<std::string, size_t> input_index_map;
  std::map<std::string, std::string> mutable_attr_type_map;
  int input_index = 0;
  for (auto& t : input_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    input_index_map[t.name] = input_index++;
    if (t.is_mutable_attribute) {
      mutable_attr_type_map[t.name] = t.type_name;
    }
  }

  auto attr_info = std::get<1>(op_yaml_info);
  std::map<std::string, std::string> attr_type_map;
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }

  auto attr_map = op->attributes();
  auto runtime_info = std::get<3>(op_yaml_info);

  // int input_index = 0;
  std::vector<std::string> vec_param_list = runtime_info.infer_meta_param;

  for (auto& t : vec_param_list) {
    if (input_index_map.count(t)) {
      // get information from input
      ir::Value ptr = op->operand(input_index_map[t]).source();
      auto in_var_name = name_map.at(ptr);

      if (mutable_attr_type_map.count(t)) {
        VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t"
                << in_var_name;
        if (mutable_attr_type_map[t] == "paddle::dialect::IntArrayAttribute") {
          std::cerr << "build int array" << std::endl;
          phi::Attribute r1 = phi::TensorRefScalar(nullptr);
          std::cerr << "r1 index " << r1.index() << std::endl;
          ctx->EmplaceBackAttr(r1);
          auto r2 = ctx->AttrAt(0);
          std::cerr << "get attr " << r2.index() << std::endl;
          auto t1 =
              phi::IntArray((scope->Var(in_var_name)->Get<phi::DenseTensor>()));
          std::cerr << "fin build int array" << std::endl;
          ctx->EmplaceBackAttr(t1);
        } else if (mutable_attr_type_map[t] ==
                   "paddle::dialect::ScalarAttribute") {
          std::cerr << "begin to build scope " << std::endl;
          auto t1 =
              phi::Scalar((scope->Var(in_var_name)->Get<phi::DenseTensor>()));
          ctx->EmplaceBackAttr(t1);
        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  mutable_attr_type_map[t]));
        }

      } else {
        VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;
        auto var = scope->Var(in_var_name);
        const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
        ctx->EmplaceBackInput(const_cast<phi::TensorBase*>(tensor_in));
      }
    }

    if (attr_type_map.count(t)) {
      auto type_name = attr_type_map[t];
      if (type_name == "paddle::dialect::IntArrayAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
      } else if (type_name == "paddle::dialect::DataTypeAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
      } else if (type_name == "ir::Int32Attribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::Int32Attribute>().data());
      } else if (type_name == "paddle::dialect::PlaceAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
      } else if (type_name == "paddle::dialect::ScalarAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                type_name));
      }
      VLOG(6) << "ctx->EmplaceBackAttr: " << t;
    }
  }

  ir::Value out_ptr = op->result(0);
  auto name = name_map.at(out_ptr);

  ctx->EmplaceBackOutput(scope->Var(name)->Get<phi::DenseTensor>());
}

void BuildPhiKernelContext(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    const OpInfoTuple& op_yaml_info,
    phi::KernelContext* ctx,
    std::map<std::string, std::vector<int>>* input_map,
    std::map<std::string, std::vector<int>>* output_map) {
  // inputs include input and mutable attributes
  auto input_info = std::get<0>(op_yaml_info);
  std::map<std::string, size_t> input_index_map;
  std::map<std::string, std::string> mutable_attr_type_map;
  int input_index = 0;
  for (auto& t : input_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    input_index_map[t.name] = input_index++;
    if (t.is_mutable_attribute) {
      mutable_attr_type_map[t.name] = t.type_name;
    }
  }

  auto attr_info = std::get<1>(op_yaml_info);
  std::map<std::string, std::string> attr_type_map;
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }

  auto attr_map = op->attributes();
  auto runtime_info = std::get<3>(op_yaml_info);

  // int input_index = 0;
  std::vector<std::string> vec_param_list = runtime_info.kernel_param;
  for (auto& t : vec_param_list) {
    if (input_index_map.count(t)) {
      // get information from input
      ir::Value ptr = op->operand(input_index_map[t]).source();
      auto in_var_name = name_map.at(ptr);

      if (input_map != nullptr) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        size_t tmp_id = std::atol(in_var_name.substr(4, 100).c_str());
        (*input_map)[std::to_string(input_index_map.at(t))].push_back(tmp_id);
      }

      if (mutable_attr_type_map.count(t)) {
        VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t"
                << in_var_name;
        if (mutable_attr_type_map[t] == "paddle::dialect::IntArrayAttribute") {
          ctx->EmplaceBackAttr(phi::IntArray(
              *(scope->Var(in_var_name)->GetMutable<phi::DenseTensor>())));
        } else if (mutable_attr_type_map[t] ==
                   "paddle::dialect::ScalarAttribute") {
          ctx->EmplaceBackAttr(phi::Scalar(
              *(scope->Var(in_var_name)->GetMutable<phi::DenseTensor>())));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  mutable_attr_type_map[t]));
        }

      } else {
        VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;

        PADDLE_ENFORCE_NOT_NULL(
            scope->FindLocalVar(in_var_name),
            phi::errors::PreconditionNotMet("can not find var[%s] in scope",
                                            in_var_name));

        auto var = scope->Var(in_var_name);
        const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
        ctx->EmplaceBackInput(tensor_in);
      }
    }

    if (attr_type_map.count(t)) {
      auto type_name = attr_type_map[t];
      if (type_name == "paddle::dialect::IntArrayAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
      } else if (type_name == "paddle::dialect::DataTypeAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
      } else if (type_name == "ir::Int32Attribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::Int32Attribute>().data());
      } else if (type_name == "paddle::dialect::PlaceAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
      } else if (type_name == "paddle::dialect::ScalarAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                type_name));
      }
      VLOG(6) << "ctx->EmplaceBackAttr: " << t;
    }
  }

  ir::Value out_ptr = op->result(0);
  auto name = name_map.at(out_ptr);

  ctx->EmplaceBackOutput(const_cast<phi::DenseTensor*>(
      &(scope->Var(name)->Get<phi::DenseTensor>())));

  if (output_map != nullptr) {
    // only deal with single input for now, [todo] need support multi input like
    // concat
    size_t tmp_id = std::atol(name.substr(4, 100).c_str());
    (*output_map)["out"].push_back(tmp_id);
  }
}

}  // namespace ir
