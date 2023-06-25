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
#include "paddle/fluid/ir/dialect/pd_attribute.h"
#include "paddle/phi/core/enforce.h"

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
        auto var = scope->Var("fetch");
        auto fetch_list = var->GetMutable<paddle::framework::FetchList>();
        // for now only support one fetch
        fetch_list->resize(1);
      }
      continue;
    }

    if (op_name == "pd.feed") {
      auto ptr = (*it)->result(0);
      std::string name = "inner_var_" + std::to_string(count++);
      name_map->emplace(ptr, name);
      auto var = scope->Var(name);
      // TODO(phlrain): need to update here, support StringTensor
      auto out_tensor = var->GetMutable<phi::DenseTensor>();

      name_map->emplace(ptr, name);

      auto feed_var = scope->Var("feed");
      int index =
          (*it)->attributes().at("col").dyn_cast<ir::Int32Attribute>().data();
      auto feed_list = feed_var->Get<paddle::framework::FeedList>();
      auto& in_tensor = (PADDLE_GET(phi::DenseTensor, feed_list.at(index)));

      out_tensor->ShareDataWith(in_tensor);

      continue;
    }

    if (op_name == "builtin.combine") {
      auto out_value = (*it)->result(0);

      VLOG(5) << "process builtin combine";
      std::string name;
      if (name_map->find(out_value) != name_map->end()) {
        name = name_map->at(out_value);
      } else {
        name = "inner_var_" + std::to_string(count++);
        name_map->emplace(out_value, name);
      }

      auto var = scope->Var(name);
      auto tensor_array = var->GetMutable<paddle::framework::TensorRefArray>();

      for (size_t i = 0; i < input_num; ++i) {
        auto ptr = (*it)->operand(i).source();

        PADDLE_ENFORCE_EQ(name_map->count(ptr),
                          true,
                          phi::errors::PreconditionNotMet(
                              "can not found input of combine op"));

        tensor_array->emplace_back(
            &(scope->Var(name_map->at(ptr))->Get<phi::DenseTensor>()));
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

  for (size_t input_index = 0; input_index < vec_param_list.size();
       input_index++) {
    auto& t = vec_param_list[input_index];
    if (input_index_map.count(t)) {
      // get information from input
      ir::Value ptr = op->operand(input_index_map[t]).source();
      auto in_var_name = name_map.at(ptr);

      if (mutable_attr_type_map.count(t)) {
        VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t"
                << in_var_name;
        if (mutable_attr_type_map[t] == "paddle::dialect::IntArrayAttribute") {
          phi::Attribute r1 = phi::TensorRef(
              &(scope->Var(in_var_name)->Get<phi::DenseTensor>()));
          ctx->EmplaceBackAttr(r1);
        } else if (mutable_attr_type_map[t] ==
                   "paddle::dialect::ScalarAttribute") {
          phi::Attribute r1 = phi::TensorRef(
              &(scope->Var(in_var_name)->Get<phi::DenseTensor>()));

          ctx->EmplaceBackAttr(r1);
        } else {
          PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                  mutable_attr_type_map[t]));
        }

      } else {
        VLOG(6) << "ctx->EmplaceBackInput: " << t << "\t" << in_var_name;
        auto var = scope->Var(in_var_name);
        if (var->IsType<phi::DenseTensor>()) {
          const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
          ctx->EmplaceBackInput(const_cast<phi::TensorBase*>(tensor_in));
        } else if (var->IsType<paddle::framework::TensorRefArray>()) {
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>
              inputs;
          auto& tensor_array = var->Get<paddle::framework::TensorRefArray>();
          for (size_t i = 0; i < tensor_array.size(); ++i) {
            inputs.emplace_back(std::move(phi::MetaTensor(*tensor_array[i])));
          }

          ctx->EmplaceBackInputs(std::move(inputs));
        } else {
          PADDLE_THROW(phi::errors::Unimplemented("Not support var type [%d] ",
                                                  var->Type()));
        }
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
      } else if (type_name == "ir::FloatAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::FloatAttribute>().data());
      } else if (type_name == "ir::BoolAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::BoolAttribute>().data());
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

  // TODO(phlrain): use var type instead of op name
  if (op->attributes().count("op_name") &&
      (op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data() ==
       "pd.fetch")) {
    // process fetch op
    auto fetch_var = scope->Var("fetch");
    auto* fetch_list = fetch_var->GetMutable<paddle::framework::FetchList>();
    auto* out_tensor = &(PADDLE_GET(phi::DenseTensor, fetch_list->at(0)));
    ctx->EmplaceBackOutput(out_tensor);
  } else {
    for (size_t i = 0; i < op->num_results(); ++i) {
      ir::Value out_ptr = op->result(i);
      auto name = name_map.at(out_ptr);
      ctx->EmplaceBackOutput(scope->Var(name)->Get<phi::DenseTensor>());
    }
  }
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
        // TODO(phlrain): OpFuncNode need input_index and output_index,
        // construct input_index and output_here,  should remove input_index and
        // output_index from OpFuncNode Each in_var_name named "inner_var_" +
        // index, len("inner_var_") = 10
        std::cerr << "in var name " << in_var_name << std::endl;
        size_t tmp_id = std::atol(in_var_name.substr(10, 100).c_str());
        (*input_map)[std::to_string(input_index_map.at(t))].push_back(tmp_id);
      }

      if (mutable_attr_type_map.count(t)) {
        VLOG(6) << "ctx->EmplaceBack mutable attr: " << t << "\t"
                << in_var_name;
        if (mutable_attr_type_map[t] == "paddle::dialect::IntArrayAttribute") {
          phi::Attribute r1 = phi::TensorRef(
              &(scope->Var(in_var_name)->Get<phi::DenseTensor>()));
          ctx->EmplaceBackAttr(r1);
        } else if (mutable_attr_type_map[t] ==
                   "paddle::dialect::ScalarAttribute") {
          phi::Attribute r1 = phi::TensorRef(
              &(scope->Var(in_var_name)->Get<phi::DenseTensor>()));

          ctx->EmplaceBackAttr(r1);
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
        if (var->IsType<phi::DenseTensor>()) {
          const phi::TensorBase* tensor_in = &(var->Get<phi::DenseTensor>());
          ctx->EmplaceBackInput(tensor_in);
        } else if (var->IsType<paddle::framework::TensorRefArray>()) {
          paddle::small_vector<const phi::TensorBase*> inputs;
          auto& tensor_array = var->Get<paddle::framework::TensorRefArray>();
          for (size_t i = 0; i < tensor_array.size(); ++i) {
            inputs.emplace_back(tensor_array[i]);
          }

          ctx->EmplaceBackInputs(std::move(inputs));
        } else if (var->IsType<paddle::framework::FeedList>()) {
          auto feed_list = var->Get<paddle::framework::FeedList>();
          auto* in_tensor = &(PADDLE_GET(phi::DenseTensor, feed_list.at(0)));
          ctx->EmplaceBackOutput(in_tensor);
        } else {
          PADDLE_THROW(phi::errors::Unimplemented("Not support var type [%d] ",
                                                  var->Type()));
        }
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
      } else if (type_name == "ir::FloatAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::FloatAttribute>().data());
      } else if (type_name == "ir::BoolAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::BoolAttribute>().data());
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

  // TODO(phlrain): use var type instead of op name
  if (op->attributes().count("op_name") &&
      (op->attributes().at("op_name").dyn_cast<ir::StrAttribute>().data() ==
       "pd.fetch")) {
    // process fetch op
    auto fetch_var = scope->Var("fetch");
    auto* fetch_list = fetch_var->GetMutable<paddle::framework::FetchList>();
    auto* out_tensor = &(PADDLE_GET(phi::DenseTensor, fetch_list->at(0)));
    ctx->EmplaceBackOutput(out_tensor);
  } else {
    for (size_t i = 0; i < op->num_results(); ++i) {
      ir::Value out_ptr = op->result(i);
      auto name = name_map.at(out_ptr);
      ctx->EmplaceBackOutput(const_cast<phi::DenseTensor*>(
          &(scope->Var(name)->Get<phi::DenseTensor>())));

      if (output_map != nullptr) {
        // only deal with single input for now, [todo] need support multi input
        // like concat
        // TODO(phlrain): OpFuncNode need input_index and output_index,
        // construct input_index and output_here,  should remove input_index and
        // output_index from OpFuncNode Each in_var_name named "inner_var_" +
        // index, len("inner_var_") = 10
        std::cerr << "out var name " << name << std::endl;
        size_t tmp_id = std::atol(name.substr(10, 100).c_str());
        (*output_map)["out"].push_back(tmp_id);
      }
    }
  }
}

}  // namespace ir
