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
        auto ptr = (*it)->operand(i);

        PADDLE_ENFORCE_EQ(name_map->count(ptr),
                          true,
                          phi::errors::PreconditionNotMet(
                              "can not found input of combine op"));

        tensor_array->emplace_back(
            &(scope->Var(name_map->at(ptr))->Get<phi::DenseTensor>()));
      }

      continue;
    }

    // TODO(zhangbo): support builtin.slice

    if (input_num > 0) {
      for (size_t i = 0; i < input_num; ++i) {
        auto ptr = (*it)->operand(i);
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
        // Only support DenseTensor or Vector<DenseTensor>
        if (ptr.type().isa<paddle::dialect::AllocatedDenseTensorType>()) {
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
            auto var_i = scope->Var(name_i);
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
