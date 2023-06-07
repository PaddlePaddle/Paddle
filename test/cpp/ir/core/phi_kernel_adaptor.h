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

#pragma once

#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_interface.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/ir/core/builtin_attribute.h"
#include "paddle/ir/core/builtin_dialect.h"
#include "paddle/ir/core/builtin_op.h"
#include "paddle/ir/core/ir_context.h"
#include "paddle/ir/core/program.h"
#include "paddle/ir/core/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/ir/dialect/pd_attribute.h"

#include "glog/logging.h"

void build_scope(ir::Block* block,
                 paddle::framework::Scope* scope,
                 std::unordered_map<ir::Value, std::string>* name_map) {
  int count = scope->Size();
  for (auto it = block->begin(); it != block->end(); ++it) {
    if ((*it)->name() == "builtin.get_parameter") {
      auto attr_map1 = (*it)->attributes();
      // for( auto it2 = attr_map1.begin(); it2 != attr_map1.end(); ++it2 )
      // {
      //   std::cerr << it2->first << "\t" <<
      //   it2->second.dyn_cast<ir::StrAttribute>().data() << std::endl;
      // }

      ir::Value out = (*it)->GetResultByIndex(0);
      auto name = (*it)
                      ->attributes()
                      .at("parameter_name")
                      .dyn_cast<ir::StrAttribute>()
                      .data();

      (*name_map)[out] = name;

      // auto var = scope->Var(name);
      // // need to update here, only support DenseTensor
      // var->GetMutable<phi::DenseTensor>();
      continue;
    }

    if ((*it)->name() == "builtin.set_parameter") {
      auto attr_map1 = (*it)->attributes();
      // for( auto it2 = attr_map1.begin(); it2 != attr_map1.end(); ++it2 )
      // {
      //   std::cerr << it2->first << "\t" <<
      //   it2->second.dyn_cast<ir::StrAttribute>().data() << std::endl;
      // }

      ir::Value in_value = (*it)->GetOperandByIndex(0).source();
      auto new_name = (*it)
                          ->attributes()
                          .at("parameter_name")
                          .dyn_cast<ir::StrAttribute>()
                          .data();

      auto var_name = name_map->at(in_value);

      // toto, it's danger here
      std::cerr << "change name " << var_name << "\t" << new_name << std::endl;
      scope->Rename(var_name, new_name);
      (*name_map)[in_value] = new_name;
      // need to update here, only support DenseTensor
      // var->GetMutable<phi::DenseTensor>();

      continue;
    }

    if ((*it)->name() == "pd.feed") {
      auto attr_map1 = (*it)->attributes();
      for (auto it2 = attr_map1.begin(); it2 != attr_map1.end(); ++it2) {
        std::cerr << "feed name " << it2->first << "\t"
                  << it2->second.dyn_cast<ir::StrAttribute>().data()
                  << std::endl;
      }

      ir::Value out = (*it)->GetResultByIndex(0);
      auto name = attr_map1.begin()->second.dyn_cast<ir::StrAttribute>().data();

      (*name_map)[out] = name;

      auto var = scope->Var(name);
      // need to update here, only support DenseTensor
      var->GetMutable<phi::DenseTensor>();

      continue;
    }

    std::cerr << "build " << (*it)->name() << std::endl;

    int input = (*it)->num_operands();
    if (input > 0) {
      for (int i = 0; i < input; ++i) {
        auto ptr = (*it)->GetOperandByIndex(i).source();
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          name = "var_" + std::to_string(count++);
          name_map->emplace(ptr, name);
          auto var = scope->Var(name);
          // need to update here, only support DenseTensor
          var->GetMutable<phi::DenseTensor>();
        }
      }
    }

    int out_num = (*it)->num_results();

    if (out_num > 0) {
      for (int i = 0; i < out_num; ++i) {
        ir::Value ptr = (*it)->GetResultByIndex(i);
        std::string name;
        if (name_map->find(ptr) != name_map->end()) {
          name = name_map->at(ptr);
        } else {
          name = "var_" + std::to_string(count++);
          name_map->emplace(ptr, name);
          auto var = scope->Var(name);

          var->GetMutable<phi::DenseTensor>();
        }
      }
    }
  }
}

template <typename T>
void build_context(
    ir::Operation* op,
    const std::unordered_map<ir::Value, std::string>& name_map,
    paddle::framework::Scope* scope,
    T* ctx,
    const std::unordered_map<ir::Value, std::vector<std::string>>&
        map_combine_cache,
    bool is_infer_meta = true) {
  paddle::dialect::GetOpInfoInterface op_info_interface =
      op->dyn_cast<paddle::dialect::GetOpInfoInterface>();
  auto op_info_res = op_info_interface.GetOpInfo();

  auto input_info = std::get<0>(op_info_res);

  std::set<std::string> input_set;
  for (auto& t : input_info) {
    VLOG(6) << t.name << "\t" << t.type_name;

    input_set.insert(t.name);
  }

  auto attr_map = op->attributes();

  std::map<std::string, std::string> attr_type_map;
  auto attr_info = std::get<1>(op_info_res);
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }

  auto runtime_info = std::get<3>(op_info_res);

  int input_index = 0;
  std::vector<std::string> vec_param_list;
  if (is_infer_meta) {
    vec_param_list = runtime_info.infer_meta_param;
  } else {
    vec_param_list = runtime_info.kernel_param;
  }

  if (op->name() == "pd.add_n") {
    std::cerr << "deal with add n" << std::endl;

    auto in_list = map_combine_cache.at(op->GetOperandByIndex(0).source());
    std::cerr << "in num " << in_list.size() << std::endl;
    for (auto& name : in_list) {
      ctx->EmplaceBackInput(scope->Var(name)->GetMutable<phi::DenseTensor>());
    }

    // tricky here
    vec_param_list.clear();
  }

  for (auto& t : vec_param_list) {
    if (input_set.count(t)) {
      // get information from input
      ir::Value ptr = op->GetOperandByIndex(input_index++).source();
      auto in_var_name = name_map.at(ptr);

      // need to check other input type
      ctx->EmplaceBackInput(
          scope->Var(in_var_name)->GetMutable<phi::DenseTensor>());
    }

    if (attr_type_map.count(t)) {
      auto type_name = attr_type_map[t];
      if (type_name == "paddle::dialect::IntArrayAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::IntArrayAttribute>().data());
      } else if (type_name == "paddle::dialect::DataTypeAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::DataTypeAttribute>().data());
      } else if (type_name == "paddle::dialect::ScalarAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::ScalarAttribute>().data());
      } else if (type_name == "ir::Int32_tAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<ir::Int32_tAttribute>().data());
      } else if (type_name == "ir::FloatAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::FloatAttribute>().data());
      } else if (type_name == "ir::StrAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::StrAttribute>().data());
      } else if (type_name == "ir::BoolAttribute") {
        ctx->EmplaceBackAttr(attr_map[t].dyn_cast<ir::BoolAttribute>().data());
      } else if (type_name == "paddle::dialect::PlaceAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
      } else if (type_name == "ir::ArrayAttribute<ir::Int32_tAttribute>") {
        std::vector<int32_t> res;
        auto vec_attr = attr_map[t].dyn_cast<ir::ArrayAttribute>().data();
        res.reserve(vec_attr.size());
        for (size_t i = 0; i < vec_attr.size(); ++i) {
          res.emplace_back(vec_attr[i].dyn_cast<ir::Int32_tAttribute>().data());
        }
        ctx->EmplaceBackAttr(res);
      } else {
        std::cerr << op->name() << "\t" << type_name << std::endl;
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                type_name));
      }
    }
  }

  // need build all ouoput
  auto output_info = std::get<2>(op_info_res);
  for (size_t i = 0; i < output_info.size(); ++i) {
    ir::Value out_ptr = op->GetResultByIndex(i);
    auto name = name_map.at(out_ptr);

    ctx->EmplaceBackOutput(scope->Var(name)->GetMutable<phi::DenseTensor>());
  }
}

class PhiKernelAdaptor {
 public:
  explicit PhiKernelAdaptor(paddle::framework::Scope* scope) : scope_(scope) {}

  void run(ir::Program* program) {
    std::cerr << "run " << std::endl;
    auto block = program->block();
    std::unordered_map<ir::Value, std::string> name_map;
    build_scope(block, scope_, &name_map);
    std::cerr << "fin build scope" << std::endl;

    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      if ((*it)->name() == "builtin.get_parameter" ||
          (*it)->name() == "pd.feed" ||
          (*it)->name() == "builtin.set_parameter") {
        // auto attr_map1 = (*it)->attributes();
        // for( auto it2 = attr_map1.begin(); it2 != attr_map1.end(); ++it2 )
        // {
        //   std::cerr << it2->first << "\t" <<
        //   it2->second.dyn_cast<ir::StrAttribute>().data() << std::endl;
        // }
        continue;
      }

      if ((*it)->name() == "builtin.combine") {
        size_t input_num = (*it)->num_operands();
        std::cerr << "deal combine " << input_num << std::endl;
        for (size_t i = 0; i < input_num; ++i) {
          map_combine_cache_[(*it)->GetResultByIndex(0)].push_back(
              name_map.at((*it)->GetOperandByIndex(i).source()));
        }
        continue;
      }
      std::cerr << "begin to run op " << (*it)->name() << std::endl;
      VLOG(6) << "begin to run op " << (*it)->name();

      auto attr_map = (*it)->attributes();

      InferShapeInterface interface = (*it)->dyn_cast<InferShapeInterface>();
      phi::InferMetaContext ctx;

      std::cerr << "begin to build context" << std::endl;
      build_context<phi::InferMetaContext>(
          (*it), name_map, scope_, &ctx, map_combine_cache_);
      std::cerr << "fin to build context " << std::endl;

      if (interface) {
        std::cerr << "begin to run infer shape" << std::endl;
        interface.InferShape(&ctx);
        std::cerr << "fin to run infer shape" << std::endl;
      } else {
        std::cerr << "skip infer shape " << (*it)->name() << std::endl;
      }

      paddle::dialect::GetOpInfoInterface op_info_interface =
          (*it)->dyn_cast<paddle::dialect::GetOpInfoInterface>();
      auto op_info_res = op_info_interface.GetOpInfo();

      auto runtime_info = std::get<3>(op_info_res);

      std::string phi_kernel_name = runtime_info.kernel_func[0];

      if ((*it)->name() == "pd.add_n") {
        phi_kernel_name = "add";
      }
      auto phi_kernels =
          phi::KernelFactory::Instance().SelectKernelMap(phi_kernel_name);

      phi::KernelKey kernel_key(phi::TransToPhiBackend(cpu_place),
                                phi::DataLayout::ANY,
                                phi::DataType::FLOAT32);
      auto found_it = phi_kernels.find(kernel_key);
      if (found_it == phi_kernels.end()) {
        std::cerr << "kernel name " << runtime_info.kernel_func[0] << std::endl;
        std::cerr << "kernel key " << kernel_key.backend() << "\t"
                  << kernel_key.dtype() << "\t" << kernel_key.layout()
                  << std::endl;
        PADDLE_THROW(paddle::platform::errors::NotFound(
            "can not found kerenl for [%s]", (*it)->name()));
      } else {
        phi::KernelContext kernel_ctx(dev_ctx);

        std::cerr << "begin to build kernel context" << std::endl;
        build_context<phi::KernelContext>(
            (*it), name_map, scope_, &kernel_ctx, map_combine_cache_, false);
        std::cerr << "fin to build kernel context " << std::endl;
        found_it->second(&kernel_ctx);

        auto out_value = (*it)->GetResultByIndex(0);
        std::cerr << "get name " << std::endl;
        out_name = name_map[out_value];
        std::cerr << "fin get name " << std::endl;
      }
    }
  }

  std::string out_name;

 private:
  paddle::framework::Scope* scope_;

  std::unordered_map<ir::Value, std::vector<std::string>> map_combine_cache_;
};
