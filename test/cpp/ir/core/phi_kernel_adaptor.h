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
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
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
  std::unordered_map<ir::Value, int> map_test;

  int count = 0;
  for (auto it = block->begin(); it != block->end(); ++it) {
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
        }
        auto var = scope->Var(name);
        // need to update here, only support DenseTensor
        var->GetMutable<phi::DenseTensor>();
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
        }
        auto var = scope->Var(name);

        var->GetMutable<phi::DenseTensor>();
      }
    }
  }
}

template <typename T>
void build_context(ir::Operation* op,
                   const std::unordered_map<ir::Value, std::string>& name_map,
                   paddle::framework::Scope* scope,
                   T* ctx,
                   bool is_infer_meta = true) {
  paddle::dialect::OpYamlInfoInterface op_info_interface =
      op->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
  auto op_info_res = op_info_interface.GetOpInfo();

  // inputs include input and mutable attributes
  auto input_info = std::get<0>(op_info_res);
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

  auto attr_info = std::get<1>(op_info_res);
  std::map<std::string, std::string> attr_type_map;
  for (auto& t : attr_info) {
    VLOG(6) << t.name << "\t" << t.type_name;
    attr_type_map[t.name] = t.type_name;
  }

  auto attr_map = op->attributes();
  auto runtime_info = std::get<3>(op_info_res);

  // int input_index = 0;
  std::vector<std::string> vec_param_list;
  if (is_infer_meta) {
    vec_param_list = runtime_info.infer_meta_param;
  } else {
    vec_param_list = runtime_info.kernel_param;
  }
  for (auto& t : vec_param_list) {
    if (input_index_map.count(t)) {
      // get information from input
      ir::Value ptr = op->GetOperandByIndex(input_index_map[t]).source();
      auto in_var_name = name_map.at(ptr);

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
        ctx->EmplaceBackInput(
            scope->Var(in_var_name)->GetMutable<phi::DenseTensor>());
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
      } else if (type_name == "ir::Int32_tAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<ir::Int32_tAttribute>().data());
      } else if (type_name == "paddle::dialect::PlaceAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
      } else if (type_name == "paddle::dialect::ScalarAttribute") {
        ctx->EmplaceBackAttr(paddle::dialect::TransToPhiScalar(attr_map[t]));
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                type_name));
      }
      VLOG(6) << "ctx->EmplaceBackAttr: " << t;
    }
  }

  ir::Value out_ptr = op->GetResultByIndex(0);
  auto name = name_map.at(out_ptr);

  ctx->EmplaceBackOutput(scope->Var(name)->GetMutable<phi::DenseTensor>());
}

class PhiKernelAdaptor {
 public:
  explicit PhiKernelAdaptor(paddle::framework::Scope* scope) : scope_(scope) {}

  void run(ir::Program* program) {
    auto block = program->block();
    std::unordered_map<ir::Value, std::string> name_map;
    build_scope(block, scope_, &name_map);

    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      VLOG(6) << "begin to run op " << (*it)->name();

      auto attr_map = (*it)->attributes();

      InferShapeInterface interface = (*it)->dyn_cast<InferShapeInterface>();
      phi::InferMetaContext ctx;

      build_context<phi::InferMetaContext>((*it), name_map, scope_, &ctx);

      interface.InferShape(&ctx);

      paddle::dialect::OpYamlInfoInterface op_info_interface =
          (*it)->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
      auto op_info_res = op_info_interface.GetOpInfo();

      auto runtime_info = std::get<3>(op_info_res);

      auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
          runtime_info.kernel_func[0]);

      phi::KernelKey kernel_key(phi::TransToPhiBackend(cpu_place),
                                phi::DataLayout::ANY,
                                phi::DataType::FLOAT32);
      if (runtime_info.kernel_func[0] == "full_int_array") {
        kernel_key.set_dtype(phi::DataType::INT64);
      }
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

        build_context<phi::KernelContext>(
            (*it), name_map, scope_, &kernel_ctx, false);
        found_it->second(&kernel_ctx);

        auto out_value = (*it)->GetResultByIndex(0);
        out_name = name_map[out_value];
      }
    }
  }

  std::string out_name;

 private:
  paddle::framework::Scope* scope_;
};
