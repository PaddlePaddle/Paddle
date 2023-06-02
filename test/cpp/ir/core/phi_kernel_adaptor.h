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

#include "paddle/fluid/dialect/pd_dialect.h"
#include "paddle/fluid/dialect/pd_interface.h"
#include "paddle/fluid/dialect/pd_op.h"
#include "paddle/fluid/dialect/pd_type.h"
#include "paddle/fluid/dialect/utils.h"
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

#include "paddle/fluid/dialect/pd_attribute.h"

#include "glog/logging.h"

void build_scope(ir::Block* block,
                 paddle::framework::Scope* scope,
                 std::unordered_map<void*, std::string>* name_map) {
  std::unordered_map<ir::Value, int> map_test;

  int count = 0;
  for (auto it = block->begin(); it != block->end(); ++it) {
    int input = (*it)->num_operands();
    if (input > 0) {
      for (int i = 0; i < input; ++i) {
        VLOG(6) << "input "
                << (*it)->GetOperandByIndex(i).impl()->source().impl();

        void* ptr = (*it)->GetOperandByIndex(i).impl()->source().impl();
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
        VLOG(6) << "output " << (*it)->GetResultByIndex(i).impl();

        void* ptr = (*it)->GetResultByIndex(i).impl();
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
                   const std::unordered_map<void*, std::string>& name_map,
                   paddle::framework::Scope* scope,
                   T* ctx,
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
  for (auto& t : vec_param_list) {
    if (input_set.count(t)) {
      // get information from input
      void* ptr = op->GetOperandByIndex(input_index++).impl()->source().impl();
      auto in_var_name = name_map.at(ptr);

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
      } else if (type_name == "paddle::dialect::PlaceAttribute") {
        ctx->EmplaceBackAttr(
            attr_map[t].dyn_cast<paddle::dialect::PlaceAttribute>().data());
      } else {
        PADDLE_THROW(phi::errors::Unimplemented("attr type not support [%s] ",
                                                type_name));
      }
    }
  }

  void* out_ptr = op->GetResultByIndex(0).impl();
  auto name = name_map.at(out_ptr);

  ctx->EmplaceBackOutput(scope->Var(name)->GetMutable<phi::DenseTensor>());
}

class PhiKernelAdaptor {
 public:
  explicit PhiKernelAdaptor(paddle::framework::Scope* scope) : scope_(scope) {}

  void run(ir::Program* program) {
    auto block = program->block();
    std::unordered_map<void*, std::string> name_map;
    build_scope(block, scope_, &name_map);

    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      VLOG(6) << "begin to run op " << (*it)->op_name();

      auto attr_map = (*it)->attributes();

      InferShapeInterface interface = (*it)->dyn_cast<InferShapeInterface>();
      phi::InferMetaContext ctx;

      build_context<phi::InferMetaContext>((*it), name_map, scope_, &ctx);

      interface.InferShape(&ctx);

      paddle::dialect::GetOpInfoInterface op_info_interface =
          (*it)->dyn_cast<paddle::dialect::GetOpInfoInterface>();
      auto op_info_res = op_info_interface.GetOpInfo();

      auto runtime_info = std::get<3>(op_info_res);

      auto phi_kernels = phi::KernelFactory::Instance().SelectKernelMap(
          runtime_info.kernel_func[0]);
      phi::KernelKey kernel_key(phi::TransToPhiBackend(cpu_place),
                                phi::DataLayout::ALL_LAYOUT,
                                phi::DataType::FLOAT32);
      auto found_it = phi_kernels.find(kernel_key);
      if (found_it == phi_kernels.end()) {
        PADDLE_THROW(paddle::platform::errors::NotFound(
            "can not found kerenl for [%s]", (*it)->op_name()));
      } else {
        phi::KernelContext kernel_ctx(dev_ctx);

        build_context<phi::KernelContext>(
            (*it), name_map, scope_, &kernel_ctx, false);
        found_it->second(&kernel_ctx);

        void* ptr = (*it)->GetResultByIndex(0).impl();
        out_name = name_map[ptr];
      }
    }
  }

  std::string out_name;

 private:
  paddle::framework::Scope* scope_;
};
