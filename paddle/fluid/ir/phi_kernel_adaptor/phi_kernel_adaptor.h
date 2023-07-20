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

#include "paddle/fluid/ir/dialect/op_yaml_info_util.h"
#include "paddle/fluid/ir/dialect/pd_dialect.h"
#include "paddle/fluid/ir/dialect/pd_op.h"
#include "paddle/fluid/ir/dialect/pd_type.h"
#include "paddle/fluid/ir/dialect/utils.h"
#include "paddle/fluid/ir/interface/infermeta.h"
#include "paddle/fluid/ir/interface/op_yaml_info.h"
#include "paddle/fluid/ir/interface/op_yaml_info_parser.h"
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

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"

#include "glog/logging.h"
#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"

class PhiKernelAdaptor {
 public:
  explicit PhiKernelAdaptor(paddle::framework::Scope* scope) : scope_(scope) {}

  void run_kernel_prog(ir::Program* program) {
    auto block = program->block();
    std::unordered_map<ir::Value, std::string> value_2_var_name;
    std::unordered_map<const paddle::framework::Variable*, std::string>
        variable_2_var_name;
    std::map<std::string, int> var_name_2_id;
    std::vector<paddle::framework::Variable*> variable_list;
    std::stringstream ss;
    ss << this;

    BuildScope(*block,
               scope_,
               ss.str(),
               &value_2_var_name,
               &variable_2_var_name,
               &var_name_2_id,
               &variable_list);
    ir::IrContext* ctx = ir::IrContext::Instance();

    ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      auto attr_map = (*it)->attributes();

      auto op_name =
          attr_map.at("op_name").dyn_cast<ir::StrAttribute>().AsString();

      ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op_name);

      auto impl =
          op1_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      auto yaml_info = impl->get_op_info_();

      auto attr_info = std::get<1>(yaml_info);

      auto infer_meta_impl =
          op1_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();

      phi::InferMetaContext ctx;

      paddle::dialect::OpYamlInfoParser op_yaml_info_parser(yaml_info);
      ir::BuildPhiContext<
          phi::InferMetaContext,
          phi::MetaTensor,
          phi::MetaTensor,
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
          false>(
          (*it), value_2_var_name, scope_, nullptr, op_yaml_info_parser, &ctx);

      infer_meta_impl->infer_meta_(&ctx);

      auto kernel_name =
          attr_map.at("kernel_name").dyn_cast<ir::StrAttribute>().AsString();
      auto kernel_key = attr_map.at("kernel_key")
                            .dyn_cast<paddle::dialect::KernelAttribute>()
                            .data();

      auto kernel_fn =
          phi::KernelFactory::Instance().SelectKernel(kernel_name, kernel_key);

      phi::KernelContext kernel_ctx(dev_ctx);

      ir::BuildPhiContext<phi::KernelContext,
                          const phi::TensorBase*,
                          phi::TensorBase*,
                          paddle::small_vector<const phi::TensorBase*>,
                          paddle::small_vector<phi::TensorBase*>,
                          true>((*it),
                                value_2_var_name,
                                scope_,
                                nullptr,
                                op_yaml_info_parser,
                                &kernel_ctx);
      kernel_fn(&kernel_ctx);

      auto out_value = (*it)->result(0);
      out_name = value_2_var_name[out_value];
    }
  }

  std::string out_name;

 private:
  paddle::framework::Scope* scope_;
};
