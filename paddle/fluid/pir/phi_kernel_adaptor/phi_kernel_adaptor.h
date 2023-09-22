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

#include "paddle/fluid/pir/dialect/operator/interface/infermeta.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/phi/core/meta_tensor.h"
#include "paddle/phi/infermeta/binary.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/pir/core/builtin_attribute.h"
#include "paddle/pir/core/builtin_dialect.h"
#include "paddle/pir/core/builtin_op.h"
#include "paddle/pir/core/ir_context.h"
#include "paddle/pir/core/program.h"
#include "paddle/pir/core/utils.h"

#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"

#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_context.h"
#include "paddle/phi/core/kernel_factory.h"

#include "paddle/fluid/platform/init.h"

#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"

#include "glog/logging.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/pir/phi_kernel_adaptor/phi_kernel_util.h"

class PhiKernelAdaptor {
 public:
  explicit PhiKernelAdaptor(paddle::framework::Scope* scope) : scope_(scope) {
    value_exe_info_ =
        std::make_shared<paddle::framework::ValueExecutionInfo>(scope_);
  }

  void run_kernel_prog(pir::Program* program) {
    auto block = program->block();
    std::map<pir::Block*, paddle::framework::Scope*> sub_blocks;
    std::stringstream ss;
    ss << this;

    BuildScope(*block, ss.str(), &sub_blocks, value_exe_info_.get());
    pir::IrContext* ctx = pir::IrContext::Instance();

    ctx->GetOrRegisterDialect<paddle::dialect::OperatorDialect>();

    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      auto attr_map = (*it)->attributes();

      auto op_name =
          attr_map.at("op_name").dyn_cast<pir::StrAttribute>().AsString();

      pir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op_name);

      auto impl =
          op1_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      auto yaml_info = impl->get_op_info_();

      auto attr_info = std::get<1>(yaml_info);

      auto infer_meta_impl =
          op1_info.GetInterfaceImpl<paddle::dialect::InferMetaInterface>();

      phi::InferMetaContext ctx;

      paddle::dialect::OpYamlInfoParser op_yaml_info_parser(yaml_info);
      pir::BuildPhiContext<
          phi::InferMetaContext,
          phi::MetaTensor,
          phi::MetaTensor,
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
          paddle::small_vector<phi::MetaTensor, phi::kInputSmallVectorSize>,
          false>((*it),
                 value_exe_info_->GetValue2VarName(),
                 scope_,
                 nullptr,
                 op_yaml_info_parser,
                 &ctx);

      infer_meta_impl->infer_meta_(&ctx);

      auto kernel_name =
          attr_map.at("kernel_name").dyn_cast<pir::StrAttribute>().AsString();
      auto kernel_key = attr_map.at("kernel_key")
                            .dyn_cast<paddle::dialect::KernelAttribute>()
                            .data();

      auto kernel_fn =
          phi::KernelFactory::Instance().SelectKernel(kernel_name, kernel_key);

      phi::KernelContext kernel_ctx(dev_ctx);

      pir::BuildPhiContext<phi::KernelContext,
                           const phi::TensorBase*,
                           phi::TensorBase*,
                           paddle::small_vector<const phi::TensorBase*>,
                           paddle::small_vector<phi::TensorBase*>,
                           true>((*it),
                                 value_exe_info_->GetValue2VarName(),
                                 scope_,
                                 nullptr,
                                 op_yaml_info_parser,
                                 &kernel_ctx);
      kernel_fn(&kernel_ctx);

      auto out_value = (*it)->result(0);
      out_name = value_exe_info_->GetValue2VarName()[out_value];
    }
  }

  std::string out_name;

 private:
  paddle::framework::Scope* scope_;
  std::shared_ptr<paddle::framework::ValueExecutionInfo> value_exe_info_;
};
