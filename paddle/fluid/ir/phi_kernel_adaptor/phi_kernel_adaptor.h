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
#include "paddle/fluid/ir/interface/infershape.h"
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

#include "paddle/fluid/ir/dialect/kernel_attribute.h"
#include "paddle/fluid/ir/dialect/pd_attribute.h"

#include "glog/logging.h"
#include "paddle/fluid/ir/phi_kernel_adaptor/phi_kernel_util.h"

class PhiKernelAdaptor {
 public:
  explicit PhiKernelAdaptor(paddle::framework::Scope* scope) : scope_(scope) {}

  void run(ir::Program* program) {
    auto block = program->block();
    std::unordered_map<ir::Value, std::string> name_map;
    std::cerr << "run here" << std::endl;
    ir::build_scope(block, scope_, &name_map);
    std::cerr << "after buid scope" << std::endl;
    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      VLOG(6) << "begin to run op " << (*it)->name();

      std::cerr << (*it)->name() << std::endl;
      auto attr_map = (*it)->attributes();

      paddle::dialect::OpYamlInfoInterface op_info_interface =
          (*it)->dyn_cast<paddle::dialect::OpYamlInfoInterface>();
      auto op_info_res = op_info_interface.GetOpInfo();

      paddle::dialect::InferShapeInterface interface =
          (*it)->dyn_cast<paddle::dialect::InferShapeInterface>();
      phi::InferMetaContext ctx;

      ir::build_infer_meta_context((*it), name_map, scope_, op_info_res, &ctx);

      interface.InferShape(&ctx);

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

        ir::build_phi_kernel_context(
            (*it), name_map, scope_, op_info_res, &kernel_ctx);
        found_it->second(&kernel_ctx);

        auto out_value = (*it)->result(0);
        out_name = name_map[out_value];
      }
    }
  }

  void run_kernel_prog(ir::Program* program) {
    auto block = program->block();
    std::unordered_map<ir::Value, std::string> name_map;
    build_scope(block, scope_, &name_map);
    ir::IrContext* ctx = ir::IrContext::Instance();

    ctx->GetOrRegisterDialect<paddle::dialect::PaddleDialect>();

    auto* dev_ctx = phi::DeviceContextPool::Instance().Get(phi::CPUPlace());
    phi::Place cpu_place(phi::AllocationType::CPU);
    for (auto it = block->begin(); it != block->end(); ++it) {
      auto attr_map = (*it)->attributes();

      auto op_name = attr_map.at("op_name").dyn_cast<ir::StrAttribute>().data();

      ir::OpInfo op1_info = ctx->GetRegisteredOpInfo(op_name);

      auto impl =
          op1_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      auto yaml_info = impl->get_op_info_();

      auto attr_info = std::get<1>(yaml_info);

      auto infer_shape_impl =
          op1_info.GetInterfaceImpl<paddle::dialect::InferShapeInterface>();

      phi::InferMetaContext ctx;

      ir::build_infer_meta_context((*it), name_map, scope_, yaml_info, &ctx);

      infer_shape_impl->infer_shape_(&ctx);

      auto kernel_name =
          attr_map.at("kernel_name").dyn_cast<ir::StrAttribute>().data();
      auto kernel_key = attr_map.at("kernel_key")
                            .dyn_cast<paddle::dialect::KernelAttribute>()
                            .data();

      auto kernel_fn =
          phi::KernelFactory::Instance().SelectKernel(kernel_name, kernel_key);

      phi::KernelContext kernel_ctx(dev_ctx);

      ir::build_phi_kernel_context(
          (*it), name_map, scope_, yaml_info, &kernel_ctx);
      kernel_fn(&kernel_ctx);

      auto out_value = (*it)->result(0);
      out_name = name_map[out_value];
    }
  }

  std::string out_name;

 private:
  paddle::framework::Scope* scope_;
};
