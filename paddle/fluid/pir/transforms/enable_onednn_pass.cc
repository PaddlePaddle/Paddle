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

#include "paddle/fluid/pir/transforms/enable_onednn_pass.h"

#include <iostream>
#include <regex>
#include <string>
#include <unordered_set>

#include "paddle/common/flags.h"
#include "paddle/fluid/framework/op_kernel_type.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_attribute.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_dialect.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_op.h"
#include "paddle/fluid/pir/dialect/kernel/ir/kernel_type.h"
#include "paddle/fluid/pir/dialect/operator/interface/op_yaml_info.h"
#include "paddle/fluid/pir/dialect/operator/interface/parse_kernel_key.h"
#include "paddle/fluid/pir/dialect/operator/ir/control_flow_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/manual_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_attribute.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_dialect.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_type.h"
#include "paddle/fluid/pir/dialect/operator/ir/pd_op.h"
#include "paddle/fluid/pir/dialect/operator/trait/inplace.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_parser.h"
#include "paddle/fluid/pir/dialect/operator/utils/op_yaml_info_util.h"
#include "paddle/fluid/pir/dialect/operator/utils/utils.h"
#include "paddle/fluid/pir/transforms/transform_general_functions.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/phi/api/lib/data_transform.h"
#include "paddle/phi/api/lib/kernel_dispatch.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/common/type_traits.h"
#include "paddle/phi/core/compat/convert_utils.h"
#include "paddle/phi/core/kernel_factory.h"
#include "paddle/pir/dialect/control_flow/ir/cf_op.h"
#include "paddle/pir/include/core/builtin_op.h"

#ifdef PADDLE_WITH_DNNL
#include "build/paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/pir/dialect/operator/ir/onednn_op.h"
#include "paddle/fluid/pir/dialect/operator/ir/op_onednn_dialect.h"
#include "paddle/fluid/pir/dialect/operator/trait/onednn.h"
#endif

COMMON_DECLARE_bool(print_ir);
// COMMON_DECLARE_string(pir_onednn_kernel_blacklist);

namespace paddle {
namespace dialect {

void EnableOneDNNPass(pir::Program* prog) {
  if (FLAGS_print_ir) {
    std::cout << "IR before enable onednn pass = " << *prog << std::endl;
  }

  auto block = prog->block();

  pir::IrContext* ctx = pir::IrContext::Instance();
  ctx->GetOrRegisterDialect<OperatorDialect>();
  ctx->GetOrRegisterDialect<KernelDialect>();
  ctx->GetOrRegisterDialect<CustomKernelDialect>();

#ifdef PADDLE_WITH_DNNL
  ctx->GetOrRegisterDialect<OneDNNOperatorDialect>();
  ctx->GetOrRegisterDialect<OneDNNKernelDialect>();
#endif

  for (auto iter = block->begin(); iter != block->end(); ++iter) {
    pir::Operation* op_item = &(*iter);
    std::string target_op_name = op_item->name();
    target_op_name.replace(0, 5, "onednn_op");
    auto op_info = ctx->GetRegisteredOpInfo(target_op_name);
    if (op_info) {
      std::vector<pir::Type> op_item_inner_output_types;
      if (op_item->num_results() > 0) {
        for (size_t i = 0; i < op_item->num_results(); ++i) {
          op_item_inner_output_types.push_back(op_item->result_type(i));
        }
      }
      auto attributes = op_item->attributes();
      auto yaml_interface =
          op_info.GetInterfaceImpl<paddle::dialect::OpYamlInfoInterface>();
      OpRunTimeInfo runtime_info =
          std::get<3>(yaml_interface->get_op_info_(target_op_name));
      for (auto& attr : runtime_info.extra_args_default_value) {
        attributes[attr.first] = attr.second;
      }
      pir::Operation* op_item_inner =
          pir::Operation::Create(op_item->operands_source(),
                                 attributes,
                                 op_item_inner_output_types,
                                 op_info);
      op_item->ReplaceAllUsesWith(op_item_inner->results());
      for (auto iter = block->begin(); iter != block->end(); ++iter) {
        if (*iter == *op_item) {
          block->Assign(iter, op_item_inner);
          break;
        }
      }
    }
  }

  if (FLAGS_print_ir) {
    std::cout << "IR after enable onednn pass = " << *prog << std::endl;
  }
}
}  // namespace dialect
}  // namespace paddle
