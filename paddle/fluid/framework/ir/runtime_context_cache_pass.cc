/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/ir/runtime_context_cache_pass.h"
#include <memory>
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"

namespace paddle {
namespace framework {
namespace ir {

void RuntimeContextCachePass::ApplyImpl(ir::Graph* graph) const {
  VLOG(3) << "Applies Runtime Context Cache strategy.";
  auto& op_info = OpInfoMap::Instance();
  for (const Node* n : graph->Nodes()) {
    if (n->IsOp()) {
      LOG(INFO) << "n is a Op Node.";
      auto* op_desc = n->Op();
      LOG(INFO) << "type: " << op_desc->Type();
      auto op_info_ptr = op_info.Get(op_desc->Type());
      LOG(INFO) << "Get op info";
      if (op_info_ptr.HasOpProtoAndChecker()) {
        // proto::OpProto* proto = op_info_ptr.proto_;
        // bool has_attr = false;
        // for (int i = 0; i != proto->attrs_size(); ++i) {
        //   const proto::OpProto::Attr &attr = proto->attrs(i);
        //   LOG(INFO) << "attr name: " << attr.name();
        //   if (attr.name() == kEnableCacheRuntimeContext) {
        //     has_attr = true;
        //     n->Op()->SetAttr(kEnableCacheRuntimeContext, true);
        //     break;
        //   }
        // }
        // if (!has_attr) {
        //   auto *attr = proto->add_attrs();
        //   attr->set_name(kEnableCacheRuntimeContext);
        // }
        n->Op()->SetAttr(kEnableCacheRuntimeContext, true);
      } else {
        // PADDLE_THROW("Operator %s doesn't register op proto info.",
        // op_desc->Type());
        LOG(INFO) << "Operator " << op_desc->Type()
                  << " doesn't register op proto info.";
      }
    }
  }
}

}  // namespace ir
}  // namespace framework
}  // namespace paddle

REGISTER_PASS(runtime_context_cache_pass,
              paddle::framework::ir::RuntimeContextCachePass);
