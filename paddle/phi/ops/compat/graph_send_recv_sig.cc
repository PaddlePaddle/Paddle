/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/core/compat/op_utils.h"

namespace phi {

KernelSignature GraphSendRecvGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  std::string pool_type = paddle::any_cast<std::string>(ctx.Attr("pool_type"));
  if (pool_type == "Mean") {
    return KernelSignature(
        "graph_send_recv_grad_mean",
        {GradVarName("Out"), "Src_index", "Dst_index", "Dst_count"},
        {"pool_type"},
        {GradVarName("X")});
  } else if (pool_type == "MAX" || pool_type == "MIN") {
    return KernelSignature(
        "graph_send_recv_grad_minmax",
        {GradVarName("Out"), "X", "Out", "Src_index", "Dst_index"},
        {"pool_type"},
        {GradVarName("X")});
  }
  return KernelSignature("graph_send_recv_grad_sum",
                         {GradVarName("Out"), "Src_index", "Dst_index"},
                         {"pool_type"},
                         {GradVarName("X")});
}

}  // namespace phi

PD_REGISTER_ARG_MAPPING_FN(graph_send_recv_grad,
                           phi::GraphSendRecvGradOpArgumentMapping);
