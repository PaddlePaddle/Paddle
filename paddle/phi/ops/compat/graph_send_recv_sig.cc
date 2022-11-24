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

KernelSignature GraphSendRecvOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  if (ctx.HasInput("Out_size")) {
<<<<<<< HEAD
    return KernelSignature("graph_send_recv",
                           {"X", "Src_index", "Dst_index"},
                           {"pool_type", "Out_size"},
                           {"Out", "Dst_count"});
  } else {
    return KernelSignature("graph_send_recv",
                           {"X", "Src_index", "Dst_index"},
                           {"pool_type", "out_size"},
=======
    return KernelSignature("send_u_recv",
                           {"X", "Src_index", "Dst_index"},
                           {"reduce_op", "Out_size"},
                           {"Out", "Dst_count"});
  } else {
    return KernelSignature("send_u_recv",
                           {"X", "Src_index", "Dst_index"},
                           {"reduce_op", "out_size"},
>>>>>>> 43b92b633f5d2db98f45d4b9597e5389f6f9712f
                           {"Out", "Dst_count"});
  }
}

KernelSignature GraphSendRecvGradOpArgumentMapping(
    const ArgumentMappingContext& ctx) {
  return KernelSignature(
      "send_u_recv_grad",
      {"X", "Src_index", "Dst_index", "Out", "Dst_count", "Out@GRAD"},
      {"reduce_op"},
      {"X@GRAD"});
}

}  // namespace phi

PD_REGISTER_BASE_KERNEL_NAME(graph_send_recv, send_u_recv);
PD_REGISTER_BASE_KERNEL_NAME(graph_send_recv_grad, send_u_recv_grad);

PD_REGISTER_ARG_MAPPING_FN(graph_send_recv,
                           phi::GraphSendRecvOpArgumentMapping);

PD_REGISTER_ARG_MAPPING_FN(graph_send_recv_grad,
                           phi::GraphSendRecvGradOpArgumentMapping);
