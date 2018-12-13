/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. */

#include "paddle/fluid/operators/jit/refer/refer.h"
#include "paddle/fluid/operators/jit/registry.h"

namespace refer = paddle::operators::jit::refer;

#define REGISTER_REFER_KERNEL(key, func)                    \
  REGISTER_JITKERNEL_REFER(key, refer::func##Kernel<float>, \
                           refer::func##Kernel<double>)

REGISTER_REFER_KERNEL(vmul, VMul);
REGISTER_REFER_KERNEL(vadd, VAdd);
REGISTER_REFER_KERNEL(vaddrelu, VAddRelu);
REGISTER_REFER_KERNEL(vsub, VSub);

REGISTER_REFER_KERNEL(vscal, VScal);
REGISTER_REFER_KERNEL(vaddbias, VAddBias);

REGISTER_REFER_KERNEL(vrelu, VRelu);
REGISTER_REFER_KERNEL(videntity, VIdentity);
REGISTER_REFER_KERNEL(vexp, VExp);
REGISTER_REFER_KERNEL(vsigmoid, VSigmoid);
REGISTER_REFER_KERNEL(vtanh, VTanh);

REGISTER_REFER_KERNEL(lstmctht, LSTMCtHt);
REGISTER_REFER_KERNEL(lstmc1h1, LSTMC1H1);

REGISTER_REFER_KERNEL(gruh1, GRUH1);
REGISTER_REFER_KERNEL(gruhtpart1, GRUHtPart1);
REGISTER_REFER_KERNEL(gruhtpart2, GRUHtPart2);

#undef REGISTER_REFER_KERNEL
