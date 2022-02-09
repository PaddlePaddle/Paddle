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

#define REGISTER_REFER_KERNEL(func)                             \
  REGISTER_JITKERNEL_REFER(k##func, refer::func##Kernel<float>, \
                           refer::func##Kernel<double>)

REGISTER_REFER_KERNEL(VMul);
REGISTER_REFER_KERNEL(VAdd);
REGISTER_REFER_KERNEL(VAddRelu);
REGISTER_REFER_KERNEL(VSub);

REGISTER_REFER_KERNEL(VScal);
REGISTER_REFER_KERNEL(StrideScal);
REGISTER_REFER_KERNEL(VAddBias);

REGISTER_REFER_KERNEL(VRelu);
REGISTER_REFER_KERNEL(VCopy);
REGISTER_REFER_KERNEL(VIdentity);
REGISTER_REFER_KERNEL(VSquare);
REGISTER_REFER_KERNEL(VExp);
REGISTER_REFER_KERNEL(VSigmoid);
REGISTER_REFER_KERNEL(VTanh);

REGISTER_REFER_KERNEL(LSTMCtHt);
REGISTER_REFER_KERNEL(LSTMC1H1);

REGISTER_REFER_KERNEL(GRUH1);
REGISTER_REFER_KERNEL(GRUHtPart1);
REGISTER_REFER_KERNEL(GRUHtPart2);

REGISTER_REFER_KERNEL(CRFDecoding);
REGISTER_REFER_KERNEL(LayerNorm);
REGISTER_REFER_KERNEL(NCHW16CMulNC);
REGISTER_REFER_KERNEL(SeqPool);
REGISTER_REFER_KERNEL(MatMul);
REGISTER_REFER_KERNEL(HMax);
REGISTER_REFER_KERNEL(HSum);
REGISTER_REFER_KERNEL(StrideASum);
REGISTER_REFER_KERNEL(Softmax);
REGISTER_REFER_KERNEL(EmbSeqPool);
REGISTER_REFER_KERNEL(Adam);
REGISTER_REFER_KERNEL(Sgd);
REGISTER_REFER_KERNEL(VBroadcast);

#undef REGISTER_REFER_KERNEL
