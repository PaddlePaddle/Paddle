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

REGISTER_REFER_KERNEL(kVMul, VMul);
REGISTER_REFER_KERNEL(kVAdd, VAdd);
REGISTER_REFER_KERNEL(kVAddRelu, VAddRelu);
REGISTER_REFER_KERNEL(kVSub, VSub);

REGISTER_REFER_KERNEL(kVScal, VScal);
REGISTER_REFER_KERNEL(kVAddBias, VAddBias);

REGISTER_REFER_KERNEL(kVRelu, VRelu);
REGISTER_REFER_KERNEL(kVIdentity, VIdentity);
REGISTER_REFER_KERNEL(kVSquare, VSquare);
REGISTER_REFER_KERNEL(kVExp, VExp);
REGISTER_REFER_KERNEL(kVSigmoid, VSigmoid);
REGISTER_REFER_KERNEL(kVTanh, VTanh);

REGISTER_REFER_KERNEL(kLSTMCtHt, LSTMCtHt);
REGISTER_REFER_KERNEL(kLSTMC1H1, LSTMC1H1);

REGISTER_REFER_KERNEL(kGRUH1, GRUH1);
REGISTER_REFER_KERNEL(kGRUHtPart1, GRUHtPart1);
REGISTER_REFER_KERNEL(kGRUHtPart2, GRUHtPart2);

REGISTER_REFER_KERNEL(kCRFDecoding, CRFDecoding);
REGISTER_REFER_KERNEL(kLayerNorm, LayerNorm);

REGISTER_REFER_KERNEL(kNCHW16CMulNC, NCHW16CMulNC);

REGISTER_REFER_KERNEL(kSeqPool, SeqPool);

REGISTER_REFER_KERNEL(kMatMul, MatMul);

#undef REGISTER_REFER_KERNEL
