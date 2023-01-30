<<<<<<< HEAD
// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/save_combine_op.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"

PD_REGISTER_KERNEL(save_combine_tensor,
                   GPU,
                   ALL_LAYOUT,
                   paddle::operators::SaveCombineTensorKernel,
                   int,
                   int64_t,
                   float,
                   double) {}

PD_REGISTER_KERNEL(save_combine_vocab,
                   GPU,
                   ALL_LAYOUT,
                   paddle::operators::SaveCombineVocabKernel,
                   int,
                   int64_t,
                   float,
                   double) {}
=======
/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/save_combine_op.h"

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(save_combine,
                        ops::SaveCombineOpKernel<phi::GPUContext, float>,
                        ops::SaveCombineOpKernel<phi::GPUContext, double>,
                        ops::SaveCombineOpKernel<phi::GPUContext, int>,
                        ops::SaveCombineOpKernel<phi::GPUContext, int64_t>);
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
