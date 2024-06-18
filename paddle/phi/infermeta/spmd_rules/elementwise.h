/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <vector>

#include "paddle/phi/core/distributed/auto_parallel/dist_meta_tensor.h"
#include "paddle/phi/core/distributed/type_defs.h"

namespace phi {
namespace distributed {

SpmdInfo ElementwiseUnaryInferSpmd(const DistMetaTensor& x);

SpmdInfo ElementwiseUnaryWithPartialInferSpmd(const DistMetaTensor& x);

SpmdInfo ElementwiseUnaryInferSpmdReverse(const DistMetaTensor& x,
                                          const DistMetaTensor& out);

SpmdInfo ElementwiseUnaryGradInferSpmd(const DistMetaTensor& x,
                                       const DistMetaTensor& out_grad);

SpmdInfo ElementwiseUnaryGradInferSpmd(const DistMetaTensor& x,
                                       const DistMetaTensor& out,
                                       const DistMetaTensor& out_grad);

SpmdInfo ElementwiseBinaryInferSpmd(const DistMetaTensor& x,
                                    const DistMetaTensor& y);

SpmdInfo ElementwiseBinaryInferSpmdReverse(const DistMetaTensor& x,
                                           const DistMetaTensor& y,
                                           const DistMetaTensor& out);

SpmdInfo ElementwiseBinaryGradInferSpmd(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& out_grad,
                                        int64_t axis = -1);

SpmdInfo ElementwiseBinaryGradInferSpmd(const DistMetaTensor& x,
                                        const DistMetaTensor& y,
                                        const DistMetaTensor& out,
                                        const DistMetaTensor& out_grad,
                                        int64_t axis = -1);

SpmdInfo SwiGLUInferSpmd(const DistMetaTensor& x, const DistMetaTensor& y);

SpmdInfo SwiGLUInferSpmdReverse(const DistMetaTensor& x,
                                const DistMetaTensor& y,
                                const DistMetaTensor& out);

SpmdInfo SwiGLUGradInferSpmd(const DistMetaTensor& x,
                             const DistMetaTensor& y,
                             const DistMetaTensor& out_grad);

}  // namespace distributed
}  // namespace phi
