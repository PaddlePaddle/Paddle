// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/eig_op.h"

namespace paddle {
namespace operators {}  // namespace operators
}  // namespace paddle

using complex64 = paddle::platform::complex<float>;
using complex128 = paddle::platform::complex<double>;
namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(eig, ops::EigKernel<paddle::platform::CUDADeviceContext,
                                            float, complex64, float>,
                        ops::EigKernel<paddle::platform::CUDADeviceContext,
                                       double, complex128, double>,
                        ops::EigKernel<paddle::platform::CUDADeviceContext,
                                       complex64, complex64, float>,
                        ops::EigKernel<paddle::platform::CUDADeviceContext,
                                       complex128, complex128, double>);
REGISTER_OP_CUDA_KERNEL(eig_grad,
                        ops::EigGradKernel<paddle::platform::CUDADeviceContext,
                                           float, complex64, float>,
                        ops::EigGradKernel<paddle::platform::CUDADeviceContext,
                                           double, complex128, double>,
                        ops::EigGradKernel<paddle::platform::CUDADeviceContext,
                                           complex64, complex64, float>,
                        ops::EigGradKernel<paddle::platform::CUDADeviceContext,
                                           complex128, complex128, double>);