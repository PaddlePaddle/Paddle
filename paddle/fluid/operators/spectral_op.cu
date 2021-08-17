/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/operators/spectral_op.h"
#include "paddle/fluid/platform/complex.h"

namespace paddle {
namespace operators {

template <typename T>
struct FFTC2CFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx, const Tensor* X,
                  Tensor* out, const std::vector<int64_t>& axes,
                  FFTNormMode normalization, bool forward) {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    fft_c2c, ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CKernel<paddle::platform::CUDADeviceContext, double>);

REGISTER_OP_CUDA_KERNEL(
    fft_c2c_grad,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::FFTC2CGradKernel<paddle::platform::CUDADeviceContext, double>);
