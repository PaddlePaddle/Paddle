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

namespace {
template <typename DeviceContext, typename T>
void fft_c2c_cufft(const DeviceContext& ctx, const Tensor* X, Tensor* out,
                   const std::vector<int64_t>& axes, int64_t normalization,
                   bool forward) {
  // const auto x_dims = x->dims();
}

template <typename DeviceContext, typename T>
void fft_c2c_cufft_backward(const DeviceContext& ctx, const Tensor* X,
                            Tensor* out, const std::vector<int64_t>& axes,
                            int64_t normalization, bool forward) {}

}  // anonymous namespace

template <typename T>
class FFTC2CKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    using U = paddle::platform::complex<T>;
    auto& dev_ctx = ctx.device_context();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const std::string norm_str = ctx.Attr<std::string>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    auto* x = ctx.Input<Tensor>("X");
    auto* y = ctx.Output<Tensor>("Out");

    auto* y_data = y->mutable_data<T>(ctx.GetPlace());
    auto normalization = get_norm_from_string(norm_str, forward);

    fft_c2c_cufft<platform::CUDADeviceContext, U>(dev_ctx, x, y, axes,
                                                  normalization, forward);
  }
};

template <typename T>
class FFTC2CGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    using U = FFTC2CParamType<T>;
    auto& dev_ctx = ctx.device_context();

    auto axes = ctx.Attr<std::vector<int64_t>>("axes");
    const int64_t normalization = ctx.Attr<int64_t>("normalization");
    const bool forward = ctx.Attr<bool>("forward");
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto *d_y_data = d_y->mutable_data<T>(ctx.GetPlace()
    auto normalization = get_norm_from_string(norm_str, forward);

    fft_c2c_cufft_backward<platform::CUDADeviceContext, U>(dev_ctx, d_x, d_y,
                                                axes, normalization, forward);
  }
};
}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(fft_c2c, ops::FFTC2CKernel<float>,
                        ops::FFTC2CKernel<double>);

REGISTER_OP_CUDA_KERNEL(fft_c2c_grad, ops::FFTC2CGradKernel<float>,
                        ops::FFTC2CGradKernel<double>);
