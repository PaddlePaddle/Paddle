/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at
   http://www.apache.org/licenses/LICENSE-2.0
   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include <algorithm>
#include <string>

#include "paddle/fluid/operators/interpolate_op.cu.h"
#include "paddle/fluid/operators/interpolate_op.h"

namespace paddle {
namespace operators {

template <typename T>
class InterpolateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        platform::errors::NotFound("This kernel only runs on GPU device."));
    auto* input = ctx.Input<Tensor>("X");
    auto* output = ctx.Output<Tensor>("Out");

    auto input_dims = input->dims();
    if (input_dims.size() == 3) {  // 1D interpolation
      Interpolate1DCUDAFwd<T>(ctx, *input, output);
    } else if (input_dims.size() == 4) {  // 2D interpolation
      Interpolate2DCUDAFwd<T>(ctx, *input, output);
    } else if (input_dims.size() == 5) {  // 3D interpolation
      Interpolate3DCUDAFwd<T>(ctx, *input, output);
    }
  }
};

template <typename T>
class InterpolateGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()),
        true,
        platform::errors::NotFound("This kernel only runs on GPU device."));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));

    auto output_grad_dims = output_grad->dims();
    if (output_grad_dims.size() == 3) {  // 1D interpolation
      Interpolate1DCUDABwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 4) {  // 2D interpolation
      Interpolate2DCUDABwd<T>(ctx, input_grad, *output_grad);
    } else if (output_grad_dims.size() == 5) {  // 3D interpolation
      Interpolate3DCUDABwd<T>(ctx, input_grad, *output_grad);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(bilinear_interp,
                        ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bilinear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(nearest_interp,
                        ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(nearest_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(trilinear_interp,
                        ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(trilinear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(linear_interp,
                        ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(linear_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(bicubic_interp,
                        ops::InterpolateOpCUDAKernel<float>,
                        ops::InterpolateOpCUDAKernel<double>,
                        ops::InterpolateOpCUDAKernel<int>);
REGISTER_OP_CUDA_KERNEL(bicubic_interp_grad,
                        ops::InterpolateGradOpCUDAKernel<float>,
                        ops::InterpolateGradOpCUDAKernel<double>);
