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

#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T>
static HOSTDEVICE inline T sign(T x) {
  if (x > 0)
    return static_cast<T>(1);
  else if (x == static_cast<T>(0))
    return static_cast<T>(0);
  else
    return static_cast<T>(-1);
}

template <typename T>
__global__ void KeAffinityPropagate2DFw(T* output, const T* input,
                                        const T* gate_weight, const T* mask,
                                        const int kernel_size, const int n,
                                        const int chw, const int h,
                                        const int w) {
  int nthreads = n * chw;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int ni = tid / chw;
    int hi = (tid % (h * w)) / w;
    int wi = tid % w;

    int side_num = (kernel_size - 1) / 2;
    int channel_num = kernel_size * kernel_size - 1;

    int rc = 0;
    T result = 0.;
    T weight_sum = 0.;
    for (int i = -side_num; i <= side_num; i++) {
      for (int j = -side_num; j <= side_num; j++) {
        if (i != 0 || j != 0) {
          int rh = hi - i;
          int rw = wi - j;
          int gate_idx = ni * channel_num * h * w + rc * h * w + hi * w + wi;
          weight_sum += gate_weight[gate_idx];
          if (rh >= 0 && rh < h && rw >= 0 && rw < w) {
            int input_idx = tid - i * w - j;
            result += input[input_idx] * gate_weight[gate_idx];
          }
          rc++;
        }
      }
    }

    output[tid] = result + (1. - weight_sum) * input[tid];
    if (mask) {
      T rm = sign(mask[tid]);
      output[tid] = (1. - rm) * output[tid] + rm * input[tid];
    }
  }
}

template <typename T>
__global__ void KeAffinityPropagate2DBw(T* input_grad, const T* output_grad,
                                        const T* input, const T* gate_weight,
                                        const T* mask, const int kernel_size,
                                        const int n, const int chw, const int h,
                                        const int w) {}

template <typename T>
class AffinityPropagateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto* input = ctx.Input<Tensor>("X");
    auto* gate_weight = ctx.Input<Tensor>("GateWeight");
    auto* mask = ctx.Input<Tensor>("Mask");
    auto* output = ctx.Output<Tensor>("Out");

    const int kernel_size = ctx.Attr<int>("kernel_size");
    int pad_size = kernel_size - 1;

    const T* input_data = input->data<T>();
    const T* gate_weight_data = gate_weight->data<T>();
    const T* mask_data;
    if (mask)
      mask_data = mask->data<T>();
    else
      mask_data = NULL;

    auto input_dims = input->dims();
    if (input_dims.size() == 4) {
      const int n = input_dims[0];
      const int c = input_dims[1];
      const int h = input_dims[2];
      const int w = input_dims[3];

      auto* output_data = output->mutable_data<T>({n, c, h, w}, ctx.GetPlace());

      const int chw = c * h * w;
      int pixelNum = n * chw;
      int grid_dim = (pixelNum + 512 - 1) / 512;
      grid_dim = grid_dim > 8 ? 8 : grid_dim;

      KeAffinityPropagate2DFw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          output_data, input_data, gate_weight_data, mask_data, kernel_size, n,
          chw, h, w);
    }
  }
};

template <typename T>
class AffinityPropagateGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gate_weight = ctx.Input<Tensor>("GateWeight");
    auto* mask = ctx.Input<Tensor>("Mask");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));

    const int kernel_size = ctx.Attr<int>("kernel_size");
    int pad_size = kernel_size - 1;

    const T* input_data = input->data<T>();
    const T* gate_weight_data = gate_weight->data<T>();
    const T* output_grad_data = output_grad->data<T>();
    const T* mask_data;
    if (mask)
      mask_data = mask->data<T>();
    else
      mask_data = NULL;

    auto input_dims = input->dims();
    if (input_dims.size() == 4) {
      const int n = input_dims[0];
      const int c = input_dims[1];
      const int h = input_dims[2];
      const int w = input_dims[3];

      auto* input_grad_data =
          input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());

      const int chw = c * h * w;
      int pixelNum = n * chw;
      int grid_dim = (pixelNum + 512 - 1) / 512;
      grid_dim = grid_dim > 8 ? 8 : grid_dim;

      KeAffinityPropagate2DBw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_grad_data, output_grad_data, input_data, gate_weight_data,
          mask_data, kernel_size, n, chw, h, w);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(affinity_propagate,
                        ops::AffinityPropagateOpCUDAKernel<float>,
                        ops::AffinityPropagateOpCUDAKernel<double>);
REGISTER_OP_CUDA_KERNEL(affinity_propagate_grad,
                        ops::AffinityPropagateGradOpCUDAKernel<float>,
                        ops::AffinityPropagateGradOpCUDAKernel<double>);
