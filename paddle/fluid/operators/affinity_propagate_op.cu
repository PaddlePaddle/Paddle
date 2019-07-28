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
#include "paddle/fluid/operators/math/math_function.h"
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
                                        const T* gate_weight,
                                        const int kernel_size, const int n,
                                        const int chw, const int h,
                                        const int w) {
  int side_num = (kernel_size - 1) / 2;
  int channel_num = kernel_size * kernel_size - 1;
  int nthreads = n * chw;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int ni = tid / chw;
    int hi = (tid % (h * w)) / w;
    int wi = tid % w;

    int rc = 0;
    T result = 0.;
    for (int i = -side_num; i <= side_num; i++) {
      for (int j = -side_num; j <= side_num; j++) {
        if (i != 0 || j != 0) {
          int rh = hi - i;
          int rw = wi - j;
          if (rh >= 0 && rh < h && rw >= 0 && rw < w) {
            int input_idx = tid - i * w - j;
            int gate_idx = ni * channel_num * h * w + rc * h * w + hi * w + wi;
            result += input[input_idx] * gate_weight[gate_idx];
          }
          rc++;
        }
      }
    }
    output[tid] = result;
  }
}

template <typename T>
__global__ void KeAffinityPropagate2DBw(T* input_grad, T* gate_weight_grad,
                                        const T* output_grad, const T* input,
                                        const T* gate_weight,
                                        const int kernel_size, const int n,
                                        const int chw, const int h,
                                        const int w) {
  int side_num = (kernel_size - 1) / 2;
  int channel_num = kernel_size * kernel_size - 1;
  int nthreads = n * chw;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int ni = tid / chw;
    int hi = (tid % (h * w)) / w;
    int wi = tid % w;

    int rc = 0;
    for (int i = -side_num; i <= side_num; i++) {
      for (int j = -side_num; j <= side_num; j++) {
        if (i != 0 || j != 0) {
          int rh = hi - i;
          int rw = wi - j;
          if (rh >= 0 && rh < h && rw >= 0 && rw < w) {
            int input_idx = tid - i * w - j;
            int gate_idx = ni * channel_num * h * w + rc * h * w + hi * w + wi;
            platform::CudaAtomicAdd(&input_grad[input_idx],
                                    output_grad[tid] * gate_weight[gate_idx]);
            platform::CudaAtomicAdd(&gate_weight_grad[gate_idx],
                                    input[input_idx] * output_grad[tid]);
          }
          rc++;
        }
      }
    }
  }
}

template <typename T>
__global__ void KeAffinityPropagate3DFw(T* output, const T* input,
                                        const T* gate_weight,
                                        const int kernel_size, const int n,
                                        const int cdhw, const int d,
                                        const int h, const int w) {
  int side_num = (kernel_size - 1) / 2;
  int channel_num = kernel_size * kernel_size * kernel_size - 1;
  int nthreads = n * cdhw;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int ni = tid / cdhw;
    int di = (tid % (d * h * w)) / h / w;
    int hi = (tid % (h * w)) / w;
    int wi = tid % w;

    int rc = 0;
    T result = 0.;
    for (int i = -side_num; i <= side_num; i++) {
      for (int j = -side_num; j <= side_num; j++) {
        for (int l = -side_num; l <= side_num; l++) {
          if (i != 0 || j != 0 || l != 0) {
            int rd = di - i;
            int rh = hi - j;
            int rw = wi - l;
            if (rd >= 0 && rd < d && rh >= 0 && rh < h && rw >= 0 && rw < w) {
              int input_idx = tid - i * h * w - j * w - l;
              int gate_idx = ni * channel_num * d * h * w + rc * d * h * w +
                             di * h * w + hi * w + wi;
              result += input[input_idx] * gate_weight[gate_idx];
            }
            rc++;
          }
        }
      }
    }
    output[tid] = result;
  }
}

template <typename T>
__global__ void KeAffinityPropagate3DBw(T* input_grad, T* gate_weight_grad,
                                        const T* output_grad, const T* input,
                                        const T* gate_weight,
                                        const int kernel_size, const int n,
                                        const int cdhw, const int d,
                                        const int h, const int w) {
  int side_num = (kernel_size - 1) / 2;
  int channel_num = kernel_size * kernel_size * kernel_size - 1;
  int nthreads = n * cdhw;
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (; tid < nthreads; tid += stride) {
    int ni = tid / cdhw;
    int di = (tid % (d * h * w)) / h / w;
    int hi = (tid % (h * w)) / w;
    int wi = tid % w;

    int rc = 0;
    for (int i = -side_num; i <= side_num; i++) {
      for (int j = -side_num; j <= side_num; j++) {
        for (int l = -side_num; l <= side_num; l++) {
          if (i != 0 || j != 0 || l != 0) {
            int rd = di - i;
            int rh = hi - j;
            int rw = wi - l;
            if (rd >= 0 && rd < d && rh >= 0 && rh < h && rw >= 0 && rw < w) {
              int input_idx = tid - i * h * w - j * w - l;
              int gate_idx = ni * channel_num * d * h * w + rc * d * h * w +
                             di * h * w + hi * w + wi;
              platform::CudaAtomicAdd(&input_grad[input_idx],
                                      output_grad[tid] * gate_weight[gate_idx]);
              platform::CudaAtomicAdd(&gate_weight_grad[gate_idx],
                                      input[input_idx] * output_grad[tid]);
            }
            rc++;
          }
        }
      }
    }
  }
}

template <typename T>
class AffinityPropagateOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE(platform::is_gpu_place(ctx.GetPlace()),
                   "This kernel only runs on GPU device.");

    auto* input = ctx.Input<Tensor>("X");
    auto* gate_weight = ctx.Input<Tensor>("GateWeight");
    auto* output = ctx.Output<Tensor>("Out");

    const int kernel_size = ctx.Attr<int>("kernel_size");

    const T* input_data = input->data<T>();
    const T* gate_weight_data = gate_weight->data<T>();

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
          output_data, input_data, gate_weight_data, kernel_size, n, chw, h, w);
    } else if (input_dims.size() == 5) {
      const int n = input_dims[0];
      const int c = input_dims[1];
      const int d = input_dims[2];
      const int h = input_dims[3];
      const int w = input_dims[4];

      auto* output_data =
          output->mutable_data<T>({n, c, d, h, w}, ctx.GetPlace());

      const int cdhw = c * d * h * w;
      int pixelNum = n * cdhw;
      int grid_dim = (pixelNum + 512 - 1) / 512;
      grid_dim = grid_dim > 8 ? 8 : grid_dim;

      KeAffinityPropagate3DFw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          output_data, input_data, gate_weight_data, kernel_size, n, cdhw, d, h,
          w);
    }
  }
};

template <typename T>
class AffinityPropagateGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("X");
    auto* gate_weight = ctx.Input<Tensor>("GateWeight");
    auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* gate_weight_grad =
        ctx.Output<Tensor>(framework::GradVarName("GateWeight"));

    const int kernel_size = ctx.Attr<int>("kernel_size");

    auto& device_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    math::SetConstant<platform::CUDADeviceContext, T> zero;

    const T* input_data = input->data<T>();
    const T* gate_weight_data = gate_weight->data<T>();
    const T* output_grad_data = output_grad->data<T>();

    auto input_dims = input->dims();
    if (input_dims.size() == 4) {
      const int n = input_dims[0];
      const int c = input_dims[1];
      const int h = input_dims[2];
      const int w = input_dims[3];
      const int gate_c = gate_weight->dims()[1];

      auto* input_grad_data =
          input_grad->mutable_data<T>({n, c, h, w}, ctx.GetPlace());
      zero(device_ctx, input_grad, static_cast<T>(0.0));
      auto* gate_weight_grad_data =
          gate_weight_grad->mutable_data<T>({n, gate_c, h, w}, ctx.GetPlace());
      zero(device_ctx, gate_weight_grad, static_cast<T>(0.0));

      const int chw = c * h * w;
      int pixelNum = n * chw;
      int grid_dim = (pixelNum + 512 - 1) / 512;
      grid_dim = grid_dim > 8 ? 8 : grid_dim;

      KeAffinityPropagate2DBw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_grad_data, gate_weight_grad_data, output_grad_data, input_data,
          gate_weight_data, kernel_size, n, chw, h, w);
    }
    if (input_dims.size() == 5) {
      const int n = input_dims[0];
      const int c = input_dims[1];
      const int d = input_dims[2];
      const int h = input_dims[3];
      const int w = input_dims[4];
      const int gate_c = gate_weight->dims()[1];

      auto* input_grad_data =
          input_grad->mutable_data<T>({n, c, d, h, w}, ctx.GetPlace());
      zero(device_ctx, input_grad, static_cast<T>(0.0));
      auto* gate_weight_grad_data = gate_weight_grad->mutable_data<T>(
          {n, gate_c, d, h, w}, ctx.GetPlace());
      zero(device_ctx, gate_weight_grad, static_cast<T>(0.0));

      const int cdhw = c * d * h * w;
      int pixelNum = n * cdhw;
      int grid_dim = (pixelNum + 512 - 1) / 512;
      grid_dim = grid_dim > 8 ? 8 : grid_dim;

      KeAffinityPropagate3DBw<
          T><<<grid_dim, 512, 0, ctx.cuda_device_context().stream()>>>(
          input_grad_data, gate_weight_grad_data, output_grad_data, input_data,
          gate_weight_data, kernel_size, n, cdhw, d, h, w);
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
