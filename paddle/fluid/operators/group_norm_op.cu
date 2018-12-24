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

#include <cub/cub.cuh>
#include "paddle/fluid/operators/group_norm_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void GroupNormForwardGetMeanAndVar(const T* x, int N, int C,
                                              int imsize, int groups,
                                              int group_size, T* mean, T* var) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_mean = 0, x_var = 0;
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val = x[(bid * C + ccid) * imsize + imid];
    x_mean += val;
    x_var += val * val;
  }
  x_mean /= number * imsize;
  x_var /= number * imsize;
  __shared__ T s_mem[2];
  if (threadIdx.x == 0) {
    s_mem[0] = s_mem[1] = 0;
  }
  __syncthreads();
  paddle::platform::CudaAtomicAdd(&s_mem[0], x_mean);
  paddle::platform::CudaAtomicAdd(&s_mem[1], x_var);
  __syncthreads();
  if (threadIdx.x == 0) {
    paddle::platform::CudaAtomicAdd(&mean[bid * groups + gid], s_mem[0]);
    paddle::platform::CudaAtomicAdd(&var[bid * groups + gid], s_mem[1]);
  }
}

template <typename T>
__global__ void GroupNormForward(const T* x, const T* mean, const T* var,
                                 const T* scale, const T* bias, int N, int C,
                                 int imsize, int groups, int group_size,
                                 T epsilon, T* y, T* real_var) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_mean = mean[bid * groups + gid];
  T x_var = var[bid * groups + gid];
  x_var = x_var - x_mean * x_mean;
  T var_inv = 1.0 / sqrt(x_var + epsilon);
  if (cid == 0 && threadIdx.x == 0) real_var[bid * groups + gid] = x_var;
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T val = x[(bid * C + ccid) * imsize + imid];
    val = (val - x_mean) * var_inv;
    if (scale) val *= scale[gid * group_size + cid];
    if (bias) val += bias[gid * group_size + cid];
    y[(bid * C + ccid) * imsize + imid] = val;
  }
}

template <typename T>
class GroupNormKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* bias = ctx.Input<Tensor>("Bias");
    auto* x = ctx.Input<Tensor>("X");

    auto* y = ctx.Output<Tensor>("Y");
    auto* mean = ctx.Output<Tensor>("Mean");
    auto* var = ctx.Output<Tensor>("Variance");
    const auto groups = ctx.Attr<int>("groups");

    const auto x_dims = x->dims();
    const int group_size = (x_dims[1] - 1) / groups + 1;

    y->mutable_data<T>(ctx.GetPlace());
    mean->mutable_data<T>(ctx.GetPlace());
    var->mutable_data<T>(ctx.GetPlace());
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    Tensor temp_var;
    temp_var.mutable_data<T>(var->dims(), ctx.GetPlace());

    set_zero(dev_ctx, mean, static_cast<T>(0));
    set_zero(dev_ctx, &temp_var, static_cast<T>(0));

    auto* x_data = x->data<T>();
    auto* y_data = y->data<T>();
    auto* mean_data = mean->data<T>();
    auto* var_data = var->data<T>();
    auto* temp_var_data = temp_var.data<T>();

    const T* scale_data = nullptr;
    if (scale) scale_data = scale->data<T>();
    const T* bias_data = nullptr;
    if (bias) bias_data = bias->data<T>();

    int imsize = x_dims[2] * x_dims[3];
    int block_size = std::min(512, imsize);
    dim3 grid(group_size, groups, x_dims[0]);
    dim3 threads(block_size, 1, 1);
    GroupNormForwardGetMeanAndVar<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, x_dims[0], x_dims[1], imsize, groups, group_size, mean_data,
        temp_var_data);
    GroupNormForward<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, mean_data, temp_var_data, scale_data, bias_data, x_dims[0],
        x_dims[1], imsize, groups, group_size, epsilon, y_data, var_data);
  }
};

template <typename T>
__global__ void GroupNormBackwardGetMeanAndVar(
    const T* x, const T* mean, const T* var, const T* scale, const T* d_y,
    int N, int C, int imsize, int groups, int group_size, T epsilon, T* d_x,
    T* d_mean, T* d_var, T* d_scale, T* d_bias) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_mean = mean[bid * groups + gid];
  T x_var = var[bid * groups + gid];
  T var_inv = 1.0 / sqrt(x_var + epsilon);
  T d_var_inv = 0, d_x_mean = 0;
  T d_mean_data = 0, d_var_data = 0, d_scale_data = 0, d_bias_data = 0;

  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T tmp = x[(bid * C + ccid) * imsize + imid];
    T val = (tmp - x_mean) * var_inv;
    T dval = d_y[(bid * C + ccid) * imsize + imid];
    if (d_bias) d_bias_data += dval;
    if (d_scale) d_scale_data += val * dval;
    if (scale) dval = dval * scale[ccid];
    d_var_data += (tmp - x_mean) * dval;
    T d_tmp = dval * var_inv;
    if (d_x) d_x[(bid * C + ccid) * imsize + imid] = d_tmp;
    d_mean_data -= d_tmp;
  }

  __shared__ T s_mem[4];
  if (threadIdx.x == 0) {
    s_mem[0] = s_mem[1] = 0;
    if (d_scale) s_mem[2] = 0;
    if (d_bias) s_mem[3] = 0;
  }
  __syncthreads();
  paddle::platform::CudaAtomicAdd(&s_mem[0], d_mean_data);
  paddle::platform::CudaAtomicAdd(&s_mem[1], d_var_data);
  if (d_scale) paddle::platform::CudaAtomicAdd(&s_mem[2], d_scale_data);
  if (d_bias) paddle::platform::CudaAtomicAdd(&s_mem[3], d_bias_data);
  __syncthreads();
  if (threadIdx.x == 0) {
    paddle::platform::CudaAtomicAdd(&d_mean[bid * groups + gid], s_mem[0]);
    paddle::platform::CudaAtomicAdd(&d_var[bid * groups + gid], s_mem[1]);
    if (d_scale) paddle::platform::CudaAtomicAdd(&d_scale[ccid], s_mem[2]);
    if (d_bias) paddle::platform::CudaAtomicAdd(&d_bias[ccid], s_mem[3]);
  }
}

template <typename T>
__global__ void GroupNormBackward(const T* x, const T* mean, const T* var,
                                  const T* d_mean, const T* d_var, int N, int C,
                                  int imsize, int groups, int group_size,
                                  T epsilon, T* d_x) {
  int gid = blockIdx.y;
  int cid = blockIdx.x;
  int bid = blockIdx.z;
  int number = min(group_size, static_cast<int>(C - gid * group_size));
  int ccid = gid * group_size + cid;
  if (ccid >= C) return;
  T x_mean = mean[bid * groups + gid];
  T x_var = var[bid * groups + gid];
  T d_x_mean = d_mean[bid * groups + gid];
  T d_var_inv = d_var[bid * groups + gid];

  T d_x_var =
      -1.0 / (2 * (x_var + epsilon) * sqrt(x_var + epsilon)) * d_var_inv;
  d_x_mean -= 2 * d_x_var * x_mean;
  d_x_var /= number * imsize;
  d_x_mean /= number * imsize;
  for (int imid = threadIdx.x; imid < imsize; imid += blockDim.x) {
    T tmp = x[(bid * C + ccid) * imsize + imid];
    if (d_x)
      d_x[(bid * C + ccid) * imsize + imid] += d_x_mean + tmp * 2 * d_x_var;
  }
}

template <typename T>
class GroupNormGradKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const float epsilon = ctx.Attr<float>("epsilon");
    auto* x = ctx.Input<Tensor>("X");
    auto* mean = ctx.Input<Tensor>("Mean");
    auto* var = ctx.Input<Tensor>("Variance");
    auto* scale = ctx.Input<Tensor>("Scale");
    auto* d_y = ctx.Input<Tensor>(framework::GradVarName("Y"));
    const auto groups = ctx.Attr<int>("groups");

    // init output
    auto* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    auto* d_scale = ctx.Output<Tensor>(framework::GradVarName("Scale"));
    auto* d_bias = ctx.Output<Tensor>(framework::GradVarName("Bias"));

    const auto& x_dims = x->dims();
    const int group_size = (x_dims[1] - 1) / groups + 1;

    T* d_x_data = nullptr;
    if (d_x) {
      d_x->mutable_data<T>(ctx.GetPlace());
      d_x_data = d_x->data<T>();
    }
    math::SetConstant<platform::CUDADeviceContext, T> set_zero;
    auto& dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();

    Tensor temp_var;
    temp_var.mutable_data<T>(var->dims(), ctx.GetPlace());
    set_zero(dev_ctx, &temp_var, static_cast<T>(0));
    T* temp_var_data = temp_var.data<T>();

    Tensor temp_mean;
    temp_mean.mutable_data<T>(var->dims(), ctx.GetPlace());
    set_zero(dev_ctx, &temp_mean, static_cast<T>(0));
    T* temp_mean_data = temp_mean.data<T>();

    auto* x_data = x->data<T>();
    auto* y_data = d_y->data<T>();
    auto* mean_data = mean->data<T>();
    auto* var_data = var->data<T>();
    T* d_scale_data = nullptr;
    if (d_scale) {
      d_scale->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_scale, static_cast<T>(0));
      d_scale_data = d_scale->data<T>();
    }
    T* d_bias_data = nullptr;
    if (d_bias) {
      d_bias->mutable_data<T>(ctx.GetPlace());
      set_zero(dev_ctx, d_bias, static_cast<T>(0));
      d_bias_data = d_bias->data<T>();
    }

    const T* scale_data = nullptr;
    if (scale) scale_data = scale->data<T>();

    int imsize = x_dims[2] * x_dims[3];
    int block_size = std::min(512, imsize);
    dim3 grid(group_size, groups, x_dims[0]);
    dim3 threads(block_size, 1, 1);
    GroupNormBackwardGetMeanAndVar<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, mean_data, var_data, scale_data, y_data, x_dims[0], x_dims[1],
        imsize, groups, group_size, epsilon, d_x_data, temp_mean_data,
        temp_var_data, d_scale_data, d_bias_data);
    GroupNormBackward<T><<<grid, threads, 0, dev_ctx.stream()>>>(
        x_data, mean_data, var_data, temp_mean_data, temp_var_data, x_dims[0],
        x_dims[1], imsize, groups, group_size, epsilon, d_x_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    group_norm,
    ops::GroupNormKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GroupNormKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    group_norm_grad,
    ops::GroupNormGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::GroupNormGradKernel<paddle::platform::CUDADeviceContext, double>);
