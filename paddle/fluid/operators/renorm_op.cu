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

#include "paddle/fluid/operators/renorm_op.h"

#include <algorithm>
#include <cstdio>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/reduce_ops/reduce_op.cu.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"

namespace paddle {
namespace operators {

__device__ __forceinline__ float inline_pow(float base, float exponent) {
  return pow(base, exponent);
}

__device__ __forceinline__ double inline_pow(double base, double exponent) {
  return pow(base, exponent);
}

__device__ __forceinline__ float inline_abs(float x) { return abs(x); }
__device__ __forceinline__ double inline_abs(double x) { return abs(x); }

template <typename Tx, typename Ty = Tx>
struct UnsignedPowFunctor {
  HOSTDEVICE explicit inline UnsignedPowFunctor(float porder) {
    this->porder = porder;
  }
  HOSTDEVICE inline Ty operator()(const Tx x) const {
    return static_cast<Ty>(inline_pow(inline_abs(x), static_cast<Tx>(porder)));
  }
  float porder;
};

template <typename T>
__global__ void RenormKernelFunc3(int64_t size, T* dim_value, float p,
                                  float max_norm) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  if (i < size) {
    T temp = pow(dim_value[i], (T)(1.0 / p));
    dim_value[i] = 1.0;
    if (temp > max_norm) dim_value[i] = max_norm / temp;
  }
}

template <typename T>
__global__ void RenormKernelFunc4(const T* x_data, T* out_data, int64_t size,
                                  T* dim_value, int64_t dimension_each,
                                  int64_t dim_divisor) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  auto dim_index = i / dim_divisor % dimension_each;
  if (i < size) {
    if (dim_value[dim_index] < 1.0)
      out_data[i] = dim_value[dim_index] * x_data[i];
    else
      out_data[i] = x_data[i];
  }
}

template <typename T>
__global__ void RenormGradKernelFunc1(const T* x_data, const T* dout_data,
                                      T* pow_value, T* mul_value, int64_t size,
                                      int64_t dimension_each, float p,
                                      int64_t dim_divisor) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  auto dim_index = i / dim_divisor % dimension_each;
  if (i < size) {
    pow_value[i] = pow(abs(x_data[i]), (T)p);
    mul_value[i] = x_data[i] * dout_data[i];
  }
}

template <typename T>
__global__ void RenormGradKernelFunc2(const T* x_data, const T* dout_data,
                                      T* dx_data, int64_t size, T* dim_value,
                                      T* dim_power_sum, T* weight_derivative,
                                      int64_t dimension_each, float p,
                                      float max_norm, int64_t dim_divisor) {
  int64_t i = ((int64_t)blockIdx.x) * blockDim.x + threadIdx.x;
  auto dim_index = i / dim_divisor % dimension_each;
  if (i < dimension_each) {
    dim_power_sum[i] = 0;
    auto temp = pow(dim_value[i], (T)(1.0 / p));
    if (temp > max_norm) {
      dim_power_sum[i] = pow(dim_value[i], (T)(-1.0 - 1.0 / p)) * -1 * max_norm;
      dim_value[i] = max_norm / temp;
    } else {
      dim_value[i] = 1.0;
    }
  }
  __syncthreads();
  if (i < size) {
    dx_data[i] = dim_value[dim_index] * dout_data[i];
    dx_data[i] = dx_data[i] +
                 weight_derivative[dim_index] * dim_power_sum[dim_index] *
                     pow(abs(x_data[i]), T(p - 1.0)) *
                     (x_data[i] >= 0 ? 1 : -1);
  }
}

template <typename T>
class CUDARenormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const Tensor* x = context.Input<Tensor>("X");
    Tensor* out = context.Output<Tensor>("Out");
    auto numel = x->numel();
    const T* x_data = x->data<T>();
    auto input_dims = x->dims();
    float max_norm = context.Attr<float>("max_norm");
    float p = context.Attr<float>("p");
    int dim = context.Attr<int>("axis");
    auto dimension_each = input_dims[dim];
    auto dim_size = input_dims.size();
    framework::Tensor pow_value, dim_value;
    int64_t dim_divisor = 1, pre_mul = 1;
    for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];
    for (int i = 0; i < dim; i++) pre_mul *= input_dims[i];
    pow_value.Resize(
        framework::make_ddim({pre_mul, dimension_each, dim_divisor}));
    dim_value.Resize(framework::make_ddim({dimension_each}));
    pow_value.mutable_data<T>(context.GetPlace());
    out->Resize(framework::make_ddim(framework::vectorize(input_dims)));
    T* out_data = out->mutable_data<T>(context.GetPlace());
    auto stream = context.cuda_device_context().stream();
    int block = std::min(numel, static_cast<int64_t>(256));
    using MT = typename details::MPTypeTrait<T>::Type;
    int grid = (numel + block - 1) / block;

    int block2 = std::min(dimension_each, static_cast<int64_t>(256));
    int grid2 = (dimension_each + block2 - 1) / block2;
    std::vector<const framework::Tensor*> ins = {x};
    std::vector<framework::Tensor*> outs = {&pow_value};
    auto func = UnsignedPowFunctor<MT, T>(p);
    const auto& cuda_ctx =
        context.template device_context<platform::CUDADeviceContext>();

    paddle::operators::LaunchSameDimsElementwiseCudaKernel<
        ElementwiseType::kUnary, MT, T, UnsignedPowFunctor<MT, T>>(
        cuda_ctx, ins, &outs, func);
    std::vector<int> reduce_axis = {0, 2};
    TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        pow_value, &dim_value, kps::IdentityFunctor<T>(), reduce_axis, stream);
    RenormKernelFunc3<T><<<grid2, block2, 0, stream>>>(
        numel, dim_value.mutable_data<T>(context.GetPlace()), p, max_norm);
    RenormKernelFunc4<T><<<grid, block, 0, stream>>>(
        x_data, out_data, numel, dim_value.mutable_data<T>(context.GetPlace()),
        dimension_each, dim_divisor);
    // platform::GpuStreamSync(stream);
  }
};

template <typename T>
class CUDAGradRenormKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));

    auto numel = d_out->numel();
    const T* dout_data = d_out->data<T>();
    const T* x_data = x->data<T>();
    auto input_dims = x->dims();
    float max_norm = ctx.Attr<float>("max_norm");
    float p = ctx.Attr<float>("p");
    int dim = ctx.Attr<int>("axis");
    auto dimension_each = input_dims[dim];
    auto dim_size = input_dims.size();
    int64_t dim_divisor = 1, pre_mul = 1;
    for (int i = dim + 1; i < dim_size; i++) dim_divisor *= input_dims[i];
    for (int i = 0; i < dim; i++) pre_mul *= input_dims[i];
    d_x->Resize(framework::make_ddim(framework::vectorize(input_dims)));
    T* dx_data = d_x->mutable_data<T>(ctx.GetPlace());
    framework::Tensor pow_value, mul_value, dim_value, dim_power_sum,
        weight_derivative;
    pow_value.Resize(
        framework::make_ddim({pre_mul, dimension_each, dim_divisor}));
    mul_value.Resize(
        framework::make_ddim({pre_mul, dimension_each, dim_divisor}));
    dim_value.Resize(framework::make_ddim({dimension_each}));
    dim_power_sum.Resize(framework::make_ddim({dimension_each}));
    weight_derivative.Resize(framework::make_ddim({dimension_each}));
    auto stream = ctx.cuda_device_context().stream();
    int block = std::min(numel, static_cast<int64_t>(256));
    int grid = (numel + block - 1) / block;
    pow_value.mutable_data<T>(ctx.GetPlace());
    mul_value.mutable_data<T>(ctx.GetPlace());
    dim_value.mutable_data<T>(ctx.GetPlace());
    dim_power_sum.mutable_data<T>(ctx.GetPlace());
    weight_derivative.mutable_data<T>(ctx.GetPlace());
    RenormGradKernelFunc1<T><<<grid, block, 0, stream>>>(
        x_data, dout_data, pow_value.mutable_data<T>(ctx.GetPlace()),
        mul_value.mutable_data<T>(ctx.GetPlace()), numel, dimension_each, p,
        dim_divisor);
    std::vector<int> reduce_axis = {0, 2};
    TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        pow_value, &dim_value, kps::IdentityFunctor<T>(), reduce_axis, stream);
    TensorReduceFunctorImpl<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
        mul_value, &weight_derivative, kps::IdentityFunctor<T>(), reduce_axis,
        stream);
    RenormGradKernelFunc2<T><<<grid, block, 0, stream>>>(
        x_data, dout_data, dx_data, numel,
        dim_value.mutable_data<T>(ctx.GetPlace()),
        dim_power_sum.mutable_data<T>(ctx.GetPlace()),
        weight_derivative.mutable_data<T>(ctx.GetPlace()), dimension_each, p,
        max_norm, dim_divisor);
    // platform::GpuStreamSync(stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(renorm, ops::CUDARenormKernel<float>,
                        ops::CUDARenormKernel<double>);

REGISTER_OP_CUDA_KERNEL(renorm_grad, ops::CUDAGradRenormKernel<float>,
                        ops::CUDAGradRenormKernel<double>);
