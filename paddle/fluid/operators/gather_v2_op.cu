/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/operators/gather.cu.h"
#include "paddle/fluid/operators/gather_op.h"
#include "paddle/fluid/operators/gather_v2_op.h"
#include "paddle/fluid/operators/scatter.cu.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

template <typename T, typename U>
__global__ void GatherGPUKernel(const T* input, const U* index, T* out,
                                int outer_dim_size, int inner_dim_size,
                                int index_dim_size, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    int inner_dim_index = idx / (outer_dim_size * index_dim_size);
    int out_dim_index = idx % outer_dim_size;
    int input_dim_index = idx / outer_dim_size;
    int input_index = inner_dim_index * (outer_dim_size * index_dim_size) +
                      index[input_dim_index] * outer_dim_size + out_dim_index;
    out[idx] = input[0];
  }
}

template <typename T, typename U>
__global__ void GatherGradGPUKernel(const T* input, const U* index, T* out,
                                    int outer_dim_size, int inner_dim_size,
                                    int index_dim_size, int size) {
  int idx = blockDim.x * blockIdx.x + threadIdx.x;
  for (; idx < size; idx += blockDim.x * gridDim.x) {
    int inner_dim_index = idx / (outer_dim_size * index_dim_size);
    int out_dim_index = idx % outer_dim_size;
    int input_dim_index = idx / outer_dim_size;
    int out_index = inner_dim_index * (outer_dim_size * index_dim_size) +
                    index[input_dim_index] * outer_dim_size + out_dim_index;
    paddle::platform::CudaAtomicAdd(out + out_index, *(input + idx));
  }
}

template <typename T, typename U, typename V>
void GatherV2CUDAFunction(const Tensor* input, const Tensor* index,
                          const Tensor* axis, Tensor* out,
                          const paddle::platform::Place& place,
                          const framework::ExecutionContext& ctx) {
  int axis_size = axis->numel();
  int index_size = index->numel();
  int input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();
  auto* index_data = index->data<U>();

  if (input->numel() == 0) return;
  PADDLE_ENFORCE_EQ(axis_size, 1,
                    platform::errors::InvalidArgument(
                        "Axis size should be 1, but received %d", axis_size));
  Tensor cpu_axis;
  framework::TensorCopy(*axis, platform::CPUPlace(), &cpu_axis);
  int axis_index = cpu_axis.data<V>()[0];
  int index_dim_size = input_dim[axis_index];
  PADDLE_ENFORCE_LE(
      index_size, index_dim_size,
      platform::errors::InvalidArgument(
          "The size that index should be less equal than the dim size of "
          "input,"
          "but received index size:%d, the dim size of input %d.",
          axis_size, index_dim_size));

  int inner_dim_size = 1;
  int outer_dim_size = 1;
  std::vector<int> out_dim_vec;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  out_dim_vec.push_back(index_size);
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
    out_dim_vec.push_back(input_dim[i]);
  }
  auto out_dim = framework::make_ddim(out_dim_vec);

  out->Resize(out_dim);
  auto* out_data = out->mutable_data<T>(place);
  int out_size = out->numel();

  int threads = 512;
  int grid = (out_size + threads - 1) / threads;
  auto stream = ctx.cuda_device_context().stream();
  GatherGPUKernel<T, U><<<grid, threads, 0, stream>>>(
      input_data, index_data, out_data, outer_dim_size, inner_dim_size,
      index_dim_size, out_size);
}

template <typename T, typename U, typename V>
void GatherV2GradCUDAFunction(const Tensor* input, const Tensor* index,
                              const Tensor* axis, Tensor* out,
                              const paddle::platform::Place& place,
                              const framework::ExecutionContext& ctx) {
  auto* axis_data = axis->data<V>();
  auto* index_data = index->data<U>();

  int axis_size = axis->numel();
  int index_size = index->numel();
  int input_size = input->numel();
  auto input_dim = input->dims();
  auto* input_data = input->data<T>();

  if (input->numel() == 0) return;
  PADDLE_ENFORCE_EQ(axis_size, 1,
                    platform::errors::InvalidArgument(
                        "Axis size should be 1, but received %d", axis_size));
  Tensor cpu_axis;
  framework::TensorCopy(*axis, platform::CPUPlace(), &cpu_axis);
  int axis_index = cpu_axis.data<V>()[0];
  int index_dim_size = input_dim[axis_index];
  PADDLE_ENFORCE_LE(
      index_size, index_dim_size,
      platform::errors::InvalidArgument(
          "The size that index should be less equal than the dim size of "
          "input,"
          "but received index size:%d, the dim size of input %d.",
          axis_size, index_dim_size));

  int inner_dim_size = 1;
  int outer_dim_size = 1;

  for (int i = 0; i < axis_index; i++) {
    inner_dim_size *= input_dim[i];
  }
  for (int i = axis_index + 1; i < input_dim.size(); i++) {
    outer_dim_size *= input_dim[i];
  }

  auto* out_data = out->mutable_data<T>(place);
  auto* dev_ctx = platform::DeviceContextPool::Instance().Get(place);
  operators::math::set_constant(*dev_ctx, out, 0.0);

  int threads = 512;
  int grid = (input_size + threads - 1) / threads;
  auto stream = ctx.cuda_device_context().stream();
  GatherGradGPUKernel<T, U><<<grid, threads, 0, stream>>>(
      input_data, index_data, out_data, outer_dim_size, inner_dim_size,
      index_dim_size, input_size);
}

template <typename DeviceContext, typename T>
class GatherV2OpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("X");
    const Tensor* index = ctx.Input<Tensor>("Index");
    const Tensor* axis = ctx.Input<Tensor>("Axis");
    Tensor* out = ctx.Output<Tensor>("Y");

    const auto& index_type = index->type();
    const auto& axis_type = axis->type();
    auto place = ctx.GetPlace();
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2CUDAFunction<T, int32_t, int32_t>(input, index, axis, out, place,
                                                ctx);
    }
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2CUDAFunction<T, int32_t, int64_t>(input, index, axis, out, place,
                                                ctx);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2CUDAFunction<T, int64_t, int32_t>(input, index, axis, out, place,
                                                ctx);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2CUDAFunction<T, int64_t, int64_t>(input, index, axis, out, place,
                                                ctx);
    }
  }
};

template <typename DeviceContext, typename T>
class GatherV2GradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* input = ctx.Input<Tensor>("X");
    const Tensor* index = ctx.Input<Tensor>("Index");
    const Tensor* axis = ctx.Input<Tensor>("Axis");
    Tensor* out = ctx.Output<Tensor>("Y");

    const auto& index_type = index->type();
    const auto& axis_type = axis->type();
    auto place = ctx.GetPlace();
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2GradCUDAFunction<T, int32_t, int32_t>(input, index, axis, out,
                                                    place, ctx);
    }
    if (index_type == framework::proto::VarType::INT32 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2GradCUDAFunction<T, int32_t, int64_t>(input, index, axis, out,
                                                    place, ctx);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT32) {
      GatherV2GradCUDAFunction<T, int64_t, int32_t>(input, index, axis, out,
                                                    place, ctx);
    }
    if (index_type == framework::proto::VarType::INT64 &&
        axis_type == framework::proto::VarType::INT64) {
      GatherV2GradCUDAFunction<T, int64_t, int64_t>(input, index, axis, out,
                                                    place, ctx);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(gather_v2, ops::GatherV2OpCUDAKernel<CUDA, float>,
                        ops::GatherV2OpCUDAKernel<CUDA, double>,
                        ops::GatherV2OpCUDAKernel<CUDA, int64_t>,
                        ops::GatherV2OpCUDAKernel<CUDA, int>,
                        ops::GatherV2OpCUDAKernel<CUDA, bool>,
                        ops::GatherV2OpCUDAKernel<CUDA, plat::float16>);

REGISTER_OP_CUDA_KERNEL(gather_v2_grad,
                        ops::GatherV2GradOpCUDAKernel<CUDA, float>,
                        ops::GatherV2GradOpCUDAKernel<CUDA, double>,
                        ops::GatherV2GradOpCUDAKernel<CUDA, int64_t>,
                        ops::GatherV2GradOpCUDAKernel<CUDA, int>,
                        ops::GatherV2GradOpCUDAKernel<CUDA, plat::float16>);
