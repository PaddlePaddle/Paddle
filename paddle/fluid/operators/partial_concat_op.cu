/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <paddle/fluid/platform/device_context.h>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/partial_concat_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

#define CEIL_DIV(x, y) (((x) + (y)-1) / (y))

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <class T>
__global__ void ConcatPartialCUDAKernel(T **in, T *out, int64_t all_length,
                                        int64_t in_batch_len,
                                        int64_t start_index,
                                        int64_t out_batch_len,
                                        int64_t part_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < all_length) {
    int64_t bs_id = id / out_batch_len;
    int64_t bs_index = id % out_batch_len;
    int64_t var_id = bs_index / part_length;
    int64_t part_index = bs_index % part_length;
    int64_t in_id = start_index + part_index;
    const T *tmp = in[var_id];
    out[id] = tmp[bs_id * in_batch_len + in_id];
    id += blockDim.x * gridDim.x;
  }
}

template <class T>
__global__ void ConcatPartialGradCUDAKernel(
    T **in, const T *out, int64_t all_length, int64_t in_batch_len,
    int64_t start_index, int64_t out_batch_len, int64_t part_length) {
  int id = blockIdx.x * blockDim.x + threadIdx.x;
  while (id < all_length) {
    int64_t bs_id = id / out_batch_len;
    int64_t bs_index = id % out_batch_len;
    int64_t var_id = bs_index / part_length;
    int64_t part_index = bs_index % part_length;
    int64_t in_id = start_index + part_index;
    T *tmp = in[var_id];
    tmp[bs_id * in_batch_len + in_id] = out[id];
    id += blockDim.x * gridDim.x;
  }
}

template <typename T>
class PartialConcatOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_vars = ctx.MultiInput<Tensor>("X");
    Tensor *out = ctx.Output<Tensor>("Out");
    PADDLE_ENFORCE_EQ(in_vars[0] != nullptr, true,
                      platform::errors::InvalidArgument(
                          "The input of partial concat should not be null."));

    auto input_dim = in_vars[0]->dims();
    PADDLE_ENFORCE_EQ(input_dim.size(), 2,
                      platform::errors::InvalidArgument(
                          "Only supports 2-D array with batch size in the 1st "
                          "dimension and data in the 2nd."));
    auto in_size = input_dim[1];
    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);

    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) {
      partial_len = in_size - start_index;
    }

    int in_num = in_vars.size();
    int batch_size = input_dim[0];
    int out_batch_len = partial_len * in_num;
    int all_length = batch_size * out_batch_len;

    constexpr size_t theory_sm_threads = 1024;
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    auto sm_count = max_threads / theory_sm_threads;
    size_t tile_size = 0;
    int grids;
    int blocks;
    auto ComputeKernelParameter = [&](size_t length) {
      if (length >= max_threads)
        tile_size = 1024;
      else if (length < max_threads && length > sm_count * 128)
        tile_size = 512;
      else if (length <= sm_count * 128)
        tile_size = 256;
      grids = CEIL_DIV(length, tile_size);
      blocks = tile_size;
    };

    auto place = ctx.GetPlace();
    T *out_data = out->mutable_data<T>(place);

    std::vector<const T *> in_data;
    for (int i = 0; i < in_num; ++i)
      in_data.emplace_back(in_vars[i]->data<T>());

    auto tmp_in_array = memory::Alloc(dev_ctx, in_data.size() * sizeof(T *));
    memory::Copy(dev_ctx.GetPlace(), tmp_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(in_data.data()),
                 in_data.size() * sizeof(T *), dev_ctx.stream());

    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());
    ComputeKernelParameter(all_length);
    ConcatPartialCUDAKernel<T><<<grids, blocks, 0, stream>>>(
        in_array_data, out->data<T>(), all_length, in_size, start_index,
        out_batch_len, partial_len);
  }
};

template <typename T>
class PartialConcatGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *out_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto ins = ctx.MultiInput<LoDTensor>("X");
    auto outs = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    PADDLE_ENFORCE_EQ(ins[0] != nullptr, true,
                      platform::errors::InvalidArgument(
                          "The input of partial concat should not be null."));
    // all parameters
    auto batch_size = ins[0]->dims()[0];
    auto in_size = ins[0]->dims()[1];
    // may be negative
    auto start_index = ctx.Attr<int>("start_index");
    start_index = ComputeStartIndex(start_index, in_size);
    auto partial_len = ctx.Attr<int>("length");
    if (partial_len < 0) partial_len = in_size - start_index;

    auto in_num = ins.size();
    auto grad_batch_len = partial_len * in_num;
    auto all_length = grad_batch_len * batch_size;
    // initialize
    auto &place = *ctx.template device_context<platform::CUDADeviceContext>()
                       .eigen_device();
    for (size_t i = 0; i < outs.size(); ++i) {
      outs[i]->mutable_data<T>(ctx.GetPlace());
      auto dxt = framework::EigenVector<T>::Flatten(*outs[i]);
      dxt.device(place) = dxt.constant(static_cast<T>(0));
    }

    constexpr size_t theory_sm_threads = 1024;
    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    auto stream = dev_ctx.stream();
    auto max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    auto sm_count = max_threads / theory_sm_threads;
    size_t tile_size = 0;
    int grids;
    int blocks;
    auto ComputeKernelParameter = [&](size_t length) {
      if (length >= max_threads)
        tile_size = 1024;
      else if (length < max_threads && length > sm_count * 128)
        tile_size = 512;
      else if (length <= sm_count * 128)
        tile_size = 256;
      grids = CEIL_DIV(length, tile_size);
      blocks = tile_size;
    };

    std::vector<const T *> out_data;
    for (size_t i = 0; i < in_num; ++i) {
      out_data.emplace_back(outs[i]->data<T>());
    }
    auto tmp_out_array = memory::Alloc(dev_ctx, out_data.size() * sizeof(T *));

    memory::Copy(dev_ctx.GetPlace(), tmp_out_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(out_data.data()),
                 out_data.size() * sizeof(T *), dev_ctx.stream());

    T **out_grad_data = reinterpret_cast<T **>(tmp_out_array->ptr());
    ComputeKernelParameter(all_length);
    ConcatPartialGradCUDAKernel<T><<<grids, blocks, 0, stream>>>(
        out_grad_data, out_grad->data<T>(), all_length, in_size, start_index,
        grad_batch_len, partial_len);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(partial_concat, ops::PartialConcatOpCUDAKernel<float>,
                        ops::PartialConcatOpCUDAKernel<double>,
                        ops::PartialConcatOpCUDAKernel<int>,
                        ops::PartialConcatOpCUDAKernel<int64_t>,
                        ops::PartialConcatOpCUDAKernel<plat::float16>);

REGISTER_OP_CUDA_KERNEL(partial_concat_grad,
                        ops::PartialConcatGradOpCUDAKernel<float>,
                        ops::PartialConcatGradOpCUDAKernel<double>,
                        ops::PartialConcatGradOpCUDAKernel<int>,
                        ops::PartialConcatGradOpCUDAKernel<int64_t>,
                        ops::PartialConcatGradOpCUDAKernel<plat::float16>);
