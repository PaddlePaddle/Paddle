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

#include <algorithm>
#include <string>
#include <vector>
#include "cub/cub.cuh"
#include "paddle/fluid/memory/malloc.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/sequence_pool_all_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;

template <typename T, int BlockDim>
__global__ void sequence_pool_all_kernel(T **input, const T pad_value,
                                         size_t **all_lod, const int batch_size,
                                         const size_t item_dim, T **output) {
  int tid = blockIdx.x;  // tensor index
  int bid = blockIdx.y;  // batch index
  if (bid >= batch_size) return;
  size_t start = all_lod[tid][bid];
  size_t end = all_lod[tid][bid + 1];

  int offset = end - start;
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ou_storage;

  for (int i = blockIdx.z; i < item_dim; i += gridDim.z) {
    if (offset == 0) {
      output[tid][bid * item_dim + i] = pad_value;
    } else {
      T ou = static_cast<T>(0);
      for (int j = threadIdx.x; j < offset; j += blockDim.x) {
        const int index = j * item_dim + i;
        ou += static_cast<T>(input[tid][item_dim * start + index]);
      }
      ou = BlockReduce(ou_storage).Reduce(ou, cub::Sum());
      __syncthreads();
      if (threadIdx.x == 0) {
        output[tid][bid * item_dim + i] = ou;
      }
    }
  }
}

template <typename T>
struct SumPoolGradFunctor {
  HOSTDEVICE void operator()(const T *out_grad, const size_t start,
                             const size_t end, const size_t item_dim,
                             T *in_grad, const int *index) {
    for (int tid = threadIdx.x; tid < item_dim; tid += blockDim.x) {
      for (int i = start; i < end; ++i) {
        in_grad[item_dim * i + tid] = out_grad[tid];
      }
    }
  }
};

template <typename T, typename Range_OP>
__global__ void sequence_pool_all_grad_kernel(Range_OP op, T **input,
                                              size_t **all_lod,
                                              const int batch_size,
                                              const size_t item_dim,
                                              T **output) {
  int tid = blockIdx.x;  // tensor index
  int bid = blockIdx.y;  // batch index
  if (bid >= batch_size) return;
  size_t start = all_lod[tid][bid];
  size_t end = all_lod[tid][bid + 1];

  op(&input[tid][bid * item_dim], start, end, item_dim, output[tid], nullptr);
}

template <typename DeviceContext, typename T>
class SequencePoolAllCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto in_vars = ctx.MultiInput<LoDTensor>("X");
    auto out_vars = ctx.MultiOutput<Tensor>("Out");
    std::string pooltype = ctx.Attr<std::string>("pooltype");
    PADDLE_ENFORCE_EQ(
        pooltype, "SUM",
        platform::errors::InvalidArgument(
            "Currently, it only supports SUM for sequence_pool_all op"));
    T pad_value = static_cast<T>(ctx.Attr<float>("pad_value"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int batch_size = in_vars[0]->lod()[0].size() - 1;

    auto in_num = in_vars.size();
    std::vector<const size_t *> lod_data;
    for (int i = 0; i < in_num; ++i) {
      auto dims = in_vars[i]->dims();
      auto &lod = in_vars[i]->lod();
      PADDLE_ENFORCE_EQ(lod.size(), 1,
                        platform::errors::InvalidArgument(
                            "Currently, it only supports lod_level = 1."));
      PADDLE_ENFORCE_EQ(
          batch_size,
          /*batch size = */ static_cast<int64_t>(lod[0].size() - 1),
          platform::errors::InvalidArgument(
              "There are error of lod mesage in inputs."));
      PADDLE_ENFORCE_GE(
          dims[0],
          /*batch size = */ static_cast<int64_t>(lod[0].size() - 1),
          platform::errors::InvalidArgument("The first dimension of Input(X) "
                                            "must be large than batch size."));

      lod_data.emplace_back(lod[0].CUDAData(dev_ctx.GetPlace()));
      dims[0] = batch_size;
      out_vars[i]->Resize({dims});
      out_vars[i]->mutable_data<T>(ctx.GetPlace());
    }
    const size_t item_dim = out_vars[0]->numel() / out_vars[0]->dims()[0];

    std::vector<const T *> in_data;
    std::vector<T *> out_data;
    for (int i = 0; i < in_num; ++i) {
      in_data.emplace_back(in_vars[i]->data<T>());
      out_data.emplace_back(out_vars[i]->data<T>());
    }
    auto tmp_in_array = memory::Alloc(dev_ctx, in_data.size() * sizeof(T *));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(in_data.data()),
                 in_data.size() * sizeof(T *), dev_ctx.stream());
    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());

    auto tmp_lod_array =
        memory::Alloc(dev_ctx, lod_data.size() * sizeof(size_t *));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_lod_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(lod_data.data()),
                 lod_data.size() * sizeof(size_t *), dev_ctx.stream());
    size_t **lod_array_data = reinterpret_cast<size_t **>(tmp_lod_array->ptr());

    auto tmp_out_array = memory::Alloc(dev_ctx, out_data.size() * sizeof(T *));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_out_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(out_data.data()),
                 out_data.size() * sizeof(T *), dev_ctx.stream());
    T **out_array_data = reinterpret_cast<T **>(tmp_out_array->ptr());

    const int block = 1024;
    dim3 threads(block, 1);
    dim3 grid(out_data.size(), std::max(batch_size, 1),
              std::max(static_cast<int>(item_dim), 1));

    sequence_pool_all_kernel<T, block><<<grid, threads, 0, dev_ctx.stream()>>>(
        in_array_data, pad_value, lod_array_data, batch_size, item_dim,
        out_array_data);
  }
};

template <typename T>
class SequencePoolAllGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto grad_vars = ctx.MultiInput<Tensor>(framework::GradVarName("Out"));
    auto in_vars = ctx.MultiInput<LoDTensor>("X");
    auto out_vars = ctx.MultiOutput<LoDTensor>(framework::GradVarName("X"));

    auto &dev_ctx = ctx.template device_context<platform::CUDADeviceContext>();
    int batch_size = in_vars[0]->lod()[0].size() - 1;
    auto in_num = in_vars.size();
    std::vector<const size_t *> lod_data;
    for (int i = 0; i < in_num; ++i) {
      auto dims = in_vars[i]->dims();
      auto &lod = in_vars[i]->lod();
      PADDLE_ENFORCE_EQ(lod.size(), 1,
                        platform::errors::InvalidArgument(
                            "Currently, it only supports lod_level = 1."));
      PADDLE_ENFORCE_EQ(
          batch_size,
          /*batch size = */ static_cast<int64_t>(lod[0].size() - 1),
          platform::errors::InvalidArgument(
              "There are error of lod mesage in inputs."));
      PADDLE_ENFORCE_GE(
          dims[0],
          /*batch size = */ static_cast<int64_t>(lod[0].size() - 1),
          platform::errors::InvalidArgument("The first dimension of Input(X) "
                                            "must be large than batch size."));
      lod_data.emplace_back(lod[0].CUDAData(dev_ctx.GetPlace()));

      out_vars[i]->mutable_data<T>(ctx.GetPlace());
    }
    const size_t item_dim = grad_vars[0]->numel() / grad_vars[0]->dims()[0];

    std::vector<const T *> in_data;
    std::vector<T *> out_data;
    for (int i = 0; i < in_num; ++i) {
      in_data.emplace_back(grad_vars[i]->data<T>());
      out_data.emplace_back(out_vars[i]->data<T>());
    }

    auto tmp_in_array = memory::Alloc(dev_ctx, in_data.size() * sizeof(T *));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_in_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(in_data.data()),
                 in_data.size() * sizeof(T *), dev_ctx.stream());
    T **in_array_data = reinterpret_cast<T **>(tmp_in_array->ptr());

    auto tmp_lod_array =
        memory::Alloc(dev_ctx, lod_data.size() * sizeof(size_t *));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_lod_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(lod_data.data()),
                 lod_data.size() * sizeof(size_t *), dev_ctx.stream());
    size_t **lod_array_data = reinterpret_cast<size_t **>(tmp_lod_array->ptr());

    auto tmp_out_array = memory::Alloc(dev_ctx, out_data.size() * sizeof(T *));
    memory::Copy(boost::get<platform::CUDAPlace>(dev_ctx.GetPlace()),
                 tmp_out_array->ptr(), platform::CPUPlace(),
                 reinterpret_cast<void *>(out_data.data()),
                 out_data.size() * sizeof(T *), dev_ctx.stream());
    T **out_array_data = reinterpret_cast<T **>(tmp_out_array->ptr());

    dim3 threads(std::min(1024, static_cast<int>(item_dim)), 1);
    dim3 grid(out_data.size(), std::max(batch_size, 1));

    sequence_pool_all_grad_kernel<
        T, SumPoolGradFunctor<T>><<<grid, threads, 0, dev_ctx.stream()>>>(
        SumPoolGradFunctor<T>(), in_array_data, lod_array_data, batch_size,
        item_dim, out_array_data);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(sequence_pool_all,
                        ops::SequencePoolAllCUDAKernel<GPUCtx, float>,
                        ops::SequencePoolAllCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(sequence_pool_all_grad,
                        ops::SequencePoolAllGradOpCUDAKernel<float>,
                        ops::SequencePoolAllGradOpCUDAKernel<double>);
