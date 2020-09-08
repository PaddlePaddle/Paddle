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

#pragma once

#ifdef __NVCC__

#include <cub/cub.cuh>
#include <limits>
#include <set>
#include <string>
#include <typeinfo>
#include <vector>
#include "paddle/fluid/framework/ddim.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace operators {

namespace {  // NOLINT
template <typename K, typename V>
using KeyValuePair = cub::KeyValuePair<K, V>;
using Tensor = framework::Tensor;

}  // end namespace

#define FIXED_BLOCK_DIM_CASE_BASE(log2_block_dim, ...)  \
  case (1 << (log2_block_dim)): {                       \
    constexpr auto kBlockDim = (1 << (log2_block_dim)); \
    __VA_ARGS__;                                        \
  } break

#define FIXED_BLOCK_DIM_CASE(...)               \
  FIXED_BLOCK_DIM_CASE_BASE(10, ##__VA_ARGS__); \
  FIXED_BLOCK_DIM_CASE_BASE(9, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(8, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(7, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(6, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(5, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(4, ##__VA_ARGS__);  \
  FIXED_BLOCK_DIM_CASE_BASE(3, ##__VA_ARGS__);

template <typename T, typename OutT, class Reducer, size_t BlockDim>
__global__ void ReduceCUDAKernel(const int height, const int width,
                                 const Reducer reducer, const T init,
                                 const T* in, OutT* out) {
  typedef cub::BlockReduce<T, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  for (int idx = blockIdx.x; idx < height; idx += gridDim.x) {
    T tmp_v = init;
    for (int k = threadIdx.x; k < width; k += blockDim.x) {
      tmp_v = reducer(in[idx * width + k], tmp_v);
    }
    tmp_v = BlockReduce(temp_storage).Reduce(tmp_v, reducer);
    if (threadIdx.x == 0) {
      out[idx] = static_cast<OutT>(tmp_v);
    }
    __syncthreads();
  }
}

template <typename T, typename OutT, class Reducer>
void ComputeFullReduce(const platform::CUDADeviceContext& ctx,
                       const Tensor& input, Tensor* output, const int num_rows,
                       const int num_cols) {
  auto cu_stream = ctx.stream();
  auto ComputeBlockSize = [](int col) {
    if (col > 512)
      return 1024;
    else if (col > 256)
      return 512;
    else if (col > 128)
      return 256;
    else if (col > 64)
      return 128;
    else if (col > 32)
      return 64;
    else if (col > 16)
      return 32;
    else if (col > 8)
      return 16;
    else
      return 8;
  };

  int max_grid_dimx = ctx.GetCUDAMaxGridDimSize().x;
  int grid_size = num_rows < max_grid_dimx ? num_rows : max_grid_dimx;

  const T* in_data = input.data<T>();
  OutT* out_data = output->mutable_data<OutT>(ctx.GetPlace());

  if (typeid(Reducer) == typeid(cub::Max)) {
    switch (ComputeBlockSize(num_cols)) {
      FIXED_BLOCK_DIM_CASE(
          ReduceCUDAKernel<T, OutT, Reducer,
                           kBlockDim><<<grid_size, kBlockDim, 0, cu_stream>>>(
              num_rows, num_cols, Reducer(), std::numeric_limits<T>::lowest(),
              in_data, out_data));
    }
  } else {
    switch (ComputeBlockSize(num_cols)) {
      FIXED_BLOCK_DIM_CASE(
          ReduceCUDAKernel<T, OutT, Reducer,
                           kBlockDim><<<grid_size, kBlockDim, 0, cu_stream>>>(
              num_rows, num_cols, Reducer(), std::numeric_limits<T>::max(),
              in_data, out_data));
    }
  }
}

template <typename T, typename Functor>
struct ReduceKernelCudaFunctor {
  const Tensor* input;
  Tensor* output;
  std::vector<int> dims;
  bool keep_dim;
  bool reduce_all;
  const framework::ExecutionContext& context;
  ReduceKernelCudaFunctor(const Tensor* input, Tensor* output,
                          const std::vector<int>& dims, bool keep_dim,
                          bool reduce_all,
                          const framework::ExecutionContext& context)
      : input(input),
        output(output),
        dims(dims),
        keep_dim(keep_dim),
        reduce_all(reduce_all),
        context(context) {}

  template <typename OutT>
  void apply() const {
    output->mutable_data<OutT>(context.GetPlace());
    if (reduce_all) {
    } else {
      int ndim = input->dims().size();
      int rdim = dims.size();
    }
  }
};

template <typename T, class Reducer>
class ReduceMinMaxCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    bool reduce_all = ctx.Attr<bool>("reduce_all");
    auto* output = ctx.Output<Tensor>("Out");
    auto dims = ctx.Attr<std::vector<int>>("dim");
    bool keep_dim = ctx.Attr<bool>("keep_dim");
    int out_dtype = ctx.Attr<int>("out_dtype");
    framework::proto::VarType::Type cast_out_dtype;
    auto* input = ctx.Input<Tensor>("X");

    auto in_dims = input->dims();
    const auto& input_dim_size = input->dims().size();
    for (size_t i = 0; i < dims.size(); i++) {
      if (dims[i] < 0) dims[i] += input_dim_size;
    }

    std::set<int> dims_set(dims.begin(), dims.end());
    bool full_dim = true;
    for (auto i = 0; i < input_dim_size; i++) {
      if (dims_set.find(i) == dims_set.end()) {
        full_dim = false;
        break;
      }
    }

    reduce_all = (reduce_all || full_dim);

    std::vector<int> trans;
    int height = 1;
    int width = 1;

    if (reduce_all) {
      width = input->numel();
      const auto& dev_ctx = ctx.cuda_device_context();
      auto real_output_dims = output->dims();
      ComputeFullReduce<T, T, Reducer>(dev_ctx, *input, output, height, width);
      // get the real shape
      // resize to the real output shape.
      output->Resize(real_output_dims);
    } else {
      for (int i = 0; i < input_dim_size; i++) {
        if (dims_set.find(i) == dims_set.end()) {
          trans.push_back(i);
          height *= in_dims[i];
        } else {
          width *= in_dims[i];
        }
      }
      trans.insert(trans.end(), dims_set.begin(), dims_set.end());

      Tensor trans_input;
      framework::DDim trans_dims(in_dims);
      for (int i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
      }
      T* trans_inp_data =
          trans_input.mutable_data<T>(trans_dims, ctx.GetPlace());
      const auto& dev_ctx = ctx.cuda_device_context();
      const T* input_data = input->data<T>();
      int ndims = trans.size();
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, *input,
                                                   &trans_input, trans);
      auto real_output_dims = output->dims();

      ComputeFullReduce<T, T, Reducer>(dev_ctx, trans_input, output, height,
                                       width);
      // get the real shape
      // resize to the real output shape.
      output->Resize(real_output_dims);
    }
  }
};

#endif

}  // namespace operators
}  // namespace paddle
