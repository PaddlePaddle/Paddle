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

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/kthvalue_op.h"
#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/fluid/operators/top_k_v2_op.h"
#ifdef __NVCC__
#include "cub/cub.cuh"
#endif
#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
#endif

namespace paddle {
namespace operators {

int getBlockSize(int col) {
  if (col > 512)
    return 1024;
  else if (col > 256 && col <= 512)
    return 512;
  else if (col > 128 && col <= 256)
    return 256;
  else if (col > 64 && col <= 128)
    return 128;
  else
    return 64;
}

template <typename T>
bool SortKthvalue(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor* input_tensor, const int64_t num_cols,
                  const int64_t num_rows, const int k,
                  framework::Tensor* out_tensor,
                  framework::Tensor* indices_tensor) {
  auto cu_stream = ctx.stream();
  framework::Tensor input_indices;
  const std::vector<int64_t> dims = {num_rows, num_cols};
  auto dim = framework::make_ddim(dims);
  input_indices.Resize(dim);
  input_indices.mutable_data<int64_t>(ctx.GetPlace());
  size_t temp_storage_bytes = -1;
  int block_size = getBlockSize(num_cols);
  unsigned int maxGridDimX = ctx.GetCUDAMaxGridDimSize()[0];
  unsigned int grid_size = num_rows < maxGridDimX
                               ? static_cast<unsigned int>(num_rows)
                               : maxGridDimX;
  InitIndex<int64_t><<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<int64_t>(), num_rows, num_cols);
  cub::CountingInputIterator<int64_t> counting_iter(0);
  cub::TransformInputIterator<int64_t, SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter, SegmentOffsetIter(num_cols));
  T* sorted_values_ptr;
  int64_t* sorted_indices_ptr;
  framework::Tensor temp_values, temp_indices;
  const T* input = input_tensor->data<T>();
  T* values = out_tensor->data<T>();
  int64_t* indices = indices_tensor->mutable_data<int64_t>(ctx.GetPlace());
  temp_values.Resize(dim);
  temp_indices.Resize(dim);
  sorted_values_ptr = temp_values.mutable_data<T>(ctx.GetPlace());
  sorted_indices_ptr = temp_indices.mutable_data<int64_t>(ctx.GetPlace());
  auto err = cub::DeviceSegmentedRadixSort::SortPairs(
      nullptr, temp_storage_bytes, input, sorted_values_ptr,
      input_indices.data<int64_t>(), sorted_indices_ptr, num_cols * num_rows,
      num_rows, segment_offsets_t, segment_offsets_t + 1, 0, sizeof(T) * 8,
      cu_stream);
#ifdef __HIPCC__
  if (err != hipSuccess) {
    LOG(ERROR) << "KthvalueOP failed as could not launch "
                  "hipcub::DeviceSegmentedRadixSort::SortPairs, status: "
               << hipGetErrorString(err);
    return false;
  }
#else
  if (err != cudaSuccess) {
    LOG(ERROR) << "KthvalueOP failed as could not launch "
                  "cub::DeviceSegmentedRadixSort::SortPairs, status: "
               << cudaGetErrorString(err);
    return false;
  }
#endif
  framework::Tensor temp_storage;
  temp_storage.mutable_data<uint8_t>(ctx.GetPlace(), temp_storage_bytes);

  err = cub::DeviceSegmentedRadixSort::SortPairs(
      temp_storage.data<uint8_t>(), temp_storage_bytes, input,
      sorted_values_ptr, input_indices.data<int64_t>(), sorted_indices_ptr,
      num_cols * num_rows, num_rows, segment_offsets_t, segment_offsets_t + 1,
      0, sizeof(T) * 8, cu_stream);
#ifdef __HIPCC__
  if (err != hipSuccess) {
    LOG(ERROR) << "KthvalueOP failed as could not launch "
                  "hipcub::DeviceSegmentedRadixSort::SortPairs, "
               << temp_storage_bytes << ", status: " << hipGetErrorString(err);
    return false;
  }
#else
  if (err != cudaSuccess) {
    LOG(ERROR) << "KthvalueOP failed as could not launch "
                  "cub::DeviceSegmentedRadixSort::SortPairs, "
               << temp_storage_bytes << ", status: " << cudaGetErrorString(err);
    return false;
  }
#endif
  auto& dev = *ctx.eigen_device();
  const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, k - 1};
  const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, 1};
  auto e_indices = framework::EigenMatrix<int64_t>::From(*indices_tensor, dim);
  auto e_tmp_indices = framework::EigenMatrix<int64_t>::From(
      static_cast<const framework::Tensor>(temp_indices));
  std::vector<int> odims = {static_cast<int>(num_rows), static_cast<int>(1)};
  dim = framework::make_ddim(odims);
  auto e_values = framework::EigenMatrix<T>::From(*out_tensor, dim);
  auto e_tmp_values = framework::EigenMatrix<T>::From(
      static_cast<const framework::Tensor>(temp_values));

  EigenSlice<std::decay_t<decltype(dev)>, int64_t, 2>::Eval(
      dev, e_indices, e_tmp_indices, slice_indices, slice_sizes);
  EigenSlice<std::decay_t<decltype(dev)>, T, 2>::Eval(
      dev, e_values, e_tmp_values, slice_indices, slice_sizes);
  return true;
}

template <typename DeviceContext, typename T>
class KthvalueOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(ctx.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* input = ctx.Input<framework::Tensor>("X");
    auto* output = ctx.Output<framework::Tensor>("Out");
    auto* indices = ctx.Output<framework::Tensor>("Indices");
    int k = static_cast<int>(ctx.Attr<int>("k"));
    int axis = static_cast<int>(ctx.Attr<int>("axis"));
    bool keepdim = static_cast<bool>(ctx.Attr<bool>("keepdim"));
    const auto& in_dims = input->dims();
    if (axis < 0) axis += in_dims.size();
    auto out_dims = output->dims();
    const T* input_data = input->data<T>();
    T* output_data = output->mutable_data<T>(ctx.GetPlace());
    int64_t* indices_data = indices->mutable_data<int64_t>(ctx.GetPlace());

    if (axis == in_dims.size() - 1) {
      const int64_t& input_height = framework::product(
          framework::slice_ddim(in_dims, 0, in_dims.size() - 1));
      const int64_t& input_width = in_dims[in_dims.size() - 1];
      const auto& dev_ctx = ctx.cuda_device_context();
      PADDLE_ENFORCE_EQ(SortKthvalue<T>(dev_ctx, input, input_width,
                                        input_height, k, output, indices),
                        true, platform::errors::External(
                                  "KthvalueOP: Error when use cub sorting"));
      return;
    } else {
      std::vector<int> trans;
      for (int i = 0; i < axis; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(in_dims.size() - 1);
      for (int i = axis + 1; i < in_dims.size() - 1; i++) {
        trans.emplace_back(i);
      }
      trans.emplace_back(axis);
      if (!keepdim) {
        std::vector<int> tmp_out_shape;
        for (int i = 0; i < axis; i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        tmp_out_shape.emplace_back(1);
        for (int i = axis + 1; i < in_dims.size(); i++) {
          tmp_out_shape.emplace_back(in_dims[i]);
        }
        framework::DDim tmp_out_dims = framework::make_ddim(tmp_out_shape);
        output->Resize(tmp_out_dims);
        indices->Resize(tmp_out_dims);
      }
      framework::DDim trans_dims(in_dims);
      framework::DDim trans_out_dims(in_dims);
      for (int i = 0; i < trans.size(); i++) {
        trans_dims[i] = in_dims[trans[i]];
        trans_out_dims[i] = in_dims[trans[i]];
      }
      trans_out_dims[in_dims.size() - 1] = 1;
      framework::Tensor trans_input;
      trans_input.mutable_data<T>(trans_dims, ctx.GetPlace());
      int ndims = trans.size();
      const auto& dev_ctx = ctx.cuda_device_context();
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, *input,
                                                   &trans_input, trans);
      framework::Tensor trans_ind, trans_out;
      trans_ind.mutable_data<int64_t>(trans_out_dims, ctx.GetPlace());
      trans_out.mutable_data<T>(trans_out_dims, ctx.GetPlace());
      const int64_t input_height = framework::product(
          framework::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
      const int64_t input_width = trans_dims[trans_dims.size() - 1];
      PADDLE_ENFORCE_EQ(
          SortKthvalue<T>(dev_ctx, &trans_input, input_width, input_height, k,
                          &trans_out, &trans_ind),
          true,
          platform::errors::External("KthvalueOP: Error when use cub sorting"));
      TransCompute<platform::CUDADeviceContext, int64_t>(
          ndims, dev_ctx, trans_ind, indices, trans);
      TransCompute<platform::CUDADeviceContext, T>(ndims, dev_ctx, trans_out,
                                                   output, trans);
      if (!keepdim) {
        output->Resize(out_dims);
        indices->Resize(out_dims);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class KthvalueOpGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    PADDLE_ENFORCE_EQ(
        platform::is_gpu_place(context.GetPlace()), true,
        platform::errors::InvalidArgument(
            "It must use CUDAPlace, you must check your device set."));
    auto* x = context.Input<framework::Tensor>("X");
    auto* out_grad =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* indices = context.Input<framework::Tensor>("Indices");
    auto* x_grad =
        context.Output<framework::Tensor>(framework::GradVarName("X"));
    int axis = context.Attr<int>("axis");
    int k = static_cast<int>(context.Attr<int>("k"));
    const auto& in_dims = x->dims();
    auto out_dims = indices->dims();
    if (axis < 0) axis += in_dims.size();
    T* x_grad_data = x_grad->mutable_data<T>(context.GetPlace());
    const T* out_grad_data = out_grad->data<T>();
    const int64_t* indices_data = indices->data<int64_t>();
    int pre, n, post;
    GetDims(in_dims, axis, &pre, &n, &post);
    auto& dev_ctx = context.cuda_device_context();
    int block_size = getBlockSize(post * k);
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(((max_threads - 1) / block_size + 1), 1);
    int grid_size = std::min(max_blocks, pre);
    AssignGradWithAxis<T><<<grid_size, block_size, 64 * 4, dev_ctx.stream()>>>(
        out_grad_data, indices_data, x_grad_data, pre, post, n, 1);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    kthvalue,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::KthvalueOpCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>);
REGISTER_OP_CUDA_KERNEL(
    kthvalue_grad,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext, double>,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::KthvalueOpGradCUDAKernel<paddle::platform::CUDADeviceContext,
                                  int64_t>);
