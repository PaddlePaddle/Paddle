// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/kthvalue_kernel.h"

#include "paddle/fluid/operators/top_k_function_cuda.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace phi {
inline int getBlockSize(int col) {
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
bool SortKthvalue(const phi::GPUContext& dev_ctx,
                  const DenseTensor* input_tensor,
                  const int64_t num_cols,
                  const int64_t num_rows,
                  const int k,
                  DenseTensor* out_tensor,
                  DenseTensor* indices_tensor) {
  auto cu_stream = dev_ctx.stream();
  DenseTensor input_indices;
  const std::vector<int64_t> dims = {num_rows, num_cols};
  auto dim = phi::make_ddim(dims);
  input_indices.Resize(dim);
  dev_ctx.template Alloc<int64_t>(&input_indices);
  size_t temp_storage_bytes = -1;
  int block_size = getBlockSize(num_cols);
  unsigned int maxGridDimX = dev_ctx.GetCUDAMaxGridDimSize()[0];
  unsigned int grid_size = num_rows < maxGridDimX
                               ? static_cast<unsigned int>(num_rows)
                               : maxGridDimX;
  paddle::operators::InitIndex<
      int64_t><<<grid_size, block_size, 0, cu_stream>>>(
      input_indices.data<int64_t>(), num_rows, num_cols);
  cub::CountingInputIterator<int64_t> counting_iter(0);
  cub::TransformInputIterator<int64_t,
                              paddle::operators::SegmentOffsetIter,
                              cub::CountingInputIterator<int64_t>>
      segment_offsets_t(counting_iter,
                        paddle::operators::SegmentOffsetIter(num_cols));
  T* sorted_values_ptr;
  int64_t* sorted_indices_ptr;
  DenseTensor temp_values, temp_indices;
  const T* input = input_tensor->data<T>();
  T* values = out_tensor->data<T>();
  int64_t* indices = indices_tensor->mutable_data<int64_t>(dev_ctx.GetPlace());
  temp_values.Resize(dim);
  temp_indices.Resize(dim);
  sorted_values_ptr = dev_ctx.template Alloc<T>(&temp_values);
  sorted_indices_ptr = dev_ctx.template Alloc<int64_t>(&temp_indices);
  auto err =
      cub::DeviceSegmentedRadixSort::SortPairs(nullptr,
                                               temp_storage_bytes,
                                               input,
                                               sorted_values_ptr,
                                               input_indices.data<int64_t>(),
                                               sorted_indices_ptr,
                                               num_cols * num_rows,
                                               num_rows,
                                               segment_offsets_t,
                                               segment_offsets_t + 1,
                                               0,
                                               sizeof(T) * 8,
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
  DenseTensor temp_storage;
  temp_storage.Resize({static_cast<int>(temp_storage_bytes / sizeof(uint8_t))});
  uint8_t* temp_storage_data = dev_ctx.template Alloc<uint8_t>(&temp_storage);

  err = cub::DeviceSegmentedRadixSort::SortPairs(temp_storage_data,
                                                 temp_storage_bytes,
                                                 input,
                                                 sorted_values_ptr,
                                                 input_indices.data<int64_t>(),
                                                 sorted_indices_ptr,
                                                 num_cols * num_rows,
                                                 num_rows,
                                                 segment_offsets_t,
                                                 segment_offsets_t + 1,
                                                 0,
                                                 sizeof(T) * 8,
                                                 cu_stream);
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
  auto& dev = *dev_ctx.eigen_device();
  const Eigen::DSizes<Eigen::DenseIndex, 2> slice_indices{0, k - 1};
  const Eigen::DSizes<Eigen::DenseIndex, 2> slice_sizes{num_rows, 1};
  auto e_indices = EigenMatrix<int64_t>::From(*indices_tensor, dim);
  auto e_tmp_indices =
      EigenMatrix<int64_t>::From(static_cast<const DenseTensor>(temp_indices));
  std::vector<int> odims = {static_cast<int>(num_rows), static_cast<int>(1)};
  dim = phi::make_ddim(odims);
  auto e_values = EigenMatrix<T>::From(*out_tensor, dim);
  auto e_tmp_values =
      EigenMatrix<T>::From(static_cast<const DenseTensor>(temp_values));

  funcs::EigenSlice<std::decay_t<decltype(dev)>, int64_t, 2>::Eval(
      dev, e_indices, e_tmp_indices, slice_indices, slice_sizes);
  funcs::EigenSlice<std::decay_t<decltype(dev)>, T, 2>::Eval(
      dev, e_values, e_tmp_values, slice_indices, slice_sizes);
  return true;
}

template <typename T, typename Context>
void KthvalueKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    int k,
                    int axis,
                    bool keepdim,
                    DenseTensor* output,
                    DenseTensor* indices) {
  const auto& in_dims = x.dims();
  if (axis < 0) axis += in_dims.size();
  auto out_dims = output->dims();
  const T* input_data = x.data<T>();
  T* output_data = dev_ctx.template Alloc<T>(output);
  int64_t* indices_data = dev_ctx.template Alloc<int64_t>(indices);

  if (axis == in_dims.size() - 1) {
    const int64_t& input_height =
        phi::product(phi::slice_ddim(in_dims, 0, in_dims.size() - 1));
    const int64_t& input_width = in_dims[in_dims.size() - 1];
    PADDLE_ENFORCE_EQ(
        SortKthvalue<T>(
            dev_ctx, &x, input_width, input_height, k, output, indices),
        true,
        phi::errors::External("KthvalueOP: Error when use cub sorting"));
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
      DDim tmp_out_dims = phi::make_ddim(tmp_out_shape);
      output->Resize(tmp_out_dims);
      indices->Resize(tmp_out_dims);
    }
    DDim trans_dims(in_dims);
    DDim trans_out_dims(in_dims);
    for (int i = 0; i < trans.size(); i++) {
      trans_dims[i] = in_dims[trans[i]];
      trans_out_dims[i] = in_dims[trans[i]];
    }
    trans_out_dims[in_dims.size() - 1] = 1;
    DenseTensor trans_input;
    trans_input.mutable_data<T>(trans_dims, dev_ctx.GetPlace());
    int ndims = trans.size();
    funcs::TransCompute<phi::GPUContext, T>(
        ndims, dev_ctx, x, &trans_input, trans);
    DenseTensor trans_ind, trans_out;
    trans_ind.mutable_data<int64_t>(trans_out_dims, dev_ctx.GetPlace());
    trans_out.mutable_data<T>(trans_out_dims, dev_ctx.GetPlace());
    const int64_t input_height =
        phi::product(phi::slice_ddim(trans_dims, 0, trans_dims.size() - 1));
    const int64_t input_width = trans_dims[trans_dims.size() - 1];
    PADDLE_ENFORCE_EQ(
        SortKthvalue<T>(dev_ctx,
                        &trans_input,
                        input_width,
                        input_height,
                        k,
                        &trans_out,
                        &trans_ind),
        true,
        phi::errors::External("KthvalueOP: Error when use cub sorting"));
    funcs::TransCompute<phi::GPUContext, int64_t>(
        ndims, dev_ctx, trans_ind, indices, trans);
    funcs::TransCompute<phi::GPUContext, T>(
        ndims, dev_ctx, trans_out, output, trans);
    if (!keepdim) {
      output->Resize(out_dims);
      indices->Resize(out_dims);
    }
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(kthvalue,
                   GPU,
                   ALL_LAYOUT,
                   phi::KthvalueKernel,
                   float,
                   double,
                   int,
                   int64_t) {}
