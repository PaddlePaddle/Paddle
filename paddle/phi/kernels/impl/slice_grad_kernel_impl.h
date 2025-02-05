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

#pragma once

#include "paddle/common/macros.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/slice_utils.h"
#include "paddle/phi/kernels/slice_grad_kernel.h"
namespace phi {

template <typename T, typename Context, size_t D>
void LaunchEigenPadding(
    const Context& context,
    DenseTensor* d_input,
    const DDim& in_dims,
    const DenseTensor* d_out,
    const DDim& out_dims,
    const std::array<std::pair<int64_t, int64_t>, D>& paddings) {
  auto& place = *context.eigen_device();
  auto d_in_t = EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
      *d_input, in_dims);
  auto d_out_t = EigenTensor<T, D, Eigen::RowMajor, Eigen::DenseIndex>::From(
      *d_out, out_dims);

  if (d_input->numel() <= Eigen::NumTraits<int>::highest()) {
    // similar to tf.pad:
    // if element number less than INT_MAX, change the type of index to int
    std::array<std::pair<int, int>, D> paddings_32bit;
    for (size_t i = 0; i < D; i++) {
      paddings_32bit[i] = std::make_pair(paddings[i].first, paddings[i].second);
    }
    funcs::EigenPad<std::decay_t<decltype(place)>, T, D>::Eval32(
        place,
        To32BitIndex(d_in_t),
        To32BitIndex(d_out_t),
        paddings_32bit,
        static_cast<T>(0));
  } else {
    funcs::EigenPad<std::decay_t<decltype(place)>, T, D>::Eval(
        place, d_in_t, d_out_t, paddings, static_cast<T>(0));
  }
}

template <typename T, typename Context, size_t D>
void EigenPaddingCompute(
    const Context& context,
    DenseTensor* d_input,
    const DDim& in_dims,
    const DenseTensor* d_out,
    const DDim& out_dims,
    const std::array<std::pair<int64_t, int64_t>, D>& paddings) {
  if (D <= 3) {
    // if dimension less than 3, cannot reduce dimension
    LaunchEigenPadding<T, Context, D>(
        context, d_input, in_dims, d_out, out_dims, paddings);
  } else {  // else we can reduce dimension
    // count not-zero padding number, and record the dimension
    int need_pad_num = 0, pad_dim = -1;
    for (size_t i = 0; i < D; i++) {
      if (paddings[i].first != 0 || paddings[i].second != 0) {
        need_pad_num++;
        pad_dim = i;
      }
    }

    if (need_pad_num == 1) {
      // only need padding one dimension, we can reduce dimension.
      // only the padding dimension is available for us.
      // How to reduce dimension(5 to 3 for example):
      // before(D=5):
      // in_dims:        [x1,  x2,  x3,  x4,  x5]
      // padding.first:  [0,   0,   a,   0,  0]
      // padding.second: [0,   0,   b,   0,  0]
      //                     | |
      //                     V V
      // after(D=3):
      // reshaped_in_dims:        [x1*x2,  x3,  x4*x5]
      // reshaped_padding.first:  [0,      a,     0]
      // reshaped_padding.second: [0,      b,     0]

      if (pad_dim == D - 1) {
        // only last dimension need padding,
        // reshape the dimension of tensor in 2: [preceding, padding]
        std::vector<int64_t> in_tore_shape(2, 1), out_tore_shape(2, 1);
        std::array<std::pair<int64_t, int64_t>, 2> reshaped_padding;

        // first dimension is the accumulate of preceding dimension
        for (int i = 0; i < pad_dim; i++) {
          in_tore_shape[0] *= in_dims[i];
          out_tore_shape[0] *= out_dims[i];
        }
        // second dimension is the padding dimension
        in_tore_shape[1] = in_dims[pad_dim];
        out_tore_shape[1] = out_dims[pad_dim];

        // convert array from std::vector to DDim
        DDim reshaped_in_dims = common::make_ddim(in_tore_shape);
        DDim reshaped_out_dims = common::make_ddim(out_tore_shape);

        // after reshape: the first dimension do not need padding,
        // set padding[0] zero
        reshaped_padding[0].first = reshaped_padding[0].second = 0;
        // the second dimension is the previous padding dimension
        reshaped_padding[1].first = paddings[pad_dim].first;
        reshaped_padding[1].second = paddings[pad_dim].second;

        LaunchEigenPadding<T, Context, 2>(context,
                                          d_input,
                                          reshaped_in_dims,
                                          d_out,
                                          reshaped_out_dims,
                                          reshaped_padding);
      } else if (pad_dim == 0) {
        // only first dimension need padding,
        // reshape the dimension of tensor in 2: [padding, succeeding]
        // similar to (D - 1)
        std::vector<int64_t> in_tore_shape(2, 1), out_tore_shape(2, 1);
        std::array<std::pair<int64_t, int64_t>, 2> reshaped_padding;

        // first dimension is the padding dimension
        in_tore_shape[0] = in_dims[pad_dim];
        out_tore_shape[0] = out_dims[pad_dim];
        // second dimension is the accumulate of succeeding dimension
        for (size_t i = pad_dim + 1; i < D; i++) {
          in_tore_shape[1] *= in_dims[i];
          out_tore_shape[1] *= out_dims[i];
        }

        // convert array from std::vector to DDim
        DDim reshaped_in_dims = common::make_ddim(in_tore_shape);
        DDim reshaped_out_dims = common::make_ddim(out_tore_shape);

        // after reshape:
        // the first dimension is the previous padding dimension
        reshaped_padding[0].first = paddings[pad_dim].first;
        reshaped_padding[0].second = paddings[pad_dim].second;
        // the second dimension do not need padding, set padding[1] zero
        reshaped_padding[1].first = reshaped_padding[1].second = 0;

        LaunchEigenPadding<T, Context, 2>(context,
                                          d_input,
                                          reshaped_in_dims,
                                          d_out,
                                          reshaped_out_dims,
                                          reshaped_padding);
      } else {
        // other dimension need padding
        // reshape the dimension of tensor in 3:
        // [preceding, padding, succeeding]
        std::vector<int64_t> in_tore_shape(3, 1), out_tore_shape(3, 1);
        std::array<std::pair<int64_t, int64_t>, 3> reshaped_padding;

        // first dimension is the accumulate of preceding dimension
        for (int i = 0; i < pad_dim; i++) {
          in_tore_shape[0] *= in_dims[i];
          out_tore_shape[0] *= out_dims[i];
        }
        // second dimension is the padding dimension
        in_tore_shape[1] = in_dims[pad_dim];
        out_tore_shape[1] = out_dims[pad_dim];
        // third dimension is the accumulate of succeeding dimension
        for (size_t i = pad_dim + 1; i < D; i++) {
          in_tore_shape[2] *= in_dims[i];
          out_tore_shape[2] *= out_dims[i];
        }

        // convert array from std::vector to DDim
        DDim reshaped_in_dims = common::make_ddim(in_tore_shape);
        DDim reshaped_out_dims = common::make_ddim(out_tore_shape);

        // after reshape:
        // the first dimension do not need padding, set padding[0] zero
        reshaped_padding[0].first = reshaped_padding[2].second = 0;
        // the second dimension is the previous padding dimension
        reshaped_padding[1].first = paddings[pad_dim].first;
        reshaped_padding[1].second = paddings[pad_dim].second;
        // the third dimension do not need padding, set padding[2] zero
        reshaped_padding[2].first = reshaped_padding[2].second = 0;

        LaunchEigenPadding<T, Context, 3>(context,
                                          d_input,
                                          reshaped_in_dims,
                                          d_out,
                                          reshaped_out_dims,
                                          reshaped_padding);
      }
    } else {
      // need padding at many dimension, cannot reduce dimension
      LaunchEigenPadding<T, Context>(
          context, d_input, in_dims, d_out, out_dims, paddings);
    }
  }
}

template <typename T, typename Context, size_t D>
void SliceGradCompute(const Context& ctx,
                      const DenseTensor& out_grad,
                      const std::vector<int64_t>& axes,
                      const std::vector<int64_t>& starts,
                      const std::vector<int64_t>& ends UNUSED,
                      const std::vector<int64_t>& infer_flags UNUSED,
                      const std::vector<int64_t>& decrease_axis,
                      DenseTensor* input_grad) {
  auto* d_out = &out_grad;
  auto* d_input = input_grad;
  ctx.template Alloc<T>(d_input);

  auto out_dims = d_out->dims();
  auto in_dims = d_input->dims();

  auto decrease_size = decrease_axis.size();
  if (decrease_size > 0) {
    if (decrease_size == static_cast<size_t>(in_dims.size())) {
      // all dims decrease
      std::vector<int> origin_out_shape(decrease_size, 1);
      out_dims = common::make_ddim(std::vector<int>(decrease_size, 1));
    } else {
      std::vector<int> origin_out_shape(out_dims.size() + decrease_size, -1);
      for (size_t i = 0; i < decrease_size; ++i) {
        origin_out_shape[decrease_axis[i]] = 1;
      }

      int index = 0;
      for (size_t i = 0; i < origin_out_shape.size(); ++i) {
        if (origin_out_shape[i] == -1) {
          origin_out_shape[i] = out_dims[index];
          ++index;
        }
      }

      out_dims = common::make_ddim(origin_out_shape);
    }
  }

  auto offsets = Eigen::array<int64_t, D>();
  auto extents = Eigen::array<int64_t, D>();
  for (size_t i = 0; i < D; ++i) {
    offsets[i] = 0;
    extents[i] = out_dims[i];
  }

  for (size_t i = 0; i < axes.size(); ++i) {
    int axis = axes[i];
    int64_t start = starts[i] < 0 ? (starts[i] + in_dims[axis]) : starts[i];
    start = std::max(start, static_cast<int64_t>(0));
    offsets[axis] = start;
  }

  std::array<std::pair<int64_t, int64_t>, D> paddings;
  for (size_t i = 0; i < paddings.size(); ++i) {
    paddings[i].first = offsets[i];
    paddings[i].second = (in_dims[i] - out_dims[i]) - offsets[i];
  }
  EigenPaddingCompute<T, Context, D>(
      ctx, d_input, in_dims, d_out, out_dims, paddings);
}

template <typename T, typename Context>
void SliceGradKernel(const Context& ctx,
                     const DenseTensor& input,
                     const DenseTensor& out_grad,
                     const std::vector<int64_t>& axes,
                     const IntArray& starts_arr,
                     const IntArray& ends_arr,
                     const std::vector<int64_t>& infer_flags,
                     const std::vector<int64_t>& decrease_axis,
                     DenseTensor* input_grad) {
  size_t rank = input.dims().size();

  auto& starts = starts_arr.GetData();
  auto& ends = ends_arr.GetData();

  switch (rank) {
    case 1:
      SliceGradCompute<T, Context, 1>(ctx,
                                      out_grad,
                                      axes,
                                      starts,
                                      ends,
                                      infer_flags,
                                      decrease_axis,
                                      input_grad);
      break;
    case 2:
      SliceGradCompute<T, Context, 2>(ctx,
                                      out_grad,
                                      axes,
                                      starts,
                                      ends,
                                      infer_flags,
                                      decrease_axis,
                                      input_grad);
      break;
    case 3:
      SliceGradCompute<T, Context, 3>(ctx,
                                      out_grad,
                                      axes,
                                      starts,
                                      ends,
                                      infer_flags,
                                      decrease_axis,
                                      input_grad);
      break;
    case 4:
      SliceGradCompute<T, Context, 4>(ctx,
                                      out_grad,
                                      axes,
                                      starts,
                                      ends,
                                      infer_flags,
                                      decrease_axis,
                                      input_grad);
      break;
    case 5:
      SliceGradCompute<T, Context, 5>(ctx,
                                      out_grad,
                                      axes,
                                      starts,
                                      ends,
                                      infer_flags,
                                      decrease_axis,
                                      input_grad);
      break;
    case 6:
      SliceGradCompute<T, Context, 6>(ctx,
                                      out_grad,
                                      axes,
                                      starts,
                                      ends,
                                      infer_flags,
                                      decrease_axis,
                                      input_grad);
      break;
    default:
      PADDLE_THROW(common::errors::InvalidArgument(
          "The rank of input should be less than 7, but received %d.", rank));
  }
}

template <typename T, typename Context>
void SliceArrayGradKernel(const Context& dev_ctx,
                          const TensorArray& input,
                          const TensorArray& out_grad,
                          const IntArray& starts,
                          const IntArray& ends UNUSED,
                          TensorArray* input_grad) {
  int64_t d_in_size = input.size();
  input_grad->resize(d_in_size);
  // If the input is TensorArray, the rank of input is 1.
  // So only use the 0th element of starts.
  int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
  start = std::max(start, static_cast<int64_t>(0));
  // set zero
  phi::funcs::SetConstant<Context, T> functor;
  for (int i = 0; i < d_in_size; ++i) {
    const auto& dim = input.at(i).dims();
    auto* in_grad_tensor = &input_grad->at(i);
    in_grad_tensor->Resize(dim);
    dev_ctx.template Alloc<T>(in_grad_tensor);
    functor(dev_ctx, in_grad_tensor, static_cast<T>(0));
  }

  int d_out_size = out_grad.size();
  for (int i = 0; i < d_out_size; ++i) {
    phi::Copy<Context>(dev_ctx,
                       out_grad[i],
                       dev_ctx.GetPlace(),
                       false,
                       &input_grad->at(start + i));
  }
}

template <typename T, typename Context>
void SliceArrayDenseGradKernel(const Context& dev_ctx,
                               const TensorArray& input,
                               const DenseTensor& out_grad,
                               const IntArray& starts,
                               TensorArray* input_grad) {
  int64_t d_in_size = input.size();
  input_grad->resize(d_in_size);
  // If the input is TensorArray, the rank of input is 1.
  // So only use the 0th element of starts.
  int64_t start = starts[0] < 0 ? (starts[0] + d_in_size) : starts[0];
  start = std::max(start, static_cast<int64_t>(0));
  // set zero
  phi::funcs::SetConstant<Context, T> functor;
  for (int i = 0; i < d_in_size; ++i) {
    const auto& dim = input.at(i).dims();
    auto* in_grad_tensor = &input_grad->at(i);
    in_grad_tensor->Resize(dim);
    dev_ctx.template Alloc<T>(in_grad_tensor);
    functor(dev_ctx, in_grad_tensor, static_cast<T>(0));
  }
  phi::Copy<Context>(
      dev_ctx, out_grad, dev_ctx.GetPlace(), false, &input_grad->at(start));
}

}  // namespace phi
