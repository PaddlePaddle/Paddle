// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <numeric>
#include <type_traits>

#include "paddle/common/flags.h"
#include "paddle/phi/kernels/concat_kernel.h"
#include "paddle/phi/kernels/contiguous_kernel.h"
#include "paddle/phi/kernels/conv_kernel.h"
#include "paddle/phi/kernels/cpu/conv_util.h"
#include "paddle/phi/kernels/elementwise_add_kernel.h"
#include "paddle/phi/kernels/fill_kernel.h"
#include "paddle/phi/kernels/full_kernel.h"
#include "paddle/phi/kernels/funcs/batch_norm_utils.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"
#include "paddle/phi/kernels/funcs/im2col_slow.cuh"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/funcs/vol2col_slow.cuh"
#include "paddle/phi/kernels/reduce_sum_kernel.h"
#include "paddle/phi/kernels/slice_kernel.h"

COMMON_DECLARE_bool(use_accuracy_compatible_kernel);

#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/dense_tensor.h"

namespace phi {
template <typename T>
static inline T div_rtn(T x, T y) {
  int q = x / y;
  int r = x % y;
  if ((r != 0) && ((r < 0) != (y < 0))) --q;
  return q;
}

template <typename C,
          std::enable_if_t<std::is_integral_v<typename C::value_type>, int> = 0>
inline int64_t multiply_integers(const C& container) {
  return std::accumulate(container.begin(),
                         container.end(),
                         static_cast<int64_t>(1),
                         std::multiplies<>());
}

template <typename Iter,
          std::enable_if_t<std::is_integral_v<
                               typename std::iterator_traits<Iter>::value_type>,
                           int> = 0>
inline int64_t multiply_integers(Iter begin, Iter end) {
  return std::accumulate(
      begin, end, static_cast<int64_t>(1), std::multiplies<>());
}

template <int64_t dim>
std::vector<int64_t> GetOutputSpatialSize(
    const DenseTensor& input,
    const std::vector<int64_t>& kernel_size,
    const std::vector<int64_t>& stride_size,
    const std::vector<int64_t>& pad_size,
    const std::vector<int64_t>& dilation_size) {
  std::vector<int64_t> sizes;
  auto input_dim = input.dims().size();

  for (int64_t index = 0; index < dim; ++index) {
    int64_t input_size = input.dims()[index + input_dim - dim];
    int64_t kernel = kernel_size[index];
    int64_t stride = stride_size[index];
    int64_t pad = pad_size[index];
    int64_t dilation = dilation_size[index];

    int64_t numerator = input_size + 2 * pad - (dilation * (kernel - 1) + 1);
    int64_t size = div_rtn<int64_t>(numerator, stride) + 1;

    sizes.push_back(size);
  }

  return sizes;
}

template <int64_t dim>
std::vector<int64_t> GetOutputSize(const DenseTensor& input,
                                   const DenseTensor& weight,
                                   const std::vector<int64_t>& kernel_size,
                                   const std::vector<int64_t>& stride_size,
                                   const std::vector<int64_t>& pad_size,
                                   const std::vector<int64_t>& dilation_size) {
  auto output_size = GetOutputSpatialSize<dim>(
      input, kernel_size, stride_size, pad_size, dilation_size);

  output_size.insert(output_size.begin(), weight.dims()[0]);

  if (input.dims().size() == dim + 2) {
    output_size.insert(output_size.begin(), input.dims()[0]);
  }

  return output_size;
}

template <typename T, typename Context, int64_t dim>
void hvol2col(const Context& dev_ctx,
              const T* data_hvol,
              int channels,
              const std::vector<int64_t>& input_size,
              const std::vector<int64_t>& output_size,
              const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride_size,
              const std::vector<int64_t>& pad_size,
              const std::vector<int64_t>& dilation_size,
              T* data_col) {
  if (dim == 3) {
    phi::funcs::vol2col_slow<T, Context>(dev_ctx,
                                         data_hvol,
                                         channels,
                                         input_size[0],
                                         input_size[1],
                                         input_size[2],
                                         output_size[0],
                                         output_size[1],
                                         output_size[2],
                                         kernel_size[0],
                                         kernel_size[1],
                                         kernel_size[2],
                                         pad_size[0],
                                         pad_size[1],
                                         pad_size[2],
                                         stride_size[0],
                                         stride_size[1],
                                         stride_size[2],
                                         dilation_size[0],
                                         dilation_size[1],
                                         dilation_size[2],
                                         data_col);
  } else if (dim == 2) {
    phi::funcs::im2col_slow<T, Context>(dev_ctx,
                                        data_hvol,
                                        channels,
                                        input_size[0],
                                        input_size[1],
                                        output_size[0],
                                        output_size[1],
                                        kernel_size[0],
                                        kernel_size[1],
                                        pad_size[0],
                                        pad_size[1],
                                        stride_size[0],
                                        stride_size[1],
                                        dilation_size[0],
                                        dilation_size[1],
                                        data_col);
  }
}

template <typename T, typename Context, int64_t dim>
void col2hvol(const Context& dev_ctx,
              const T* data_col,
              const int channels,
              const std::vector<int64_t>& input_size,
              const std::vector<int64_t>& output_size,
              const std::vector<int64_t>& kernel_size,
              const std::vector<int64_t>& stride_size,
              const std::vector<int64_t>& pad_size,
              const std::vector<int64_t>& dilation_size,
              T* data_hvol) {
  if (dim == 3) {
    phi::funcs::col2vol_slow<T, T, Context>(dev_ctx,
                                            data_col,
                                            channels,
                                            input_size[0],
                                            input_size[1],
                                            input_size[2],
                                            output_size[0],
                                            output_size[1],
                                            output_size[2],
                                            kernel_size[0],
                                            kernel_size[1],
                                            kernel_size[2],
                                            pad_size[0],
                                            pad_size[1],
                                            pad_size[2],
                                            stride_size[0],
                                            stride_size[1],
                                            stride_size[2],
                                            dilation_size[0],
                                            dilation_size[1],
                                            dilation_size[2],
                                            data_hvol);
  }
  if (dim == 2) {
    phi::funcs::col2im_slow<T, T, Context>(dev_ctx,
                                           data_col,
                                           channels,
                                           input_size[0],
                                           input_size[1],
                                           output_size[0],
                                           output_size[1],
                                           kernel_size[0],
                                           kernel_size[1],
                                           pad_size[0],
                                           pad_size[1],
                                           stride_size[0],
                                           stride_size[1],
                                           dilation_size[0],
                                           dilation_size[1],
                                           data_hvol);
  }
}

// Select View function
template <typename T>
DenseTensor Select(const DenseTensor& src, int64_t index) {
  DenseTensor out;
  out.ShareDataWith(src);
  auto dims = src.dims();
  std::vector<int64_t> new_dims;
  for (int i = 1; i < dims.size(); ++i) {
    new_dims.push_back(dims[i]);
  }
  out.Resize(common::make_ddim(new_dims));
  int64_t stride_0 = src.numel() / dims[0];
  size_t offset_bytes = index * stride_0 * sizeof(T);
  out.set_offset(src.offset() + offset_bytes);
  return out;
}

template <typename T, typename Context, int Dims>
void SlowConvDilatedAllCUDAImpl(const Context& dev_ctx,
                                DenseTensor* output,
                                const DenseTensor* input,
                                const DenseTensor* weight,
                                const DenseTensor* bias,
                                const DenseTensor* grad_output,
                                DenseTensor* grad_input,
                                DenseTensor* grad_weight,
                                DenseTensor* grad_bias,
                                const std::vector<int64_t>& kernel_size,
                                const std::vector<int64_t>& strides,
                                const std::vector<int64_t>& paddings,
                                const std::vector<int64_t>& dilations) {
  const int64_t batch_size = input->dims()[0];
  const int64_t input_channels = weight->dims()[1];
  const int64_t output_channels = weight->dims()[0];

  std::vector<int64_t> input_spatial_size;
  for (int i = 2; i < input->dims().size(); ++i) {
    input_spatial_size.push_back(input->dims()[i]);
  }

  std::vector<int64_t> output_spatial_size = GetOutputSpatialSize<Dims>(
      *input, kernel_size, strides, paddings, dilations);

  int64_t kernel_volume = multiply_integers(kernel_size);
  int64_t output_volume = multiply_integers(output_spatial_size);

  // Buffer
  int64_t col_dim0 = input_channels * kernel_volume;
  int64_t col_dim1 = output_volume;

  DenseTensor columns;
  if (output || grad_weight || grad_input) {
    columns.Resize({col_dim0, col_dim1});
    dev_ctx.template Alloc<T>(&columns);
  }

  // Initialize
  phi::funcs::SetConstant<Context, T> set_zero;
  if (grad_weight) set_zero(dev_ctx, grad_weight, static_cast<T>(0));
  if (grad_bias) set_zero(dev_ctx, grad_bias, static_cast<T>(0));
  if (output && !bias) set_zero(dev_ctx, output, static_cast<T>(0));

  // Bias CPU Mirror
  DenseTensor bias_cpu;
  const T* bias_cpu_data = nullptr;
  if (output && bias) {
    phi::Copy(dev_ctx, *bias, phi::CPUPlace(), true, &bias_cpu);
    bias_cpu_data = bias_cpu.data<T>();
  }

  DenseTensor grad_output_n;
  std::vector<int64_t> sum_axes;
  for (int i = 0; i < Dims; ++i) sum_axes.push_back(i + 1);

  auto blas = funcs::GetBlas<Context, T>(dev_ctx);

  for (int elt = 0; elt < batch_size; ++elt) {
    T* columns_ptr = columns.data<T>();

    // Prepare Input Slice View
    DenseTensor input_n = Select<T>(*input, elt);
    const T* input_ptr_raw = input_n.data<T>();

    // Forward
    if (output) {
      DenseTensor output_n = Select<T>(*output, elt);
      T* output_ptr_raw = output_n.data<T>();

      if (bias) {
        for (int n = 0; n < output_channels; ++n) {
          DenseTensor out_slice = Select<T>(output_n, n);

          phi::FillKernel<T, Context>(
              dev_ctx, out_slice, phi::Scalar(bias_cpu_data[n]), &out_slice);
        }
      }

      hvol2col<T, Context, Dims>(dev_ctx,
                                 input_ptr_raw,
                                 input_channels,
                                 input_spatial_size,
                                 output_spatial_size,
                                 kernel_size,
                                 strides,
                                 paddings,
                                 dilations,
                                 columns_ptr);
      blas.GEMM(false,                              // TransA
                false,                              // TransB
                static_cast<int>(output_channels),  // M
                static_cast<int>(col_dim1),         // N
                static_cast<int>(col_dim0),         // K
                static_cast<T>(1),                  // alpha
                weight->data<T>(),                  // A
                static_cast<int>(col_dim0),         // lda
                columns_ptr,                        // B
                static_cast<int>(col_dim1),         // ldb
                static_cast<T>(1),                  // beta = 1 (Accumulate)
                output_ptr_raw,                     // C
                static_cast<int>(col_dim1)          // ldc
      );
    } else {
      grad_output_n = Select<T>(*grad_output, elt);
    }

    // Backward Grad Input
    if (grad_input) {
      DenseTensor grad_input_n = Select<T>(*grad_input, elt);
      T* grad_input_ptr_raw = grad_input_n.data<T>();
      const T* grad_output_ptr_raw = grad_output_n.data<T>();

      blas.GEMM(true,                               // TransA
                false,                              // TransB
                static_cast<int>(col_dim0),         // M
                static_cast<int>(col_dim1),         // N
                static_cast<int>(output_channels),  // K
                static_cast<T>(1),                  // alpha
                weight->data<T>(),                  // A
                static_cast<int>(col_dim0),         // lda
                grad_output_ptr_raw,                // B
                static_cast<int>(col_dim1),         // ldb
                static_cast<T>(0),                  // beta
                columns_ptr,                        // C
                static_cast<int>(col_dim1)          // ldc
      );

      col2hvol<T, Context, Dims>(dev_ctx,
                                 columns_ptr,
                                 input_channels,
                                 input_spatial_size,
                                 output_spatial_size,
                                 kernel_size,
                                 strides,
                                 paddings,
                                 dilations,
                                 grad_input_ptr_raw);
    }

    // Backward Grad Weight
    if (grad_weight) {
      const T* grad_output_ptr_raw = grad_output_n.data<T>();

      hvol2col<T, Context, Dims>(dev_ctx,
                                 input_ptr_raw,
                                 input_channels,
                                 input_spatial_size,
                                 output_spatial_size,
                                 kernel_size,
                                 strides,
                                 paddings,
                                 dilations,
                                 columns_ptr);

      blas.GEMM(false,                              // TransA
                true,                               // TransB
                static_cast<int>(output_channels),  // M
                static_cast<int>(col_dim0),         // N
                static_cast<int>(col_dim1),         // K
                static_cast<T>(1),                  // alpha
                grad_output_ptr_raw,                // A
                static_cast<int>(col_dim1),         // lda
                columns_ptr,                        // B
                static_cast<int>(col_dim1),         // ldb
                static_cast<T>(1),                  // beta
                grad_weight->data<T>(),             // C
                static_cast<int>(col_dim0)          // ldc
      );
    }

    // Backward Grad Bias
    if (grad_bias) {
      DenseTensor sum_result =
          phi::Sum<T, Context>(dev_ctx,
                               grad_output_n,
                               phi::IntArray(sum_axes),
                               CppTypeToDataType<T>::Type(),
                               false);
      phi::Add<T, Context>(dev_ctx, *grad_bias, sum_result, grad_bias);
    }
  }
}

template <typename T, typename Context, int64_t dim>
void SlowConvBackwardNoGroup(const Context& dev_ctx,
                             const DenseTensor& grad_output,
                             const DenseTensor& input,
                             const DenseTensor& weight,
                             const std::vector<int64_t>& kernel_size,
                             const std::vector<int64_t>& strides,
                             const std::vector<int64_t>& paddings,
                             const std::vector<int64_t>& dilations,
                             DenseTensor* grad_input,
                             DenseTensor* grad_weight,
                             DenseTensor* grad_bias) {
  int64_t rank = input.dims().size();
  bool is_batch = (rank == (dim + 2));

  // tensor.unsqueeze(0)
  auto make_batch_view = [&](const DenseTensor& src, DenseTensor& dst) {
    if (!is_batch) {
      dst.ShareDataWith(src);
      std::vector<int64_t> new_shape = {1};
      for (int i = 0; i < src.dims().size(); ++i)
        new_shape.push_back(src.dims()[i]);
      dst.Resize(common::make_ddim(new_shape));
    } else {
      dst.ShareDataWith(src);
    }
  };

  DenseTensor grad_output_;
  make_batch_view(grad_output, grad_output_);

  DenseTensor input_;
  make_batch_view(input, input_);

  const DenseTensor& weight_ = weight;

  DenseTensor grad_input_view;
  DenseTensor* grad_input_ptr = nullptr;

  if (grad_input) {
    dev_ctx.template Alloc<T>(grad_input);

    if (!is_batch) {
      grad_input_view.ShareDataWith(*grad_input);
      std::vector<int64_t> new_shape = {1};
      for (int i = 0; i < grad_input->dims().size(); ++i)
        new_shape.push_back(grad_input->dims()[i]);
      grad_input_view.Resize(common::make_ddim(new_shape));
      grad_input_ptr = &grad_input_view;
    } else {
      grad_input_ptr = grad_input;
    }
  }

  DenseTensor* grad_weight_ptr = nullptr;
  if (grad_weight) {
    dev_ctx.template Alloc<T>(grad_weight);
    grad_weight_ptr = grad_weight;
  }

  DenseTensor* grad_bias_ptr = nullptr;
  if (grad_bias) {
    dev_ctx.template Alloc<T>(grad_bias);
    grad_bias_ptr = grad_bias;
  }

  SlowConvDilatedAllCUDAImpl<T, Context, dim>(
      dev_ctx,
      nullptr,          // [Output]
      &input_,          // [Input]
      &weight_,         // [Weight]
      nullptr,          // [Bias]
      &grad_output_,    // [GradOutput]
      grad_input_ptr,   // [GradInput]
      grad_weight_ptr,  // [GradWeight]
      grad_bias_ptr,    // [GradBias] (New)
      kernel_size,
      strides,
      paddings,
      dilations);
}

template <typename T, typename Context, int64_t dim>
void SlowConvNoGroup(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& weight,
                     const DenseTensor* bias,
                     const std::vector<int64_t>& kernel_size,
                     const std::vector<int64_t>& strides,
                     const std::vector<int64_t>& paddings,
                     const std::vector<int64_t>& dilations,
                     DenseTensor* output) {
  int64_t rank = input.dims().size();
  bool is_batch = (rank == (dim + 2));

  // (is_batch ? input.contiguous() : input.contiguous().unsqueeze(0));
  DenseTensor input_;
  if (!is_batch) {
    input_.ShareDataWith(input);
    std::vector<int64_t> new_shape = {1};
    for (int i = 0; i < rank; ++i) new_shape.push_back(input.dims()[i]);
    input_.Resize(common::make_ddim(new_shape));
  } else {
    input_.ShareDataWith(input);
  }

  const DenseTensor& weight_ = weight;

  // (is_batch ? output : output.unsqueeze(0));
  if (output) dev_ctx.template Alloc<T>(output);

  DenseTensor output_;
  if (!is_batch) {
    output_.ShareDataWith(*output);

    std::vector<int64_t> out_shape = {1};
    for (int i = 0; i < output->dims().size(); ++i) {
      out_shape.push_back(output->dims()[i]);
    }
    output_.Resize(common::make_ddim(out_shape));
  } else {
    output_.ShareDataWith(*output);
  }

  SlowConvDilatedAllCUDAImpl<T, Context, dim>(dev_ctx,
                                              &output_,  // [Output]
                                              &input_,   // [Input]
                                              &weight_,  // [Weight]
                                              bias,      // [Bias]
                                              nullptr,   // [GradOutput]
                                              nullptr,   // [GradInput]
                                              nullptr,   // [GradWeight]
                                              nullptr,   // [GradBias]
                                              kernel_size,
                                              strides,
                                              paddings,
                                              dilations);
}

template <typename T, typename Context, int64_t dim>
void SlowConvForward(const Context& dev_ctx,
                     const DenseTensor& input,
                     const DenseTensor& filter_t,
                     const paddle::optional<DenseTensor>& bias,
                     const std::vector<int>& strides,
                     const std::vector<int>& paddings_t,
                     const std::string& padding_algorithm,
                     int groups,
                     const std::vector<int>& dilations_t,
                     const std::string& data_format,
                     DenseTensor* output) {
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;
  DenseTensor filter = filter_t;

  if (input.numel() == 0 || filter.numel() == 0) {
    Full<T, Context>(dev_ctx, output->dims(), 0, output);
    return;
  }

  dev_ctx.template Alloc<T>(output);

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  DenseTensor transformed_input(input.type());
  DenseTensor transformed_output(output->type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);

    ResizeToChannelFirst<Context, T>(dev_ctx, output, &transformed_output);

  } else {
    transformed_input = input;
    transformed_output = *output;
  }

  // update padding and dilation
  auto trans_in_dims = transformed_input.dims();
  auto filter_dims = filter.dims();

  DDim in_data_dims = slice_ddim(trans_in_dims, 2, trans_in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());

  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  // =================================================================
  // Contiguous & Grouping
  // =================================================================
  DenseTensor input_contiguous;
  phi::ContiguousKernel<T, Context>(
      dev_ctx, transformed_input, &input_contiguous);

  DenseTensor weight_contiguous;
  phi::ContiguousKernel<T, Context>(dev_ctx, filter_t, &weight_contiguous);

  auto to_int64_vec = [](const std::vector<int>& in) {
    return std::vector<int64_t>(in.begin(), in.end());
  };

  const DenseTensor* bias_ptr = bias.get_ptr();
  DenseTensor bias_contiguous;

  if (bias_ptr) {
    phi::ContiguousKernel<T, Context>(dev_ctx, *bias_ptr, &bias_contiguous);
    bias_ptr = &bias_contiguous;
  }

  if (groups == 1) {
    SlowConvNoGroup<T, Context, dim>(dev_ctx,
                                     input_contiguous,
                                     weight_contiguous,
                                     bias_ptr,
                                     to_int64_vec(ksize),
                                     to_int64_vec(strides),
                                     to_int64_vec(paddings),
                                     to_int64_vec(dilations),
                                     &transformed_output);

  } else {
    int64_t in_rank = input_contiguous.dims().size();
    bool has_batch = (in_rank == dim + 2);
    int channel_dim = has_batch ? 1 : 0;

    int64_t in_channels = input_contiguous.dims()[channel_dim];
    int64_t out_channels = weight_contiguous.dims()[0];

    int64_t in_g_sz = in_channels / groups;
    int64_t out_g_sz = out_channels / groups;

    std::vector<DenseTensor> outputs(groups);

    for (int g = 0; g < groups; ++g) {
      // Slice Input (Channel)
      DenseTensor input_g;
      phi::SliceKernel<T, Context>(dev_ctx,
                                   input_contiguous,
                                   {channel_dim},
                                   {g * in_g_sz},
                                   {(g + 1) * in_g_sz},
                                   {1},
                                   {},
                                   &input_g);

      // Slice Weight (OutChannel dim 0)
      DenseTensor weight_g;
      phi::SliceKernel<T, Context>(dev_ctx,
                                   weight_contiguous,
                                   {0},
                                   {g * out_g_sz},
                                   {(g + 1) * out_g_sz},
                                   {1},
                                   {},
                                   &weight_g);

      // Slice Bias (OutChannel dim 0)
      DenseTensor bias_g;
      const DenseTensor* bias_g_ptr = nullptr;
      if (bias_ptr) {
        phi::SliceKernel<T, Context>(dev_ctx,
                                     *bias_ptr,
                                     {0},
                                     {g * out_g_sz},
                                     {(g + 1) * out_g_sz},
                                     {1},
                                     {},
                                     &bias_g);
        bias_g_ptr = &bias_g;
      }

      DenseTensor output_g;
      auto out_shape = transformed_output.dims();
      out_shape[channel_dim] = out_g_sz;
      output_g.Resize(out_shape);
      dev_ctx.template Alloc<T>(&output_g);

      SlowConvNoGroup<T, Context, dim>(dev_ctx,
                                       input_g,
                                       weight_g,
                                       bias_g_ptr,
                                       to_int64_vec(ksize),
                                       to_int64_vec(strides),
                                       to_int64_vec(paddings),
                                       to_int64_vec(dilations),
                                       &output_g);

      outputs[g] = output_g;
    }

    // Concat
    std::vector<const DenseTensor*> outputs_ptr;
    for (auto& t : outputs) outputs_ptr.push_back(&t);

    phi::ConcatKernel<T, Context>(
        dev_ctx, outputs_ptr, channel_dim, &transformed_output);
  }

  if (channel_last) {
    TransToChannelLast<Context, T>(dev_ctx, &transformed_output, output);
  }
}

template <typename T, typename Context, int64_t dim>
void SlowConvBackward(const Context& dev_ctx,
                      const DenseTensor& input,
                      const DenseTensor& filter_t,
                      const DenseTensor& output_grad,
                      const std::vector<int>& strides,
                      const std::vector<int>& paddings_t,
                      const std::string& padding_algorithm,
                      const std::vector<int>& dilations_t,
                      int groups,
                      const std::string& data_format,
                      DenseTensor* input_grad,
                      DenseTensor* filter_grad,
                      DenseTensor* bias_grad) {
  if (!input_grad && !filter_grad && !bias_grad) return;
  std::vector<int> paddings = paddings_t;
  std::vector<int> dilations = dilations_t;

  DenseTensor filter = filter_t;
  // 0-size
  if (input.numel() == 0 || filter_t.numel() == 0) {
    if (input_grad) dev_ctx.template Alloc<T>(input_grad);
    if (filter_grad) {
      Full<T, Context>(dev_ctx, filter_grad->dims(), 0, filter_grad);
    }
    if (bias_grad) {
      dev_ctx.template Alloc<T>(bias_grad);
      Full<T, Context>(dev_ctx, bias_grad->dims(), 0, bias_grad);
    }
    return;
  }

  if (input_grad) dev_ctx.template Alloc<T>(input_grad);
  if (filter_grad) dev_ctx.template Alloc<T>(filter_grad);
  if (bias_grad) dev_ctx.template Alloc<T>(bias_grad);

  const bool channel_last = (data_format == "NHWC" || data_format == "NDHWC");

  DenseTensor transformed_input(input.type());
  DenseTensor transformed_output_grad(output_grad.type());

  if (channel_last) {
    ResizeToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);
    TransToChannelFirst<Context, T>(dev_ctx, &input, &transformed_input);

    ResizeToChannelFirst<Context, T>(
        dev_ctx, &output_grad, &transformed_output_grad);
    TransToChannelFirst<Context, T>(
        dev_ctx, &output_grad, &transformed_output_grad);
  } else {
    transformed_input = input;
    transformed_output_grad = output_grad;
  }

  // update padding and dilation
  auto in_dims = transformed_input.dims();
  auto filter_dims = filter.dims();
  DDim in_data_dims = slice_ddim(in_dims, 2, in_dims.size());
  DDim filter_data_dims = slice_ddim(filter_dims, 2, filter_dims.size());
  std::vector<int> ksize = common::vectorize<int>(filter_data_dims);
  UpdatePaddingAndDilation<int>(
      &paddings, &dilations, padding_algorithm, in_data_dims, strides, ksize);

  // =================================================================
  // Contiguous & Grouping
  // =================================================================
  DenseTensor tmp_input_grad;
  DenseTensor* t_input_grad_ptr = nullptr;
  DenseTensor* t_filter_grad_ptr = filter_grad;
  DenseTensor* t_bias_grad_ptr = bias_grad;

  if (input_grad) {
    if (channel_last) {
      tmp_input_grad.Resize(transformed_input.dims());
      t_input_grad_ptr = &tmp_input_grad;
    } else {
      t_input_grad_ptr = input_grad;
    }
  }

  // Contiguous
  DenseTensor grad_output_cont;
  phi::ContiguousKernel<T, Context>(
      dev_ctx, transformed_output_grad, &grad_output_cont);

  DenseTensor input_cont;
  phi::ContiguousKernel<T, Context>(dev_ctx, transformed_input, &input_cont);

  DenseTensor weight_cont;
  phi::ContiguousKernel<T, Context>(dev_ctx, filter, &weight_cont);

  auto to_int64_vec = [](const std::vector<int>& in) {
    return std::vector<int64_t>(in.begin(), in.end());
  };

  // Group
  if (groups == 1) {
    SlowConvBackwardNoGroup<T, Context, dim>(dev_ctx,
                                             grad_output_cont,
                                             input_cont,
                                             weight_cont,
                                             to_int64_vec(ksize),
                                             to_int64_vec(strides),
                                             to_int64_vec(paddings),
                                             to_int64_vec(dilations),
                                             t_input_grad_ptr,
                                             t_filter_grad_ptr,
                                             t_bias_grad_ptr);
  } else {
    int64_t in_rank = input_cont.dims().size();
    bool has_batch = (in_rank == dim + 2);
    int channel_dim = has_batch ? 1 : 0;

    int64_t in_channels = input_cont.dims()[channel_dim];
    int64_t out_channels = grad_output_cont.dims()[channel_dim];

    int64_t in_g_sz = in_channels / groups;
    int64_t out_g_sz = out_channels / groups;

    std::vector<DenseTensor> grad_inputs_g(groups);
    std::vector<DenseTensor> grad_weights_g(groups);
    std::vector<DenseTensor> grad_biases_g(groups);

    for (int g = 0; g < groups; ++g) {
      // Slice GradOutput (Channel)
      DenseTensor grad_output_g;
      phi::SliceKernel<T, Context>(dev_ctx,
                                   grad_output_cont,
                                   {channel_dim},
                                   {g * out_g_sz},
                                   {(g + 1) * out_g_sz},
                                   {1},
                                   {},
                                   &grad_output_g);

      // Slice Input (Channel)
      DenseTensor input_g;
      phi::SliceKernel<T, Context>(dev_ctx,
                                   input_cont,
                                   {channel_dim},
                                   {g * in_g_sz},
                                   {(g + 1) * in_g_sz},
                                   {1},
                                   {},
                                   &input_g);

      // Slice Weight (Output Channel / dim 0)
      DenseTensor weight_g;
      phi::SliceKernel<T, Context>(dev_ctx,
                                   weight_cont,
                                   {0},
                                   {g * out_g_sz},
                                   {(g + 1) * out_g_sz},
                                   {1},
                                   {},
                                   &weight_g);

      DenseTensor grad_input_g_tensor;
      DenseTensor grad_weight_g_tensor;
      DenseTensor grad_bias_g_tensor;

      if (t_input_grad_ptr) {
        auto g_shape = t_input_grad_ptr->dims();
        g_shape[channel_dim] = in_g_sz;
        grad_input_g_tensor.Resize(g_shape);
      }
      if (t_filter_grad_ptr) {
        auto w_shape = t_filter_grad_ptr->dims();
        w_shape[0] = out_g_sz;
        grad_weight_g_tensor.Resize(w_shape);
      }
      if (t_bias_grad_ptr) {
        auto b_shape = t_bias_grad_ptr->dims();
        b_shape[0] = out_g_sz;
        grad_bias_g_tensor.Resize(b_shape);
      }

      SlowConvBackwardNoGroup<T, Context, dim>(
          dev_ctx,
          grad_output_g,
          input_g,
          weight_g,
          to_int64_vec(ksize),
          to_int64_vec(strides),
          to_int64_vec(paddings),
          to_int64_vec(dilations),
          (t_input_grad_ptr ? &grad_input_g_tensor : nullptr),
          (t_filter_grad_ptr ? &grad_weight_g_tensor : nullptr),
          (t_bias_grad_ptr ? &grad_bias_g_tensor : nullptr));

      if (t_input_grad_ptr) grad_inputs_g[g] = grad_input_g_tensor;
      if (t_filter_grad_ptr) grad_weights_g[g] = grad_weight_g_tensor;
      if (t_bias_grad_ptr) grad_biases_g[g] = grad_bias_g_tensor;
    }

    // Concat Input Grad
    if (t_input_grad_ptr) {
      std::vector<const DenseTensor*> ptrs;
      for (auto& t : grad_inputs_g) ptrs.push_back(&t);
      phi::ConcatKernel<T, Context>(
          dev_ctx, ptrs, channel_dim, t_input_grad_ptr);
    }

    // Concat Weight Grad
    if (t_filter_grad_ptr) {
      std::vector<const DenseTensor*> ptrs;
      for (auto& t : grad_weights_g) ptrs.push_back(&t);
      phi::ConcatKernel<T, Context>(dev_ctx, ptrs, 0, t_filter_grad_ptr);
    }

    // Concat Bias Grad
    if (t_bias_grad_ptr) {
      std::vector<const DenseTensor*> ptrs;
      for (auto& t : grad_biases_g) ptrs.push_back(&t);
      phi::ConcatKernel<T, Context>(dev_ctx, ptrs, 0, t_bias_grad_ptr);
    }
  }

  if (channel_last && input_grad) {
    TransToChannelLast<Context, T>(dev_ctx, t_input_grad_ptr, input_grad);
  }
}

}  // namespace phi
