// Copyright (c) 2025 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_elementwise_put_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {

template <typename T, typename Context, typename IndexT = int>
void XPUIndexElementwisePutGradKernel(
    const Context& dev_ctx,
    const DenseTensor& out_grad,
    const std::vector<const DenseTensor*>& index,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad,
    DenseTensor* value_grad) {
  int64_t numel = 0;

  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  auto sizes = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, phi::DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  std::vector<int64_t> value_dims;
  std::vector<int64_t> value_strides;
  // default value_ele_size when value_grad is nullptr
  int64_t value_ele_size = 4;
  if (value_grad) {
    value_dims = vectorize<int64_t>(value_grad->dims());
    value_strides = vectorize<int64_t>(value_grad->strides());
    value_ele_size = phi::SizeOf(value_grad->dtype());
  }

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(out_grad.dtype()),
                           value_dims,
                           value_strides,
                           value_ele_size,
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);

  if (value_grad != nullptr) {
    const int64_t N = value_grad->numel();
    PADDLE_ENFORCE_EQ(true,
                      (N >= 0 && N <= std::numeric_limits<int32_t>::max()),
                      common::errors::PreconditionNotMet(
                          "the numel of input or output should be in [0, "
                          "std::numeric_limits<int32_t>::max()]"));
  }
  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeIndexT = typename XPUTypeTrait<IndexT>::Type;
  // passed vector params for XPU
  std::vector<const XPUTypeIndexT*> index_ptrs_vec;
  std::vector<int64_t> index_numel_vec;
  for (int i = 0; i < num_indices; i++) {
    // since XPU WRAPPER_CHECK_PTR only supports original GM ptrs, so we pass
    // the IndexT* type ptrs, which is different from the CPU/GPU's char* ptr.
    index_ptrs_vec.push_back(
        reinterpret_cast<const XPUTypeIndexT*>(index[i]->data<IndexT>()));
    // index_numel_vec is for the length of WRAPPER_CHECK_PTR
    index_numel_vec.push_back(index[i]->numel());
  }
  std::vector<int64_t> sizes_vec =
      std::vector<int64_t>(sizes.begin(), sizes.begin() + num_indices);
  std::vector<int64_t> orig_strides_vec =
      std::vector<int64_t>(strides.begin(), strides.begin() + num_indices);
  std::vector<std::vector<int64_t>> strides_vec_vec =
      std::vector<std::vector<int64_t>>(strides_vec.begin(), strides_vec.end());

  const XPUType* out_grad_ptr =
      reinterpret_cast<const XPUType*>(out_grad.data<T>());
  XPUType* x_grad_ptr = x_grad == nullptr
                            ? nullptr
                            : reinterpret_cast<XPUType*>(x_grad->data<T>());
  XPUType* value_grad_ptr =
      value_grad == nullptr ? nullptr
                            : reinterpret_cast<XPUType*>(value_grad->data<T>());

  int r = xpu::index_elementwise_put_grad<XPUType, XPUTypeIndexT>(
      dev_ctx.x_context(),  // ctx
      out_grad_ptr,         // out_grad
      input_dims,           // input_shape
      index_ptrs_vec,       // index_list
      index_numel_vec,      // index_numel
      desired_shape,        // desired_shape
      sizes_vec,            // sizes
      orig_strides_vec,     // orig_strides
      strides_vec_vec,      // strides_vec
      slice_offset,         // slice_offset
      numel,                // numel
      x_grad_ptr,           // x_grad
      value_grad_ptr        // value_grad
  );
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_elementwise_put_grad");
}

template <typename T, typename Context>
void LaunchIndexElementwisePutWithTensorGradXPUKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* value_grad,
    DenseTensor* x_grad) {
  if (x_grad && !value_grad) {
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

    XPUIndexElementwisePutGradKernel<T, Context, int64_t>(dev_ctx,
                                                          out_grad,
                                                          indices,
                                                          input_dims,
                                                          input_strides,
                                                          index_dims,
                                                          index_strides,
                                                          slice_offset,
                                                          x_grad,
                                                          value_grad);
  } else if (value_grad) {
    if (x_grad) {
      Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    }
    if (value_grad->numel() == 1) {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(make_ddim(input_dims));
      dev_ctx.template Alloc<T>(&tmp_value_grad);

      XPUIndexElementwisePutGradKernel<T, Context, int64_t>(dev_ctx,
                                                            out_grad,
                                                            indices,
                                                            input_dims,
                                                            input_strides,
                                                            index_dims,
                                                            index_strides,
                                                            slice_offset,
                                                            x_grad,
                                                            &tmp_value_grad);

      std::vector<int> v_dims(tmp_value_grad.dims().size());
      std::iota(v_dims.begin(), v_dims.end(), 0);
      IntArray v_axis(v_dims);
      SumKernel<T, Context>(dev_ctx,
                            tmp_value_grad,
                            v_axis,
                            value_grad->dtype(),
                            false,
                            value_grad);
    } else if (value_grad->dims() == make_ddim(input_dims)) {
      dev_ctx.template Alloc<T>(value_grad);
      XPUIndexElementwisePutGradKernel<T, Context, int64_t>(dev_ctx,
                                                            out_grad,
                                                            indices,
                                                            input_dims,
                                                            input_strides,
                                                            index_dims,
                                                            index_strides,
                                                            slice_offset,
                                                            x_grad,
                                                            value_grad);
    } else {
      DenseTensor tmp_value_grad(value_grad->dtype());
      tmp_value_grad.Resize(make_ddim(input_dims));
      dev_ctx.template Alloc<T>(&tmp_value_grad);

      XPUIndexElementwisePutGradKernel<T, Context, int64_t>(dev_ctx,
                                                            out_grad,
                                                            indices,
                                                            input_dims,
                                                            input_strides,
                                                            index_dims,
                                                            index_strides,
                                                            slice_offset,
                                                            x_grad,
                                                            &tmp_value_grad);

      std::vector<int64_t> after_dims = vectorize(tmp_value_grad.dims());
      std::vector<int64_t> before_dims = vectorize(value_grad->dims());
      std::vector<int64_t> compress_dims;
      std::vector<int64_t> dims_without_1;

      funcs::CalCompressedDimsWith1AndWithout1(
          &after_dims, &before_dims, &compress_dims, &dims_without_1);

      auto pre_dims = value_grad->dims();
      value_grad->Resize(make_ddim(dims_without_1));
      IntArray v_axis(compress_dims);
      SumKernel<T, Context>(dev_ctx,
                            tmp_value_grad,
                            v_axis,
                            value_grad->dtype(),
                            false,
                            value_grad);
      value_grad->Resize(pre_dims);
    }
  }
}

template <typename T, typename Context>
void LaunchIndexElementwisePutGradXPUKernel(
    const Context& dev_ctx,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad) {
  if (x_grad) {
    Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);

    XPUIndexElementwisePutGradKernel<T, Context, int64_t>(dev_ctx,
                                                          out_grad,
                                                          indices,
                                                          input_dims,
                                                          input_strides,
                                                          index_dims,
                                                          index_strides,
                                                          slice_offset,
                                                          x_grad,
                                                          nullptr);
  }
}

template <typename T, typename Context>
void IndexElementwisePutGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad) {
  const auto& index_type = indices[0]->dtype();
  PADDLE_ENFORCE_EQ(
      index_type == phi::DataType::INT64 ||
          (index_type == phi::DataType::BOOL && indices.size() == 1),
      true,
      common::errors::InvalidArgument(
          "Index holds the wrong type, it holds [%s], but "
          "desires to be [%s].",
          index_type,
          phi::DataType::INT64));
  std::vector<DenseTensor> tmp_args;
  if (indices.empty()) {
    if (x_grad) {
      Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    }
    return;
  }

  LaunchIndexElementwisePutGradXPUKernel<T, Context>(dev_ctx,
                                                     indices,
                                                     out_grad,
                                                     input_dims,
                                                     input_strides,
                                                     index_dims,
                                                     index_strides,
                                                     slice_offset,
                                                     x_grad);
}

template <typename T, typename Context>
void IndexElementwisePutWithTensorGradKernel(
    const Context& dev_ctx,
    const DenseTensor& x,
    const std::vector<const DenseTensor*>& indices,
    const DenseTensor& value,
    const DenseTensor& out_grad,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    DenseTensor* x_grad,
    DenseTensor* value_grad) {
  const auto& index_type = indices[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT64));

  std::vector<DenseTensor> tmp_args;
  if (indices.empty()) {
    if (x_grad) {
      Copy(dev_ctx, out_grad, dev_ctx.GetPlace(), false, x_grad);
    }
    if (value_grad) {
      Full<T, Context>(dev_ctx, value_grad->dims(), 0.0f, value_grad);
    }
    return;
  }

  LaunchIndexElementwisePutWithTensorGradXPUKernel<T, Context>(dev_ctx,
                                                               indices,
                                                               out_grad,
                                                               input_dims,
                                                               input_strides,
                                                               index_dims,
                                                               index_strides,
                                                               slice_offset,
                                                               value_grad,
                                                               x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_put_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutGradKernel,
                   bool,
                   float,
                   double,
                   int,
                   int8_t,
                   int64_t,
                   int16_t,
                   uint8_t,
                   phi::float16,
                   phi::bfloat16) {}

PD_REGISTER_KERNEL(index_elementwise_put_with_tensor_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexElementwisePutWithTensorGradKernel,
                   bool,
                   float,
                   int,
                   int8_t,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}
