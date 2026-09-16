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

#include "paddle/phi/kernels/index_elementwise_get_kernel.h"

#include "paddle/phi/backends/xpu/xpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/index_elementwise.h"
#include "paddle/phi/kernels/funcs/index_put_utils.h"
#include "paddle/phi/kernels/funcs/stride_utils.h"
#include "paddle/phi/kernels/xpu/index_put_xpu_utils.h"

namespace phi {
template <typename T, typename Context, typename IndexT = int>
void XPUIndexElementwiseGetGradKernel(
    const Context& dev_ctx,
    const DenseTensor& input,
    const DenseTensor& value,
    const std::vector<const DenseTensor*>& index,
    const std::vector<int64_t>& input_dims,
    const std::vector<int64_t>& input_strides,
    const std::vector<int64_t>& index_dims,
    const std::vector<int64_t>& index_strides,
    const int64_t slice_offset,
    const bool accumulate,
    DenseTensor* output) {
  int64_t numel = 0;
  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  auto sizes = std::array<int64_t, DDim::kMaxRank + 1>{};
  auto strides = std::array<int64_t, DDim::kMaxRank + 1>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  auto index_ptrs = funcs::GetIndexDataPtrs<IndexT>(index);

  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;

  funcs::IndexPutStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(input.dtype()),
                           vectorize<int64_t>(value.dims()),
                           vectorize<int64_t>(value.strides()),
                           phi::SizeOf(value.dtype()),
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);

  using XPUType = typename XPUTypeTrait<T>::Type;
  using XPUTypeIndexT = typename XPUTypeTrait<IndexT>::Type;

  const XPUType* value_ptr = reinterpret_cast<const XPUType*>(value.data<T>());
  std::vector<const XPUTypeIndexT*> index_list_vec;
  std::vector<int64_t> index_numel;
  for (int i = 0; i < num_indices; i++) {
    index_list_vec.push_back(
        reinterpret_cast<const XPUTypeIndexT*>(index[i]->data<IndexT>()));
    index_numel.push_back(index[i]->numel());
  }
  std::vector<int64_t> sizes_vec =
      std::vector<int64_t>(sizes.begin(), sizes.begin() + num_indices);
  std::vector<int64_t> orig_strides_vec =
      std::vector<int64_t>(strides.begin(), strides.begin() + num_indices);
  std::vector<std::vector<int64_t>> strides_vec_vec =
      std::vector<std::vector<int64_t>>(strides_vec.begin(), strides_vec.end());

  XPUType* output_ptr = reinterpret_cast<XPUType*>(output->data<T>());

  // When accumulate=true and slice_offset=0 (the backward pass for simple
  // advanced / boolean indexing), xpu::index_elementwise_get_grad does NOT
  // use atomic operations for scatter-add, leading to race conditions on
  // duplicate indices.  Additionally its int32 instantiation produces garbage
  // values.  Replace with XPUDealWithIndices + xpu::scatter_nd (is_overwrite
  // = false), which is the same atomically-correct scatter-add primitive used
  // by index_put_grad_kernel.cc.
  //
  // For int64_t output type, xpu::index_elementwise_get_grad does NOT have a
  // <long, long> specialization in the XPU SDK, so we always use scatter_nd
  // for the int64_t case regardless of accumulate/slice_offset.
  constexpr bool kIsInt64 = std::is_same<T, int64_t>::value;
  if ((accumulate && slice_offset == 0) || kIsInt64) {
    // Merge per-dimension index tensors into a single [N, num_indices] tensor.
    auto bd_dims = funcs::BroadCastTensorsDims(index);
    DenseTensor res_indices(DataType::INT64);
    XPUDealWithIndices<Context>(dev_ctx, index, bd_dims, &res_indices);
    auto index_shape = vectorize<int64_t>(res_indices.dims());

    auto xshape = vectorize<int64_t>(output->dims());
    xpu::VectorParam<int64_t> xshape_param = {
        xshape.data(), static_cast<int64_t>(xshape.size()), nullptr};

    auto index_data = const_cast<int64_t*>(res_indices.data<int64_t>());
    xpu::VectorParam<int64_t> index_vec{
        nullptr, res_indices.numel(), index_data};

    // scatter_nd with x=nullptr and is_overwrite=false performs atomic
    // scatter-add: out[index[j]] += updates[j].
    // scatter_nd with x=nullptr and is_overwrite=true performs scatter-
    // overwrite: out[index[j]] = updates[j].
    // Since output is pre-zeroed by set_constant, x=nullptr is safe for both.
    bool is_overwrite = !accumulate;
    int r = xpu::scatter_nd<XPUType, int64_t>(dev_ctx.x_context(),
                                              nullptr,
                                              value_ptr,
                                              output_ptr,
                                              index_vec,
                                              xshape_param,
                                              index_shape,
                                              is_overwrite);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "scatter_nd (index_elementwise_get_grad)");
    return;
  }

  // Fallback path for non-accumulate or non-zero slice_offset cases
  // (only reached for float, int, float16, bfloat16 types which have
  // xpu::index_elementwise_get_grad specializations in the XPU SDK).
  int r = xpu::index_elementwise_get_grad<XPUType, XPUTypeIndexT>(
      dev_ctx.x_context(),
      value_ptr,
      input_dims,
      index_list_vec,
      index_numel,
      desired_shape,
      sizes_vec,
      orig_strides_vec,
      strides_vec_vec,
      slice_offset,
      numel,
      accumulate,
      output_ptr);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_elementwise_get_grad");
}

template <typename T, typename Context>
void IndexElementwiseGetGradKernel(const Context& dev_ctx,
                                   const DenseTensor& x,
                                   const std::vector<const DenseTensor*>& index,
                                   const DenseTensor& out_grad,
                                   const std::vector<int64_t>& input_dims,
                                   const std::vector<int64_t>& input_strides,
                                   const std::vector<int64_t>& index_dims,
                                   const std::vector<int64_t>& index_strides,
                                   const int64_t slice_offset,
                                   const bool accumulate,
                                   const bool is_combined,
                                   DenseTensor* x_grad) {
  dev_ctx.template Alloc<T>(x_grad);
  funcs::set_constant(dev_ctx, x_grad, static_cast<float>(0));
  if (out_grad.numel() == 0) return;

  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        DataType::INT64));

  XPUIndexElementwiseGetGradKernel<T, Context, int64_t>(dev_ctx,
                                                        x,
                                                        out_grad,
                                                        index,
                                                        input_dims,
                                                        input_strides,
                                                        index_dims,
                                                        index_strides,
                                                        slice_offset,
                                                        accumulate,
                                                        x_grad);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_get_grad,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseGetGradKernel,
                   float,
                   int,
                   int64_t,
                   phi::float16,
                   phi::bfloat16) {}
