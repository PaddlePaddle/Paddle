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
#include "paddle/phi/kernels/funcs/stride_utils.h"

namespace phi {
template <typename T, typename Context, typename IndexT = int>
void XPUIndexElementwiseGetKernel(const Context& dev_ctx,
                                  const DenseTensor& input,
                                  const std::vector<const DenseTensor*>& index,
                                  const std::vector<int64_t>& input_dims,
                                  const std::vector<int64_t>& input_strides,
                                  const std::vector<int64_t>& index_dims,
                                  const std::vector<int64_t>& index_strides,
                                  const int64_t slice_offset,
                                  DenseTensor* output) {
  int64_t numel = 0;
  int64_t num_indices = 0;
  std::vector<int64_t> shape_tmp;
  std::vector<int64_t> stride_tmp;
  funcs::cal_shape_stride(index_dims, &num_indices, &shape_tmp, &stride_tmp);

  auto sizes = std::array<int64_t, DDim::kMaxRank>{};
  auto strides = std::array<int64_t, DDim::kMaxRank>{};
  for (int64_t i = 0; i < num_indices; i++) {
    sizes[i] = index_dims[i];
    strides[i] = index_strides[i];
  }
  std::array<int64_t*, 3> strides_array;
  std::vector<int64_t> desired_shape;
  std::array<std::vector<int64_t>, 3> strides_vec;
  funcs::IndexGetStride<3>(input_dims,
                           input_strides,
                           phi::SizeOf(input.dtype()),
                           std::vector<int64_t>(),
                           std::vector<int64_t>(),
                           phi::SizeOf(input.dtype()),
                           shape_tmp,
                           stride_tmp,
                           phi::SizeOf(index[0]->dtype()),
                           &desired_shape,
                           &strides_array,
                           &numel,
                           strides_vec);
  const int64_t N = output->numel();
  PADDLE_ENFORCE_GE(
      N, 0, common::errors::InvalidArgument("Output numel must >= 0"));
  PADDLE_ENFORCE_LE(
      N,
      std::numeric_limits<int32_t>::max(),
      common::errors::InvalidArgument("Output numel must <= INT32_MAX"));

  dev_ctx.template Alloc<T>(output);
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

  const char* in_ptr =
      reinterpret_cast<const char*>(input.data<T>()) + slice_offset;
  char* out_ptr = reinterpret_cast<char*>(output->data<T>());

  // for checkptr and checksum in XPU
  int64_t data_size_in = input.Holder()->size() - input.meta().offset;
  int64_t data_size_out = output->Holder()->size() - output->meta().offset;

  bool is_get = true;
  int r = xpu::index_elementwise_tensor<XPUType, XPUTypeIndexT>(
      dev_ctx.x_context(),
      reinterpret_cast<const XPUType*>(in_ptr),  // XPU ptr
      reinterpret_cast<XPUType*>(out_ptr),       // XPU ptr
      index_ptrs_vec,                            // vec of XPU ptrs
      index_numel_vec,                           // CPU vec
      desired_shape,                             // CPU vec
      sizes_vec,                                 // CPU vec
      orig_strides_vec,                          // CPU vec
      strides_vec_vec,                           // CPU vec
      N,                                         // int64_t
      data_size_in,                              // int64_t
      data_size_out,                             // int64_t
      is_get);                                   // true for get, false for put
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "index_elementwise_tensor_get");
}

template <typename T, typename Context>
void IndexElementwiseGetKernel(const Context& dev_ctx,
                               const DenseTensor& x,
                               const std::vector<const DenseTensor*>& index,
                               const std::vector<int64_t>& input_dims,
                               const std::vector<int64_t>& input_strides,
                               const std::vector<int64_t>& index_dims,
                               const std::vector<int64_t>& index_strides,
                               const int64_t slice_offset,
                               const bool accumulate,
                               const bool is_combined,
                               DenseTensor* out) {
  const auto& index_type = index[0]->dtype();
  PADDLE_ENFORCE_EQ(index_type == phi::DataType::INT64,
                    true,
                    common::errors::InvalidArgument(
                        "Index holds the wrong type, it holds [%s], but "
                        "desires to be [%s].",
                        index_type,
                        phi::DataType::INT64));

  auto out_dims = out->dims();
  if (out_dims.size() > 0) {
    std::vector<int64_t> output_dims(input_dims);
    out->Resize(phi::make_ddim(output_dims));
  }
  dev_ctx.template Alloc<T>(out);
  if (out->numel() == 0) return;
  XPUIndexElementwiseGetKernel<T, Context, int64_t>(dev_ctx,
                                                    x,
                                                    index,
                                                    input_dims,
                                                    input_strides,
                                                    index_dims,
                                                    index_strides,
                                                    slice_offset,
                                                    out);
}

}  // namespace phi

PD_REGISTER_KERNEL(index_elementwise_get,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexElementwiseGetKernel,
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
