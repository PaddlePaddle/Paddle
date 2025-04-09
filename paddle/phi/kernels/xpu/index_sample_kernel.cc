// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/index_sample_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T, typename Context>
void IndexSampleKernel(const Context& ctx,
                       const DenseTensor& x,
                       const DenseTensor& index,
                       DenseTensor* out) {
  auto index_type = index.dtype();
  bool index_type_match =
      index_type == DataType::INT32 || index_type == DataType::INT64;
  PADDLE_ENFORCE_EQ(index_type_match,
                    true,
                    errors::InvalidArgument(
                        "Input(Index) holds the wrong type, it holds %s, but "
                        "desires to be %s or %s",
                        DataTypeToString(index_type),
                        DataTypeToString(DataType::INT32),
                        DataTypeToString(DataType::INT64)));

  using XPUType = typename XPUTypeTrait<T>::Type;

  auto input_dim = x.dims();
  auto index_dim = index.dims();
  int64_t batch_size = input_dim[0];
  int64_t input_length = input_dim[1];
  int64_t index_length = index_dim[1];

  const T* in_data = x.data<T>();
  T* out_data = ctx.template Alloc<T>(out);

  // template<typename T, typename TID> DLL_EXPORT int gather_element(Context*
  // ctx, const T* x, const TID* index, T* y, const std::vector<int64_t>&
  // xshape, const std::vector<int64_t>& idxshape, int64_t axis);

  if (index_type == DataType::INT64) {
    const int64_t* index_data = index.data<int64_t>();
    int r =
        xpu::gather<XPUType, int64_t>(ctx.x_context(),
                                      reinterpret_cast<const XPUType*>(in_data),
                                      index_data,
                                      reinterpret_cast<XPUType*>(out_data),
                                      {batch_size, input_length},
                                      {batch_size, index_length},
                                      1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");
  } else if (index_type == DataType::INT32) {
    const int* index_data = index.data<int>();
    int r =
        xpu::gather<XPUType, int32_t>(ctx.x_context(),
                                      reinterpret_cast<const XPUType*>(in_data),
                                      index_data,
                                      reinterpret_cast<XPUType*>(out_data),
                                      {batch_size, input_length},
                                      {batch_size, index_length},
                                      1);
    PADDLE_ENFORCE_XDNN_SUCCESS(r, "gather");
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(index_sample,
                   XPU,
                   ALL_LAYOUT,
                   phi::IndexSampleKernel,
                   phi::dtype::bfloat16,
                   phi::dtype::float16,
                   float,
                   int8_t,
                   int16_t,
                   int32_t,
                   bool) {}
