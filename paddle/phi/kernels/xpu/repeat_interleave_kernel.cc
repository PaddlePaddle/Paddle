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

#include <climits>
#include <numeric>
#include <utility>
#include <vector>

#include "paddle/phi/kernels/repeat_interleave_kernel.h"

#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/utils/data_type.h"
#include "paddle/phi/core/visit_type.h"

namespace phi {


template <typename T, typename Context>
void RepeatInterleaveKernel(const Context& ctx,
                            const DenseTensor & x,
                            const int repeats,
                            int dim,
                            DenseTensor* out) {

 PADDLE_ENFORCE_GT(repeats,
                    0,
                    phi::errors::InvalidArgument(
                        "repeats must grater than 0, but got %d", repeats));


  using XPUType = typename XPUTypeTrait<T>::Type;
  xpu::ctx_guard RAII_GUARD(ctx.x_context());
  int ret = 0;

  const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
  XPUType* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
  int64_t x_len = 1;
  int64_t idx_len = (int64_t) repeats;
  std::vector<int32_t> repeat_vec(x_len);
  repeat_vec[0] = (int32_t) idx_len;
  std::vector<int> xshape(x.dims().size());
  for (int i = 0; i < (int)xshape.size();i++) {
    xshape[i] = x.dims()[i];
  }
  auto* rep_data = RAII_GUARD.alloc_l3_or_gm<int32_t>(x_len);
  auto* idx_data = RAII_GUARD.alloc_l3_or_gm<int32_t>(idx_len);
  ret = xpu_memcpy(rep_data,
                   repeat_vec.data(),
                   repeat_vec.size() * sizeof(int32_t),
                   XPUMemcpyKind::XPU_HOST_TO_DEVICE);
  ret = xpu::repeat_interleave<int32_t>(ctx.x_context(),
                                        rep_data,
                                        idx_data,
                        x_len,
                      idx_len);
  ret = xpu::gather<XPUType, int32_t>(ctx.x_context(),
                    x_data,
                          idx_data,
                          out_data,
                                      xshape,
                                      idx_len,
                                      dim);
  PADDLE_ENFORCE_EQ(
      ret,
      xpu::Error_t::SUCCESS,
      phi::errors::External(
          "XPU gather kernel return wrong value[%d %s]", ret, XPUAPIErrorMsg[ret]));
}

template <typename T, typename Context>
void RepeatInterleaveWithTensorIndexKernel(const Context& ctx,
                            const DenseTensor & x,
                            const DenseTensor& repeats_tensor,
                            int dim,
                            DenseTensor* out) {

  PADDLE_ENFORCE_EQ(repeats_tensor.dims()[0] == x.dims()[dim],
                    true,
                    phi::errors::InvalidArgument(
                        "The length of Input(RepeatsTensor) must be the "
                        "same as length of Input(X) in axis. "
                        "But received: [%s], required: [%d].",
                        repeats_tensor.dims()[0],
                        x.dims()[dim]));

   using XPUType = typename XPUTypeTrait<T>::Type;
   xpu::ctx_guard RAII_GUARD(ctx.x_context());
   const auto & index_type = repeats_tensor.dtype();
   int ret = 0;
   bool index_type_match = index_type == DataType::INT32 || index_type == DataType::INT64;
   PADDLE_ENFORCE_EQ(
     index_type_match,
     true,
     phi::errors::InvalidArgument(
         "Input(RepeatsTensor) holds the wrong type, it holds %s, but "
         "desires to be %s or %s",
         DataTypeToString(index_type),
         DataTypeToString(phi::DataType::INT32),
         DataTypeToString(phi::DataType::INT64)));

   const XPUType* x_data = reinterpret_cast<const XPUType*>(x.data<T>());
   XPUType* out_data = reinterpret_cast<XPUType*>(ctx.template Alloc<T>(out));
   int64_t x_len = repeats_tensor.dims()[0];
   int64_t idx_len = 0;

   std::vector<int> xshape(x.dims().size());
   for (int i = 0; i < (int)xshape.size();i++) {
     xshape[i] = x.dims()[i];
   }
   if (index_type == phi::DataType::INT64) {
     std::vector<int64_t> idx_vec(x_len);
     ret = xpu_memcpy(idx_vec.data(),
          repeats_tensor.data<int64_t>(),
                      idx_vec.size() * sizeof(int64_t),
                      XPUMemcpyKind::XPU_DEVICE_TO_HOST);
     for (int i = 0; i < (int)idx_vec.size(); i++) {
       idx_len += idx_vec[i];
     }
     auto* idx_data = RAII_GUARD.alloc_l3_or_gm<int64_t>(idx_len);
     ret = xpu::repeat_interleave<int64_t>(ctx.x_context(),
                                           repeats_tensor.data<int64_t>(),
                                           idx_data,
                              x_len,
                          idx_len);
     ret = xpu::gather<XPUType, int64_t>(ctx.x_context(),
                x_data,
                      idx_data,
                      out_data,
                                  xshape,
                                  idx_len,
                                  dim);

   }
   else if (index_type == phi::DataType::INT32) {
     std::vector<int32_t> idx_vec(x_len);
     ret = xpu_memcpy(idx_vec.data(),
                      repeats_tensor.data<int32_t>(),
                      idx_vec.size() * sizeof(int32_t),
                      XPUMemcpyKind::XPU_DEVICE_TO_HOST);
     for (int i = 0; i < (int)idx_vec.size(); i++) {
       idx_len += (int64_t)idx_vec[i];
     }
     auto* idx_data = RAII_GUARD.alloc_l3_or_gm<int32_t>(idx_len);
     ret = xpu::repeat_interleave<int32_t>(ctx.x_context(),
                                           repeats_tensor.data<int32_t>(),
                                           idx_data,
                             x_len,
                         idx_len);
     ret = xpu::gather<XPUType, int32_t>(ctx.x_context(),
           x_data,
                 idx_data,
                 out_data,
                                         xshape,
                                         idx_len,
                                         dim);
  }
  PADDLE_ENFORCE_EQ(
      ret,
      xpu::Error_t::SUCCESS,
      phi::errors::External(
          "XPU gather kernel return wrong value[%d %s]", ret, XPUAPIErrorMsg[ret]));
}
}  // namespaec phi

PD_REGISTER_KERNEL(repeat_interleave,
                   XPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t);

PD_REGISTER_KERNEL(repeat_interleave_with_tensor_index,
                   XPU,
                   ALL_LAYOUT,
                   phi::RepeatInterleaveWithTensorIndexKernel,
                   float,
                   phi::dtype::float16,
                   int,
                   int64_t);
