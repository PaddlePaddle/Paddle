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

#include "paddle/phi/kernels/put_along_axis_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/common/place.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/kernels/funcs/gather_scatter_functor.h"

namespace phi {

template <typename T, typename Context>
void PutAlongAxisKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        const DenseTensor& indices,
                        const DenseTensor& values,
                        int axis,
                        const std::string& reduce,
                        bool include_self,
                        DenseTensor* out) {
  Copy(dev_ctx, x, dev_ctx.GetPlace(), false, out);

  // Brings the operands into the representation the scatter functor can
  // address: a 0-D operand becomes rank 1 and ``axis`` is normalized.
  // ``index`` and ``value`` are shallow views sharing the caller's buffer.
  // ``out`` is promoted in place because the scatter writes through it, so its
  // shape is saved here and restored once the scatter is done -- reading it
  // back from ``x`` would not do, since ``x`` and ``out`` are the same tensor
  // when the op runs inplace.
  const DDim out_dims = out->dims();
  DenseTensor index = indices;
  DenseTensor value = values;
  const auto& index_type = index.dtype();
  funcs::PreparePutAlongAxisOperands(out, &index, &value, &axis);

  if (reduce == "add") {
    if (index_type == DataType::INT32) {
      funcs::cpu_scatter_add_kernel<T, int32_t>(
          *out, axis, index, value, include_self, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::cpu_scatter_add_kernel<T, int64_t>(
          *out, axis, index, value, include_self, dev_ctx);
    }
  } else if (reduce == "multiply" || reduce == "mul") {
    if (index_type == DataType::INT32) {
      funcs::cpu_scatter_mul_kernel<T, int32_t>(
          *out, axis, index, value, include_self, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::cpu_scatter_mul_kernel<T, int64_t>(
          *out, axis, index, value, include_self, dev_ctx);
    }
  } else if (reduce == "assign") {
    if (index_type == DataType::INT32) {
      funcs::cpu_scatter_assign_kernel<T, int32_t>(
          *out, axis, index, value, include_self, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::cpu_scatter_assign_kernel<T, int64_t>(
          *out, axis, index, value, include_self, dev_ctx);
    }
  } else if (reduce == "mean") {
    if (index_type == DataType::INT32) {
      funcs::cpu_scatter_mean_kernel<T, int32_t>(
          *out, axis, index, value, include_self, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::cpu_scatter_mean_kernel<T, int64_t>(
          *out, axis, index, value, include_self, dev_ctx);
    }
  } else if (reduce == "amax") {
    if (index_type == DataType::INT32) {
      funcs::cpu_scatter_max_kernel<T, int32_t>(
          *out, axis, index, value, include_self, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::cpu_scatter_max_kernel<T, int64_t>(
          *out, axis, index, value, include_self, dev_ctx);
    }
  } else if (reduce == "amin") {
    if (index_type == DataType::INT32) {
      funcs::cpu_scatter_min_kernel<T, int32_t>(
          *out, axis, index, value, include_self, dev_ctx);
    } else if (index_type == DataType::INT64) {
      funcs::cpu_scatter_min_kernel<T, int64_t>(
          *out, axis, index, value, include_self, dev_ctx);
    }
  } else {
    PADDLE_THROW(errors::InvalidArgument(
        "can not support reduce: '%s' for scatter kernel, only "
        "support reduce op: 'add', 'assign', 'mul', 'mean', 'amin', 'amax' and "
        "'multiply', the "
        "default reduce "
        "op is 'assign' ",
        reduce));
    return;
  }
  out->Resize(out_dims);
}

}  // namespace phi

PD_REGISTER_KERNEL(put_along_axis,
                   CPU,
                   ALL_LAYOUT,
                   phi::PutAlongAxisKernel,
                   float,
                   double,
                   int,
                   int16_t,
                   uint8_t,
                   int64_t) {}
