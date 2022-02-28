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

#include "paddle/phi/kernels/linspace_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
// #include "paddle/phi/kernels/funcs/trans_data_type.h"

namespace phi {

template <typename T, typename Context>
void LinspaceKernel(const Context& ctx,
                    const DenseTensor& start,
                    const DenseTensor& stop,
                    const DenseTensor& number,
                    int dtype,
                    DenseTensor* out) {
  int32_t num = number.data<int32_t>()[0];
  auto dtype_var = static_cast<paddle::framework::proto::VarType::Type>(dtype);

  DenseTensor start_t;
  DenseTensor stop_t;
  auto start_dtype = paddle::framework::OpKernelType(
      paddle::framework::TransToProtoVarType(start.dtype()), ctx.GetPlace());
  auto stop_dtype = paddle::framework::OpKernelType(
      paddle::framework::TransToProtoVarType(stop.dtype()), ctx.GetPlace());
  auto out_dtype = paddle::framework::OpKernelType(dtype_var, ctx.GetPlace());
  // phi::funcs::TransDataType(start_dtype, out_dtype, start, &start_t);
  // phi::funcs::TransDataType(stop_dtype, out_dtype, stop, &stop_t);

  T start_data = start_t.data<T>()[0];
  T stop_data = stop_t.data<T>()[0];
  PADDLE_ENFORCE_GT(
      num,
      0,
      phi::errors::InvalidArgument("The num of linspace op should be larger "
                                   "than 0, but received num is %d",
                                   num));

  out->Resize(phi::make_ddim({num}));
  T* out_data = ctx.template Alloc<T>(out);

  if (num > 1) {
    // step should be of double type for all types
    double step = (static_cast<double>(stop_data - start_data)) / (num - 1);
    int half_num = num / 2;
    for (int i = 0; i < num; ++i) {
      if (i < half_num) {
        out_data[i] = static_cast<T>(start_data + step * i);
      } else {
        out_data[i] = static_cast<T>(stop_data - step * (num - i - 1));
      }
    }
  } else {
    out_data[0] = static_cast<T>(start_data);
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(linspace,
                   CPU,
                   ALL_LAYOUT,
                   phi::LinspaceKernel,
                   float,
                   int32_t,
                   int64_t,
                   double) {}
