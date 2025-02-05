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

#include "paddle/phi/kernels/allclose_kernel.h"

#include <type_traits>
#include "glog/logging.h"

#include "paddle/phi/common/amp_type_traits.h"
#include "paddle/phi/common/data_type.h"
#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/kernel_registry.h"

namespace phi {

template <typename T>
__global__ void AllcloseCUDAKernel(const T* in_data,
                                   const T* other_data,
                                   const double rtol,
                                   const double atol,
                                   bool equal_nan,
                                   int num,
                                   bool* out_data) {
  unsigned int idx = threadIdx.x + blockIdx.x * blockDim.x;
  bool val;
  using BaseMPType = typename phi::dtype::MPTypeTrait<T>::Type;

  using MPType =
      typename std::conditional<std::is_same<T, int32_t>::value ||
                                    std::is_same<T, int64_t>::value ||
                                    std::is_same<T, bool>::value,
                                double,
                                BaseMPType>::type;

  for (int i = idx; i < num; i += blockDim.x * gridDim.x) {
    const MPType a = static_cast<MPType>(in_data[i]);
    const MPType b = static_cast<MPType>(other_data[i]);
    if (isnan(a) || isnan(b)) {
      val = equal_nan && isnan(a) == isnan(b);
    } else {
      MPType left = (a > b ? a - b : b - a);
      MPType right = atol + (b > 0 ? rtol * b : (-rtol) * b);
      MPType diff = (left > right ? left - right : right - left);
      val = a == b || left <= right || diff <= 1e-15;
    }
    if (!val) *out_data = false;
  }
}

template <typename T, typename Context>
void AllCloseKernel(const Context& dev_ctx,
                    const DenseTensor& x,
                    const DenseTensor& y,
                    const Scalar& rtol,
                    const Scalar& atol,
                    bool equal_nan,
                    DenseTensor* out) {
  double rtol_v, atol_v;
  if (rtol.dtype() == DataType::FLOAT64) {
    rtol_v = rtol.to<double>();
  } else if (rtol.dtype() == DataType::FLOAT32) {
    rtol_v = rtol.to<float>();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input (Rtol) type must be double or float, but get %s.",
        rtol.dtype()));
  }
  if (atol.dtype() == DataType::FLOAT64) {
    atol_v = atol.to<double>();
  } else if (atol.dtype() == DataType::FLOAT32) {
    atol_v = atol.to<float>();
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Input (Atol) type must be double or float, but get %s.",
        atol.dtype()));
  }
  VLOG(3) << "rtol and atol is : " << rtol_v << " " << atol_v;
  const T* in_data = x.data<T>();
  const T* other_data = y.data<T>();
  bool* out_data = dev_ctx.template Alloc<bool>(out);

  int num = x.numel();
  int block = 1024;
  int grid = (block - 1 + num) / block;
  grid = (grid > block) ? block : grid;
#ifdef PADDLE_WITH_HIP
  hipMemset(out_data, true, sizeof(bool));
#else
  cudaMemset(out_data, true, sizeof(bool));
#endif
  AllcloseCUDAKernel<T><<<grid, block, 0, dev_ctx.stream()>>>(
      in_data, other_data, rtol_v, atol_v, equal_nan, num, out_data);
}

}  // namespace phi

PD_REGISTER_KERNEL(allclose,
                   GPU,
                   ALL_LAYOUT,
                   phi::AllCloseKernel,
                   float,
                   double,
                   bool,
                   int,
                   int64_t,
                   phi::dtype::float16) {
  kernel->OutputAt(0).SetDataType(phi::DataType::BOOL);
}
