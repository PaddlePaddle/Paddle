/* Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/phi/kernels/funcs/fake_quantize_grad_functor.h"
// #include "paddle/phi/kernels/funcs/reduce_function.h"
#include "paddle/phi/common/memory_utils.h"
#include "paddle/phi/kernels/funcs/common_shape.h"
#include "paddle/phi/kernels/reduce_sum_kernel.h"

namespace phi {
namespace funcs {

template <typename T>
struct QuantizeDataType {
  using type = T;
};

template <>
struct QuantizeDataType<phi::dtype::float16> {
  using type = float;
};

template <typename T>
__global__ void QuantizeDequantizeGradLSQKernel(const T *in,
                                                const T *scale,
                                                const T *out_grad,
                                                const float lsq_factor,
                                                const int bin_cnt,
                                                const int round_type,
                                                const int n,
                                                T *x_grad_data,
                                                T *scale_grad_data) {
  int bid = threadIdx.x + blockIdx.x * blockDim.x;
  int tid = threadIdx.x;

  using ComputeDataType = typename QuantizeDataType<T>::type;
  ComputeDataType s = static_cast<ComputeDataType>(scale[0]);
  ComputeDataType lsq_f = static_cast<ComputeDataType>(lsq_factor);
  ComputeDataType inv_s = inverse(s);
  ComputeDataType bin_cnt_t = static_cast<ComputeDataType>(bin_cnt);

  ComputeDataType max_bound = bin_cnt_t;
  ComputeDataType min_bound = -bin_cnt_t - static_cast<ComputeDataType>(1);

  for (int i = bid; i < n; i += blockDim.x * gridDim.x) {
    ComputeDataType x = static_cast<ComputeDataType>(in[i]);
    ComputeDataType y_grad = static_cast<ComputeDataType>(out_grad[i]);

    if (y_grad < min_bound || y_grad > max_bound) {
      x_grad_data[i] = static_cast<T>(0);
    } else {
      x_grad_data[i] = static_cast<T>(y_grad);
    }
    ComputeDataType x_quant_round = x;
    ComputeDataType x_quant = x * inv_s;
    if (round_type == 0) {
      ComputeDataType x_0 = roundWithTiesToEven(x_quant);
      x_0 = x_0 > max_bound ? max_bound : x_0;
      x_0 = x_0 < min_bound ? min_bound : x_0;
      x_quant_round = x_0;
    } else {
      ComputeDataType x_1 = round(x_quant);
      x_1 = x_1 > max_bound ? max_bound : x_1;
      x_1 = x_1 < min_bound ? min_bound : x_1;
      x_quant_round = x_1;
    }
    ComputeDataType elem = x_quant_round - x_quant;
    if (x_quant < min_bound) {
      elem = min_bound;
    } else if (x_quant > max_bound) {
      elem = max_bound;
    }
    elem = elem * y_grad * lsq_f;
    scale_grad_data[i] = static_cast<T>(elem);
  }
}

template <typename Context, typename T>
void FakeQuantizeDequantizeGradLSQFunctor<Context, T>::operator()(
    const Context &dev_ctx,
    const DenseTensor &x,
    const DenseTensor &scale,
    const DenseTensor &out_grad,
    const float lsq_factor,
    const int bin_cnt,
    const int round_type,
    DenseTensor *x_grad,
    DenseTensor *scale_grad) {
  int num = x.numel();
  int block = 1024;
  int grid = (block - 1 + num) / block;

  const T *in_data = x.data<T>();
  const T *scale_data = scale.data<T>();
  const T *out_grad_data = out_grad.data<T>();

  T *x_grad_data = dev_ctx.template Alloc<T>(x_grad);

  DenseTensor scale_grad_elem;
  scale_grad_elem.Resize({x.dims()});
  T *scale_grad_elem_data = dev_ctx.template Alloc<T>(
      &scale_grad_elem, scale_grad_elem.numel() * sizeof(T));

  QuantizeDequantizeGradLSQKernel<T>
      <<<grid, block, 0, dev_ctx.stream()>>>(in_data,
                                             scale_data,
                                             out_grad_data,
                                             lsq_factor,
                                             bin_cnt,
                                             round_type,
                                             num,
                                             x_grad_data,
                                             scale_grad_elem_data);

  dev_ctx.template Alloc<T>(scale_grad);

  std::vector<int> v_dims(x.dims().size());
  std::iota(v_dims.begin(), v_dims.end(), 0);
  IntArray reduce_dims(v_dims);

  phi::SumKernel<T, Context>(
      dev_ctx, scale_grad_elem, reduce_dims, x.dtype(), 0, scale_grad);
  scale_grad->Resize(scale.dims());
}

template class FakeQuantizeDequantizeGradLSQFunctor<GPUContext, float16>;
template class FakeQuantizeDequantizeGradLSQFunctor<GPUContext, float>;
template class FakeQuantizeDequantizeGradLSQFunctor<GPUContext, double>;

}  // namespace funcs
}  // namespace phi
