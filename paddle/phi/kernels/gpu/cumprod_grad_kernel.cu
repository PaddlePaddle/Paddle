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

#include "paddle/phi/kernels/cumprod_grad_kernel.h"

#include <thrust/transform.h>

#include "paddle/fluid/operators/math/inclusive_scan.h"
#include "paddle/phi/backends/gpu/gpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/complex_functors.h"
#include "paddle/phi/kernels/funcs/cumprod.h"
#include "paddle/phi/kernels/funcs/elementwise_functor.h"
#include "paddle/phi/kernels/funcs/for_range.h"
// NOTE(@xiongkun): use of IsComplex<>
#include "paddle/fluid/framework/data_type.h"

namespace phi {

template <typename T>
struct CumprodGradFunctorExceptFirstZero {
  HOSTDEVICE CumprodGradFunctorExceptFirstZero(
      const T *x,
      const T *y,
      const T *dy_mul_y_reversed_cumsum,
      const uint8_t *zero_mask,
      size_t mid_dim,
      size_t inner_dim,
      T *dx,
      int64_t *first_zero_idx,
      T *x_filled_one)
      : x_(x),
        y_(y),
        dy_mul_y_reversed_cumsum_(dy_mul_y_reversed_cumsum),
        zero_mask_(zero_mask),
        mid_dim_(mid_dim),
        inner_dim_(inner_dim),
        dx_(dx),
        first_zero_idx_(first_zero_idx),
        x_filled_one_(x_filled_one) {}

  HOSTDEVICE void operator()(size_t idx) const {
    auto inner_idx = idx % inner_dim_;
    auto outer_idx = idx / (mid_dim_ * inner_dim_);
    auto mid_idx = (idx - inner_idx) / inner_dim_ % mid_dim_;
    auto mask = zero_mask_[idx];
    bool should_fill_one = true;

    if (mask == 0) {
      dx_[idx] = dy_mul_y_reversed_cumsum_[idx] / x_[idx];
      if (mid_idx == mid_dim_ - 1) {
        // record first zero position as -1, i.e., no zero
        first_zero_idx_[outer_idx * inner_dim_ + inner_idx] = -1;
      }
    } else if (mid_idx > 0) {                  // mask > 0
      if (zero_mask_[idx - inner_dim_] > 0) {  // not first zero
        dx_[idx] = 0;
        should_fill_one = false;
      } else {
        // idx is the first zero position, it should be recorded
        dx_[idx] = y_[idx - inner_dim_];
        first_zero_idx_[outer_idx * inner_dim_ + inner_idx] = mid_idx;
      }
    } else {  // the first zero position is index 0
      dx_[idx] = 1;
      first_zero_idx_[outer_idx * inner_dim_ + inner_idx] = 0;
    }

    x_filled_one_[idx] = should_fill_one ? 1 : x_[idx];
  }

 private:
  const T *x_;
  const T *y_;
  const T *dy_mul_y_reversed_cumsum_;
  const uint8_t *zero_mask_;
  size_t mid_dim_;
  size_t inner_dim_;
  T *dx_;
  int64_t *first_zero_idx_;
  T *x_filled_one_;
};

template <typename T>
struct FillFirstZeroPositionGradFunctor {
  HOSTDEVICE FillFirstZeroPositionGradFunctor(const int64_t *first_zero_idx,
                                              const T *grad_value,
                                              size_t mid_dim,
                                              size_t inner_dim,
                                              T *dx)
      : first_zero_idx_(first_zero_idx),
        grad_value_(grad_value),
        mid_dim_(mid_dim),
        inner_dim_(inner_dim),
        dx_(dx) {}

  HOSTDEVICE void operator()(size_t idx) const {
    auto outer_idx = idx / inner_dim_;
    auto inner_idx = idx % inner_dim_;
    auto mid_idx = first_zero_idx_[idx];
    if (mid_idx >= 0) {
      auto full_idx =
          outer_idx * mid_dim_ * inner_dim_ + mid_idx * inner_dim_ + inner_idx;
      dx_[full_idx] *= grad_value_[full_idx];
    }
  }

 private:
  const int64_t *first_zero_idx_;
  const T *grad_value_;
  size_t mid_dim_;
  size_t inner_dim_;
  T *dx_;
};

template <typename T, typename Context>
void CumprodGradKernel(const Context &dev_ctx,
                       const DenseTensor &x,
                       const DenseTensor &out,
                       const DenseTensor &dout,
                       int dim,
                       DenseTensor *dx) {
  const auto *y = &out;
  const auto *dy = &dout;

  size_t outer_dim, mid_dim, inner_dim;
  GetCumprodDimInfo(x.dims(), dim, &outer_dim, &mid_dim, &inner_dim);
  if (outer_dim == 0 || mid_dim == 0 || inner_dim == 0) return;

  size_t numel = outer_dim * mid_dim * inner_dim;

  const auto *x_data = x.data<T>();
  const auto *y_data = y->data<T>();
  const auto *dy_data = dy->data<T>();

  auto place = dev_ctx.GetPlace();
  auto *dx_data = dev_ctx.template Alloc<T>(dx);

  // deal with complex
  const T *x_data_deal;
  const T *y_data_deal;
  Allocator::AllocationPtr x_conj;
  Allocator::AllocationPtr y_conj;
  if (paddle::framework::IsComplex<T>::value) {
    x_conj = const_cast<Allocator &>(dev_ctx.GetAllocator())
                 .Allocate(numel * sizeof(T));
    auto *x_data_conj = reinterpret_cast<T *>(x_conj->ptr());
    y_conj = const_cast<Allocator &>(dev_ctx.GetAllocator())
                 .Allocate(numel * sizeof(T));
    auto *y_data_conj = reinterpret_cast<T *>(y_conj->ptr());

    phi::funcs::ForRange<Context> for_range_x(dev_ctx, numel);
    phi::funcs::ConjFunctor<T> functor_x(x_data, numel, x_data_conj);
    for_range_x(functor_x);

    phi::funcs::ForRange<Context> for_range_y(dev_ctx, numel);
    phi::funcs::ConjFunctor<T> functor_y(y_data, numel, y_data_conj);
    for_range_y(functor_y);
    x_data_deal = x_data_conj;
    y_data_deal = y_data_conj;
  } else {
    x_data_deal = x_data;
    y_data_deal = y_data;
  }

// Step 1: find cummax-ed zero mask of x
#ifdef PADDLE_WITH_CUDA
  const auto &exec_policy = thrust::cuda::par.on(dev_ctx.stream());
#else
  const auto &exec_policy = thrust::hip::par.on(dev_ctx.stream());
#endif
  auto zero_mask_without_cummax =
      const_cast<Allocator &>(dev_ctx.GetAllocator())
          .Allocate(numel * sizeof(uint8_t));
  auto *zero_mask_without_cummax_data =
      reinterpret_cast<uint8_t *>(zero_mask_without_cummax->ptr());
  thrust::transform(exec_policy,
                    thrust::device_pointer_cast(x_data_deal),
                    thrust::device_pointer_cast(x_data_deal) + numel,
                    thrust::device_pointer_cast(zero_mask_without_cummax_data),
                    funcs::IsZeroFunctor<T>());

  auto zero_mask = const_cast<Allocator &>(dev_ctx.GetAllocator())
                       .Allocate(numel * sizeof(uint8_t));
  auto *zero_mask_data = reinterpret_cast<uint8_t *>(zero_mask->ptr());
  paddle::operators::math::InclusiveScan<uint8_t, cub::Max>(
      zero_mask_without_cummax_data,
      zero_mask_data,
      outer_dim,
      mid_dim,
      inner_dim,
      static_cast<uint8_t>(0),
      cub::Max(),
      /*reverse=*/false,
      dev_ctx);
  zero_mask_without_cummax = nullptr;

  // Step 2: calculate reversed cumsum(dy * y)
  auto dy_mul_y = const_cast<Allocator &>(dev_ctx.GetAllocator())
                      .Allocate(numel * sizeof(T));
  auto *dy_mul_y_data = reinterpret_cast<T *>(dy_mul_y->ptr());
  thrust::transform(exec_policy,
                    thrust::device_pointer_cast(dy_data),
                    thrust::device_pointer_cast(dy_data) + numel,
                    thrust::device_pointer_cast(y_data_deal),
                    thrust::device_pointer_cast(dy_mul_y_data),
                    funcs::MultiplyFunctor<T>());

  auto dy_mul_y_reversed_cumsum =
      const_cast<Allocator &>(dev_ctx.GetAllocator())
          .Allocate(numel * sizeof(T));
  auto *dy_mul_y_reversed_cumsum_data =
      reinterpret_cast<T *>(dy_mul_y_reversed_cumsum->ptr());
  paddle::operators::math::InclusiveScan<T, cub::Sum>(
      dy_mul_y_data,
      dy_mul_y_reversed_cumsum_data,
      outer_dim,
      mid_dim,
      inner_dim,
      static_cast<T>(0),
      cub::Sum(),
      /*reverse=*/true,
      dev_ctx);

  // Step 3: calculate the gradient value except the first zero position.
  // The gradient value of the first zero position is filled with out[idx-1],
  // while the gradient value of the other positions are calculated out
  // completely. This functor also:
  //  (1) find the first zero index, i.e., first_zero_idx_data.
  //  (2) fill x_filled_one, which satifies
  //      x_filled_one[i] = x[i], i > pos
  //      x_filled_one[i] = 1, i <= pos
  auto first_zero_idx = const_cast<Allocator &>(dev_ctx.GetAllocator())
                            .Allocate(numel * sizeof(int64_t));
  auto *first_zero_idx_data =
      reinterpret_cast<int64_t *>(first_zero_idx->ptr());
  auto *x_filled_one_data = dy_mul_y_data;  // reuse former allocated memory
  phi::funcs::ForRange<Context> for_range(dev_ctx, numel);
  CumprodGradFunctorExceptFirstZero<T> functor_except_first_zero(
      x_data_deal,
      y_data_deal,
      dy_mul_y_reversed_cumsum_data,
      zero_mask_data,
      mid_dim,
      inner_dim,
      dx_data,
      first_zero_idx_data,
      x_filled_one_data);
  for_range(functor_except_first_zero);

  // Step 4: calculate cumprod of x_filled_one
  auto *x_filled_one_cumprod_data =
      dy_mul_y_reversed_cumsum_data;  // reuse former allocated memory
  paddle::operators::math::InclusiveScan<T, funcs::MultiplyFunctor<T>>(
      x_filled_one_data,
      x_filled_one_cumprod_data,
      outer_dim,
      mid_dim,
      inner_dim,
      static_cast<T>(1),
      funcs::MultiplyFunctor<T>(),
      /*reverse=*/false,
      dev_ctx);

  // Step 5: calculate reversed cumsum(dy * x_filled_one_cumprod)
  auto *dy_mul_x_filled_one_cumprod =
      dy_mul_y_data;  // reuse former allocated memory
  thrust::transform(exec_policy,
                    thrust::device_pointer_cast(dy_data),
                    thrust::device_pointer_cast(dy_data) + numel,
                    thrust::device_pointer_cast(x_filled_one_cumprod_data),
                    thrust::device_pointer_cast(dy_mul_x_filled_one_cumprod),
                    funcs::MultiplyFunctor<T>());
  auto *dy_mul_x_filled_one_cumprod_reversed_cumsum =
      dy_mul_y_reversed_cumsum_data;  // reuse former allocated memory
  paddle::operators::math::InclusiveScan<T, cub::Sum>(
      dy_mul_x_filled_one_cumprod,
      dy_mul_x_filled_one_cumprod_reversed_cumsum,
      outer_dim,
      mid_dim,
      inner_dim,
      static_cast<T>(0),
      cub::Sum(),
      /*reverse=*/true,
      dev_ctx);

  // Step 6: fill zero pos gradient value
  phi::funcs::ForRange<Context> for_range_fill_zero_pos_grad(
      dev_ctx, outer_dim * inner_dim);
  FillFirstZeroPositionGradFunctor<T> fill_first_zero_pos_grad_functor(
      first_zero_idx_data,
      dy_mul_x_filled_one_cumprod_reversed_cumsum,
      mid_dim,
      inner_dim,
      dx_data);
  for_range_fill_zero_pos_grad(fill_first_zero_pos_grad_functor);
}

}  // namespace phi

PD_REGISTER_KERNEL(cumprod_grad,
                   GPU,
                   ALL_LAYOUT,
                   phi::CumprodGradKernel,
                   float,
                   double,
                   int,
                   int64_t,
                   phi::dtype::complex<float>,
                   phi::dtype::complex<double>) {}
