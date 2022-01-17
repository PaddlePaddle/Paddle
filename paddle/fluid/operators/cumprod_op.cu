// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <thrust/transform.h>
#include "paddle/fluid/operators/cumprod_op.h"
#include "paddle/fluid/operators/math/complex_functors.h"
#include "paddle/fluid/operators/math/inclusive_scan.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct MultiplyFunctor {
  HOSTDEVICE T operator()(T a, T b) const { return a * b; }
};

template <typename T>
class CumprodOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<framework::Tensor>("X");
    auto *y = ctx.Output<framework::Tensor>("Out");
    auto dim = ctx.Attr<int>("dim");
    size_t outer_dim, mid_dim, inner_dim;
    GetCumprodDimInfo(x->dims(), dim, &outer_dim, &mid_dim, &inner_dim);

    const auto *x_data = x->data<T>();
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());
    const auto &dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    math::InclusiveScan<T, MultiplyFunctor<T>>(
        x_data, y_data, outer_dim, mid_dim, inner_dim, static_cast<T>(1),
        MultiplyFunctor<T>(), /*reverse=*/false, dev_ctx);
  }
};

template <typename T>
struct IsZeroFunctor {
  HOSTDEVICE bool operator()(T x) const { return x == static_cast<T>(0); }
};

template <typename T>
struct CumprodGradFunctorExceptFirstZero {
  HOSTDEVICE CumprodGradFunctorExceptFirstZero(
      const T *x, const T *y, const T *dy_mul_y_reversed_cumsum,
      const uint8_t *zero_mask, size_t mid_dim, size_t inner_dim, T *dx,
      int64_t *first_zero_idx, T *x_filled_one)
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
                                              size_t mid_dim, size_t inner_dim,
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

/*
Reference to
https://github.com/pytorch/pytorch/blob/master/aten/src/ATen/native/ReduceOps.cpp
input: x, y, dL/dy
output: dL/dx
dL/dx[i] = sum{0<=j<n} (dL/dy[j])*(dy[j]/dx[i]) (1)
         = sum(0<=j<n} (dL/dy[j])*(d(x[0]*x[1]*...*x[j])/dx[i])
if x[i] != 0, dL/dx[i] = sum{i<=j<n} (dL/dy[j])*(y[j]/x[i]) (2)
if x[i] == 0, the formula(2) can not be applied directly.
Suppose k is the first index of zero element, the formula will be:
i > k, dL/dx[i] = 0;
i < k, dL/dx[i] = 1/x[i]*sum{i<=j<n} (dL/dy[j]*y[j])
i = k, dL/dx[i] = y[i-1]*sum{i<=j<n} (dL/dy[j])*(x[i+1]*...*x[j])

First, we will show the main resolution.
We need to judge the relationship between i (current index) and k (index
which corresponds to the first element of 0).
To mark the relationship, we now introduce zero_mask and we also need to
mark the index of the first zero element.
zero_mask = cummax(x[i] == 0);      //label whether x[i]==0 until the index.
zero_index = -1;                    //store the first zero element's index.
e.g. x = [1, 4, 5, 0, 2, 3, 0];
     zero_mask = [0, 0, 0, 1, 1, 1, 1];
     zero_index = 3;
When i < k, we need to calculate the result of sum{i<=j<n}(d_y[j]*y[j]), we can
use reversed cumsum to calculate it.
R = reversed_cumsum(dy[j]*y[j]);     //store the calculation result of the
sum{i<=j<n}(d_y[j]*y[j]) and x[k+1],x[k+2],...,x[j] along the index k+1 ~ j.
When i = k, we need to calculate the result of prod{i<w<j}(x[w]).
To calculate it, we introduce x_filled_one, which fill 1 before x[k+1] along
the index 0 ~ k.
e.g. x = [1, 4, 5, 0, 2, 3, 0];
     x_filled_one = [1, 1, 1, 1, 2, 3, 0];
Thus, we can use cumprod(x_filled_one[j]) to calculate the result of
prod{k<=w<j}(x[w]).

Then, we will show more detailed implementation.
for (int i = 0; i < numel; i++) {
    if (zero_mask[i] == 0) {       //case i < k
        dx[i] = R[i] / x[i];
        x_filled_one[i] = 1;
    } else {
        if (i == 0) {              //case i = k
            dx[i] = 1;
            zero_index = i;
            x_filled_one[i] = 1;
        } else {
            if (zero_mask[i-1] == 0) {    //case i = k
                dx[i] = y[i-1];
                zero_index = i;
                x_filled_one[i] = 1;
            } else {                  //case i > k
                dx[i] = 0;
                x_filled_one[i] = x[i];
            }
        }
    }
}
T = reversed_cumsum(dy[j]*cumprod(x_filled_one[j]));
if (zero_index != -1) {
    dx[zero_index] *= T[zero_index];
}
*/

template <typename T>
class CumprodGradOpCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    const auto *x = ctx.Input<framework::Tensor>("X");
    const auto *y = ctx.Input<framework::Tensor>("Out");
    const auto *dy =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto *dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto dim = ctx.Attr<int>("dim");

    size_t outer_dim, mid_dim, inner_dim;
    GetCumprodDimInfo(x->dims(), dim, &outer_dim, &mid_dim, &inner_dim);
    if (outer_dim == 0 || mid_dim == 0 || inner_dim == 0) return;

    size_t numel = outer_dim * mid_dim * inner_dim;

    const auto *x_data = x->data<T>();
    const auto *y_data = y->data<T>();
    const auto *dy_data = dy->data<T>();

    auto place = ctx.GetPlace();
    const auto &dev_ctx =
        ctx.template device_context<platform::CUDADeviceContext>();
    auto *dx_data = dx->mutable_data<T>(place);

    // deal with complex
    const T *x_data_deal;
    const T *y_data_deal;
    memory::AllocationPtr x_conj;
    memory::AllocationPtr y_conj;
    if (framework::IsComplex<T>::value) {
      x_conj = memory::Alloc(place, numel * sizeof(T));
      auto *x_data_conj = reinterpret_cast<T *>(x_conj->ptr());
      y_conj = memory::Alloc(place, numel * sizeof(T));
      auto *y_data_conj = reinterpret_cast<T *>(y_conj->ptr());

      platform::ForRange<platform::CUDADeviceContext> for_range_x(dev_ctx,
                                                                  numel);
      math::ConjFunctor<T> functor_x(x_data, numel, x_data_conj);
      for_range_x(functor_x);

      platform::ForRange<platform::CUDADeviceContext> for_range_y(dev_ctx,
                                                                  numel);
      math::ConjFunctor<T> functor_y(y_data, numel, y_data_conj);
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
        memory::Alloc(place, numel * sizeof(uint8_t));
    auto *zero_mask_without_cummax_data =
        reinterpret_cast<uint8_t *>(zero_mask_without_cummax->ptr());
    thrust::transform(
        exec_policy, thrust::device_pointer_cast(x_data_deal),
        thrust::device_pointer_cast(x_data_deal) + numel,
        thrust::device_pointer_cast(zero_mask_without_cummax_data),
        IsZeroFunctor<T>());

    auto zero_mask = memory::Alloc(place, numel * sizeof(uint8_t));
    auto *zero_mask_data = reinterpret_cast<uint8_t *>(zero_mask->ptr());
    math::InclusiveScan<uint8_t, cub::Max>(
        zero_mask_without_cummax_data, zero_mask_data, outer_dim, mid_dim,
        inner_dim, static_cast<uint8_t>(0), cub::Max(), /*reverse=*/false,
        dev_ctx);
    zero_mask_without_cummax = nullptr;

    // Step 2: calculate reversed cumsum(dy * y)
    auto dy_mul_y = memory::Alloc(place, numel * sizeof(T));
    auto *dy_mul_y_data = reinterpret_cast<T *>(dy_mul_y->ptr());
    thrust::transform(exec_policy, thrust::device_pointer_cast(dy_data),
                      thrust::device_pointer_cast(dy_data) + numel,
                      thrust::device_pointer_cast(y_data_deal),
                      thrust::device_pointer_cast(dy_mul_y_data),
                      MultiplyFunctor<T>());

    auto dy_mul_y_reversed_cumsum = memory::Alloc(place, numel * sizeof(T));
    auto *dy_mul_y_reversed_cumsum_data =
        reinterpret_cast<T *>(dy_mul_y_reversed_cumsum->ptr());
    math::InclusiveScan<T, cub::Sum>(
        dy_mul_y_data, dy_mul_y_reversed_cumsum_data, outer_dim, mid_dim,
        inner_dim, static_cast<T>(0), cub::Sum(), /*reverse=*/true, dev_ctx);

    // Step 3: calculate the gradient value except the first zero position.
    // The gradient value of the first zero position is filled with out[idx-1],
    // while the gradient value of the other positions are calculated out
    // completely. This functor also:
    //  (1) find the first zero index, i.e., first_zero_idx_data.
    //  (2) fill x_filled_one, which satifies
    //      x_filled_one[i] = x[i], i > pos
    //      x_filled_one[i] = 1, i <= pos
    auto first_zero_idx =
        memory::Alloc(place, outer_dim * inner_dim * sizeof(int64_t));
    auto *first_zero_idx_data =
        reinterpret_cast<int64_t *>(first_zero_idx->ptr());
    auto *x_filled_one_data = dy_mul_y_data;  // reuse former allocated memory
    platform::ForRange<platform::CUDADeviceContext> for_range(dev_ctx, numel);
    CumprodGradFunctorExceptFirstZero<T> functor_except_first_zero(
        x_data_deal, y_data_deal, dy_mul_y_reversed_cumsum_data, zero_mask_data,
        mid_dim, inner_dim, dx_data, first_zero_idx_data, x_filled_one_data);
    for_range(functor_except_first_zero);

    // Step 4: calculate cumprod of x_filled_one
    auto *x_filled_one_cumprod_data =
        dy_mul_y_reversed_cumsum_data;  // reuse former allocated memory
    math::InclusiveScan<T, MultiplyFunctor<T>>(
        x_filled_one_data, x_filled_one_cumprod_data, outer_dim, mid_dim,
        inner_dim, static_cast<T>(1), MultiplyFunctor<T>(), /*reverse=*/false,
        dev_ctx);

    // Step 5: calculate reversed cumsum(dy * x_filled_one_cumprod)
    auto *dy_mul_x_filled_one_cumprod =
        dy_mul_y_data;  // reuse former allocated memory
    thrust::transform(exec_policy, thrust::device_pointer_cast(dy_data),
                      thrust::device_pointer_cast(dy_data) + numel,
                      thrust::device_pointer_cast(x_filled_one_cumprod_data),
                      thrust::device_pointer_cast(dy_mul_x_filled_one_cumprod),
                      MultiplyFunctor<T>());
    auto *dy_mul_x_filled_one_cumprod_reversed_cumsum =
        dy_mul_y_reversed_cumsum_data;  // reuse former allocated memory
    math::InclusiveScan<T, cub::Sum>(
        dy_mul_x_filled_one_cumprod,
        dy_mul_x_filled_one_cumprod_reversed_cumsum, outer_dim, mid_dim,
        inner_dim, static_cast<T>(0), cub::Sum(),
        /*reverse=*/true, dev_ctx);

    // Step 6: fill zero pos gradient value
    platform::ForRange<platform::CUDADeviceContext>
        for_range_fill_zero_pos_grad(dev_ctx, outer_dim * inner_dim);
    FillFirstZeroPositionGradFunctor<T> fill_first_zero_pos_grad_functor(
        first_zero_idx_data, dy_mul_x_filled_one_cumprod_reversed_cumsum,
        mid_dim, inner_dim, dx_data);
    for_range_fill_zero_pos_grad(fill_first_zero_pos_grad_functor);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OP_CUDA_KERNEL(
    cumprod, ops::CumprodOpCUDAKernel<float>, ops::CumprodOpCUDAKernel<double>,
    ops::CumprodOpCUDAKernel<int>, ops::CumprodOpCUDAKernel<int64_t>,
    ops::CumprodOpCUDAKernel<paddle::platform::complex<float>>,
    ops::CumprodOpCUDAKernel<paddle::platform::complex<double>>);

REGISTER_OP_CUDA_KERNEL(
    cumprod_grad, ops::CumprodGradOpCUDAKernel<float>,
    ops::CumprodGradOpCUDAKernel<double>, ops::CumprodGradOpCUDAKernel<int>,
    ops::CumprodGradOpCUDAKernel<int64_t>,
    ops::CumprodGradOpCUDAKernel<paddle::platform::complex<float>>,
    ops::CumprodGradOpCUDAKernel<paddle::platform::complex<double>>);
