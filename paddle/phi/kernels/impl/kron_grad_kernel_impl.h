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

#pragma once

#include "paddle/phi/kernels/impl/kron_kernel_impl.h"

namespace phi {

template <typename T>
struct KronGradElemFunctor {
  KronGradElemFunctor(const T *dout,
                      const T *A,
                      const T *B,
                      T *dout_a,
                      T *dout_b,
                      const int64_t *stride_dout,
                      const int64_t *stride_a,
                      const int64_t *stride_b,
                      const int64_t *shape_b,
                      const int64_t numel_a,
                      const int64_t numel_b,
                      const int ndims)
      : dout_(dout),
        A_(A),
        B_(B),
        dout_a_(dout_a),
        dout_b_(dout_b),
        stride_dout_(stride_dout),
        stride_a_(stride_a),
        stride_b_(stride_b),
        shape_b_(shape_b),
        numel_a_(numel_a),
        numel_b_(numel_b),
        ndims_(ndims) {}

  HOSTDEVICE void operator()(int64_t idx) {
    int64_t index = idx;
    int64_t index_a = 0;
    int64_t index_b = 0;
    for (int i = 0; i < ndims_; i++) {
      auto pos_i = index / stride_dout_[i];
      index = index % stride_dout_[i];
      auto pos_ai = pos_i / shape_b_[i];
      auto pos_bi = pos_i % shape_b_[i];
      index_a += stride_a_[i] * pos_ai;
      index_b += stride_b_[i] * pos_bi;
    }

    if (dout_a_) {
      size_t index_out_a = index_a * numel_b_ + index_b;
      dout_a_[index_out_a] = dout_[idx] * B_[index_b];
    }
    if (dout_b_) {
      size_t index_out_b = index_b * numel_a_ + index_a;
      dout_b_[index_out_b] = dout_[idx] * A_[index_a];
    }
  }

 private:
  const T *dout_;
  const T *A_;
  const T *B_;
  T *dout_a_;
  T *dout_b_;
  const int64_t *stride_dout_;
  const int64_t *stride_a_;
  const int64_t *stride_b_;
  const int64_t *shape_b_;
  const int64_t numel_a_;
  const int64_t numel_b_;
  const int ndims_;
};

template <typename T>
struct KronGradElemFunctor<dtype::complex<T>> {
  KronGradElemFunctor(const dtype::complex<T> *dout,
                      const dtype::complex<T> *A,
                      const dtype::complex<T> *B,
                      dtype::complex<T> *dout_a,
                      dtype::complex<T> *dout_b,
                      const int64_t *stride_dout,
                      const int64_t *stride_a,
                      const int64_t *stride_b,
                      const int64_t *shape_b,
                      const int64_t numel_a,
                      const int64_t numel_b,
                      const int ndims)
      : dout_(dout),
        A_(A),
        B_(B),
        dout_a_(dout_a),
        dout_b_(dout_b),
        stride_dout_(stride_dout),
        stride_a_(stride_a),
        stride_b_(stride_b),
        shape_b_(shape_b),
        numel_a_(numel_a),
        numel_b_(numel_b),
        ndims_(ndims) {}

  HOSTDEVICE void operator()(int64_t idx) {
    int64_t index = idx;
    int64_t index_a = 0;
    int64_t index_b = 0;
    for (int i = 0; i < ndims_; i++) {
      auto pos_i = index / stride_dout_[i];
      index = index % stride_dout_[i];
      auto pos_ai = pos_i / shape_b_[i];
      auto pos_bi = pos_i % shape_b_[i];
      index_a += stride_a_[i] * pos_ai;
      index_b += stride_b_[i] * pos_bi;
    }

    if (dout_a_) {
      size_t index_out_a = index_a * numel_b_ + index_b;
      dout_a_[index_out_a] =
          dout_[idx] * dtype::complex<T>(B_[index_b].real, -B_[index_b].imag);
    }
    if (dout_b_) {
      size_t index_out_b = index_b * numel_a_ + index_a;
      dout_b_[index_out_b] =
          dout_[idx] * dtype::complex<T>(A_[index_a].real, -A_[index_a].imag);
    }
  }

 private:
  const dtype::complex<T> *dout_;
  const dtype::complex<T> *A_;
  const dtype::complex<T> *B_;
  dtype::complex<T> *dout_a_;
  dtype::complex<T> *dout_b_;
  const int64_t *stride_dout_;
  const int64_t *stride_a_;
  const int64_t *stride_b_;
  const int64_t *shape_b_;
  const int64_t numel_a_;
  const int64_t numel_b_;
  const int ndims_;
};

template <typename Context, typename T>
struct KronGradOpFunctor {
  void operator()(const Context &dev_ctx,
                  const DenseTensor &dout,
                  const DenseTensor &x,
                  const DenseTensor &y,
                  DenseTensor *dx,
                  DenseTensor *dy) {
    int ndims = dout.dims().size();
    int64_t numel = dout.numel();
    int64_t numel_x = x.numel();
    int64_t numel_y = y.numel();

    const phi::DDim &dim_x = x.dims();
    const phi::DDim &dim_y = y.dims();
    const phi::DDim &dim_dout = dout.dims();
    const phi::DDim stride_x =
        dim_x.size() == 0 ? phi::DDim(dim_x) : phi::stride(dim_x);
    const phi::DDim stride_y =
        dim_y.size() == 0 ? phi::DDim(dim_y) : phi::stride(dim_y);
    const phi::DDim stride_dout =
        dim_dout.size() == 0 ? phi::DDim(dim_dout) : phi::stride(dim_dout);

    const int64_t *p_stride_x = nullptr;
    const int64_t *p_stride_y = nullptr;
    const int64_t *p_stride_dout = nullptr;
    const int64_t *p_shape_y = nullptr;
#if defined(__NVCC__) || defined(__HIPCC__)
    thrust::device_vector<int64_t> d_stride_x(ndims);
    thrust::device_vector<int64_t> d_stride_y(ndims);
    thrust::device_vector<int64_t> d_stride_dout(ndims);
    thrust::device_vector<int64_t> d_shape_y(ndims);
    thrust::copy(stride_x.Get(), stride_x.Get() + ndims, d_stride_x.begin());
    thrust::copy(stride_y.Get(), stride_y.Get() + ndims, d_stride_y.begin());
    thrust::copy(
        stride_dout.Get(), stride_dout.Get() + ndims, d_stride_dout.begin());
    thrust::copy(dim_y.Get(), dim_y.Get() + ndims, d_shape_y.begin());

    p_stride_x = thrust::raw_pointer_cast(d_stride_x.data());
    p_stride_y = thrust::raw_pointer_cast(d_stride_y.data());
    p_stride_dout = thrust::raw_pointer_cast(d_stride_dout.data());
    p_shape_y = thrust::raw_pointer_cast(d_shape_y.data());
#else
    p_stride_x = stride_x.Get();
    p_stride_y = stride_y.Get();
    p_stride_dout = stride_dout.Get();
    p_shape_y = dim_y.Get();
#endif
    // dout_x: dout * kron(ones(X), Y) re-aranged in shape (numel_x, numel_y)
    // dout_y: dout * kron(X, ones(Y)) re-aranged in shaoe (numel_y, numel_x)
    DenseTensor dout_x;
    T *p_dout_x = nullptr;
    if (dx) {
      dout_x.Resize({numel_x, numel_y});
      dev_ctx.template Alloc<T>(&dout_x);
      p_dout_x = dout_x.data<T>();
    }
    DenseTensor dout_y;
    T *p_dout_y = nullptr;
    if (dy) {
      dout_y.Resize({numel_y, numel_x});
      dev_ctx.template Alloc<T>(&dout_y);
      p_dout_y = dout_y.data<T>();
    }

    funcs::ForRange<Context> for_range(dev_ctx, numel);
    KronGradElemFunctor<T> func(dout.data<T>(),
                                x.data<T>(),
                                y.data<T>(),
                                p_dout_x,
                                p_dout_y,
                                p_stride_dout,
                                p_stride_x,
                                p_stride_y,
                                p_shape_y,
                                numel_x,
                                numel_y,
                                ndims);
    for_range(func);

// reduce_sum along aixs 1
#if defined(__NVCC__) || defined(__HIPCC__)
    auto stream = dev_ctx.stream();  // it is a cuda device_context
    if (dx) {
      funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx, dout_x, dx, kps::IdentityFunctor<T>(), {1});
    }
    if (dy) {
      funcs::ReduceKernel<T, T, kps::AddFunctor, kps::IdentityFunctor<T>>(
          dev_ctx, dout_y, dy, kps::IdentityFunctor<T>(), {1});
    }
#else
    auto *place = dev_ctx.eigen_device();
    Eigen::array<int, 1> reduce_dim = {1};
    if (dx) {
      auto eigen_dout_x = EigenMatrix<T>::Reshape(dout_x, 1);
      auto eigen_vec_dx = EigenVector<T>::Flatten(*dx);
      eigen_vec_dx.device(*place) = eigen_dout_x.sum(reduce_dim);
    }
    if (dy) {
      auto eigen_dout_y = EigenMatrix<T>::Reshape(dout_y, 1);
      auto eigen_vec_dy = EigenVector<T>::Flatten(*dy);
      eigen_vec_dy.device(*place) = eigen_dout_y.sum(reduce_dim);
    }
#endif
  }
};

template <typename T, typename Context>
void KronGradKernel(const Context &ctx,
                    const DenseTensor &x,
                    const DenseTensor &y,
                    const DenseTensor &out_grad,
                    DenseTensor *x_grad,
                    DenseTensor *y_grad) {
  if (x_grad) {
    ctx.template Alloc<T>(x_grad);
  }
  if (y_grad) {
    ctx.template Alloc<T>(y_grad);
  }

  int ndims = out_grad.dims().size();
  DenseTensor xx = UnsqueezeTo(x, ndims);
  DenseTensor yy = UnsqueezeTo(y, ndims);

  DenseTensor *pdxx = nullptr;
  DenseTensor *pdyy = nullptr;
  DenseTensor dxx;
  DenseTensor dyy;
  if (x_grad) {
    dxx = UnsqueezeTo(*x_grad, ndims);
    pdxx = &dxx;
  }

  if (y_grad) {
    dyy = UnsqueezeTo(*y_grad, ndims);
    pdyy = &dyy;
  }

  KronGradOpFunctor<Context, T> func;
  func(ctx, out_grad, xx, yy, pdxx, pdyy);
}

}  // namespace phi
