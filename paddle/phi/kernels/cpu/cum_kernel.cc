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

#include "paddle/phi/kernels/cum_kernel.h"

#include "paddle/phi/backends/cpu/cpu_context.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/eigen/common.h"
#include "paddle/phi/kernels/funcs/eigen/eigen_function.h"

namespace phi {

template <typename Device,
          typename Dim,
          typename X,
          typename Out,
          typename Reducer>
void ComputeImp(Device d,
                const Dim& dims,
                X x,
                Out out,
                int axis,
                bool reverse,
                bool exclusive,
                Reducer reducer) {
  if (!reverse) {
    out.reshape(dims).device(d) =
        x.reshape(dims).scan(axis, reducer, exclusive);
  } else {
    std::array<bool, Dim::count> rev;
    rev.fill(false);
    rev[axis] = reverse;
    out.reshape(dims).device(d) = x.reshape(dims)
                                      .reverse(rev)
                                      .scan(axis, reducer, exclusive)
                                      .reverse(rev);
  }
}

template <typename T, typename Context, typename Reducer>
void ScanKernel(const Context& dev_ctx,
                const DenseTensor& x,
                int axis,
                bool flatten UNUSED,
                bool exclusive,
                bool reverse,
                Reducer reducer,
                DenseTensor* out) {
  dev_ctx.template Alloc<T>(out);

  if (x.numel() == 1) {
    auto raw_dims = out->dims();
    phi::Copy<Context>(dev_ctx, x, dev_ctx.GetPlace(), false, out);
    out->Resize(raw_dims);
    return;
  }
  auto out_dims = out->dims();

  PADDLE_ENFORCE_EQ(
      axis < out_dims.size() && axis >= (0 - out_dims.size()),
      true,
      common::errors::OutOfRange(
          "Attr(axis) is out of range, It's expected "
          "to be in range of [-%d, %d]. But received Attr(axis) = %d.",
          out_dims.size(),
          out_dims.size() - 1,
          axis));
  if (axis < 0) {
    axis += out_dims.size();
  }

  int pre = 1;
  int post = 1;
  int mid = static_cast<int>(out_dims[axis]);
  for (int i = 0; i < axis; ++i) {
    pre *= static_cast<int>(out_dims[i]);
  }
  for (int i = axis + 1; i < out_dims.size(); ++i) {
    post *= static_cast<int>(out_dims[i]);
  }

  auto x0 = EigenVector<T>::Flatten(x);
  auto out0 = EigenVector<T>::Flatten(*out);
  auto& place = *dev_ctx.eigen_device();

  using IndexT = Eigen::DenseIndex;
  if (pre == 1) {
    if (post == 1) {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 1>(mid),
                 x0,
                 out0,
                 /* axis= */ 0,
                 reverse,
                 exclusive,
                 reducer);
    } else {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 2>(mid, post),
                 x0,
                 out0,
                 /* axis= */ 0,
                 reverse,
                 exclusive,
                 reducer);
    }
  } else {
    if (post == 1) {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 2>(pre, mid),
                 x0,
                 out0,
                 /* axis= */ 1,
                 reverse,
                 exclusive,
                 reducer);
    } else {
      ComputeImp(place,
                 Eigen::DSizes<IndexT, 3>(pre, mid, post),
                 x0,
                 out0,
                 /* axis= */ 1,
                 reverse,
                 exclusive,
                 reducer);
    }
  }
}

template <typename T, typename Context>
void CumsumKernel(const Context& dev_ctx,
                  const DenseTensor& x,
                  const Scalar& axis,
                  bool flatten,
                  bool exclusive,
                  bool reverse,
                  DenseTensor* out) {
  using Reducer = Eigen::internal::SumReducer<T>;
  auto reducer = Reducer();
  ScanKernel<T, Context, Reducer>(
      dev_ctx, x, axis.to<int>(), flatten, exclusive, reverse, reducer, out);
}

template <typename T>
struct LogSumExp {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    auto mi = Eigen::internal::scalar_min_op<T>()(a, b);
    auto ma = Eigen::internal::scalar_max_op<T>()(a, b);

    auto sub = Eigen::internal::scalar_difference_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto log1p = Eigen::internal::scalar_log1p_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();

    auto logsumexp = add(log1p(exp(sub(mi, ma))), ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? ma : logsumexp;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T packetOp(const T& a,
                                                   const T& b) const {
    auto mi = Eigen::internal::pmin(a, b);
    auto ma = Eigen::internal::pmax(a, b);
    using Eigen::internal::padd;
    using Eigen::internal::pcmp_lt;
    using Eigen::internal::pexp;
    using Eigen::internal::plog1p;
    using Eigen::internal::pset1;
    using Eigen::internal::psub;

    auto logsumexp = padd(plog1p(pexp(psub(mi, ma))), ma);
    return pselect(
        pcmp_lt(ma, pset1(Eigen::NumTraits<T>::lowest())), ma, logsumexp);
  }
};

template <typename T>
struct LogSumExpReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp(*accum, t);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp.packetOp(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return Eigen::NumTraits<T>::lowest();
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return Eigen::internal::pset1(initialize());
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
    auto max_reducer = Eigen::internal::MaxReducer<T, Eigen::PropagateNaN>();
    auto sum_reducer = Eigen::internal::SumReducer<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();
    auto log = Eigen::internal::scalar_log_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();

    using Eigen::internal::pexp;
    using Eigen::internal::psub;

    // `ma = max(x1, ..., xn)`
    // If the max of all of the `xi` is `-infinity` then the result is
    // -infinity. If the max is larger than `-infinity` then it's safe to use
    // for normalization even if the other elements are `-infinity`.
    //
    // `logsumexp(x1, ..., xn) = ma + log (exp(x1 - ma) + ... + exp(xn - ma))`
    auto ma = max_reducer.finalizeBoth(saccum, vaccum);
    auto logsumexp = add(log(sum_reducer.finalizeBoth(
                             exp(saccum - ma), pexp(psub(vaccum, pset1(ma))))),
                         ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? initialize() : logsumexp;
  }
};

template <typename T, typename Context>
void LogcumsumexpKernel(const Context& dev_ctx,
                        const DenseTensor& x,
                        int axis,
                        bool flatten,
                        bool exclusive,
                        bool reverse,
                        DenseTensor* out) {
  using Reducer = LogSumExpReducer<T>;
  auto reducer = Reducer();
  ScanKernel<T, Context, Reducer>(
      dev_ctx, x, axis, flatten, exclusive, reverse, reducer, out);
}

}  // namespace phi

PD_REGISTER_KERNEL(cumsum,
                   CPU,
                   ALL_LAYOUT,
                   phi::CumsumKernel,
                   float,
                   double,
                   int16_t,
                   int,
                   int64_t) {}

PD_REGISTER_KERNEL(
    logcumsumexp, CPU, ALL_LAYOUT, phi::LogcumsumexpKernel, float, double) {}
