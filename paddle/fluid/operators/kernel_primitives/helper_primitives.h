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

#pragma once

namespace paddle {
namespace operators {
namespace kernel_primitives {
namespace details {

static __device__ __forceinline__ platform::float16 ExpFunctor(
    platform::float16 x) {
  return ::Eigen::numext::exp(x);
}
static __device__ __forceinline__ float ExpFunctor(float x) { return expf(x); }
static __device__ __forceinline__ double ExpFunctor(double x) { return exp(x); }
static __device__ __forceinline__ platform::float16 LogFunctor(
    platform::float16 x) {
  return ::Eigen::numext::log(x);
}
static __device__ __forceinline__ float LogFunctor(float x) { return logf(x); }
static __device__ __forceinline__ double LogFunctor(double x) { return log(x); }

/*************************** Compute Functor****************************/
// for margin_cross_entropy
template <typename Tx, typename Ty = Tx>
struct ExpLogitTransformer {
  HOSTDEVICE explicit inline ExpLogitTransformer(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx* x) const {
    return static_cast<Ty>(details::ExpFunctor(x[0]));
  }

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(details::ExpFunctor(x));
  }
};

// Post processing function for sum, max, min, prod, any
template <typename Tx, typename Ty = Tx>
struct IdentityFunctor {
  HOSTDEVICE explicit inline IdentityFunctor(int n) {}

  HOSTDEVICE inline Ty operator()(const Tx* x) const {
    return static_cast<Ty>(x[0]);
  }

  HOSTDEVICE inline Ty operator()(const Tx& x) const {
    return static_cast<Ty>(x);
  }
};

// Post processing function for mean
template <typename T>
struct DivideFunctor {
  HOSTDEVICE explicit inline DivideFunctor(int n) : n_inv((T)(1.0 / n)) {}

  HOSTDEVICE inline T operator()(const T* x) const { return x[0] * n_inv; }

  HOSTDEVICE inline T operator()(const T& x) const { return x * n_inv; }

 private:
  T n_inv;
};
}  // namespace details
struct DimConfig {
  int split_num_x;
  int split_num_y;
  int split_num_z;
  int deal_size_x;
  int deal_size_y;
  int deal_size_z;
  int rem_x;
  int rem_y;
  int rem_z;

  HOSTDEVICE explicit inline DimConfig(int split_x, int split_y, int split_z,
                                       int size_x, int size_y, int size_z) {
    split_num_x = split_x;
    split_num_y = split_y;
    split_num_z = split_z;
    deal_size_x = size_x;
    deal_size_y = size_y;
    deal_size_z = size_z;
  }

  HOSTDEVICE void setRem(int rem_nx, int rem_ny, int rem_nz) {
    rem_x = rem_nx;
    rem_y = rem_ny;
    rem_z = rem_nz;
  }
};
}  // namespace kernel_primitives
}  // namespace operators
}  // namespace paddle
