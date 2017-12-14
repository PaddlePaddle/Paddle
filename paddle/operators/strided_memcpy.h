/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#pragma once
#include "paddle/operators/detail/strided_memcpy.h"

namespace paddle {
namespace operators {

// Strided memory copy from src to dst.
//
// The src and dst should be both on dev_ctx.GetPlace(), otherwise, there will
// be a segment fault.
//
// The stride of an array (also referred to as increment, pitch or step size) is
// the number of locations in memory between beginnings of successive array
// elements
//
// For example, for tensor like [1, 3, 300, 300]. If there is no padding, the
// stride is [270000, 90000, 300, 1].
//
// NOTE: When use GPU, the memcpy is async. To sync memcpy, please invoke
// `dev_ctx.Wait()`.
template <typename T>
inline void StridedMemcpy(const platform::DeviceContext& dev_ctx, const T* src,
                          const framework::DDim& src_stride,
                          const framework::DDim& dst_dim,
                          const framework::DDim& dst_stride, T* dst) {
  using namespace detail;
  StridedCopyDimVisitor<T> func(dev_ctx, src, src_stride, dst_stride, dst);
  boost::apply_visitor(func, dst_dim);
}
}  // namespace operators
}  // namespace paddle
