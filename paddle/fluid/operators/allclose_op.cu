// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/operators/allclose_op.h"

namespace paddle {
namespace operators {

template <typename T>
__global__ void AllcloseCUDAKernel(T* out_data, const T* in_a, const T* in_b,
                                   const float rtol, const float atol,
                                   bool equal_nan) {
  int tid = threadIdx.x;
  const T a = in_a[tid], b = in_b[tid];
  T tol, dif;
  double threshold = 1e-7;
  bool val;
  if (isnan(a) || isnan(b)) {
    val = equal_nan && isnan(a) == isnan(b);
  } else {
    dif = fabs(fabs(a - b) - (atol + rtol * fabs(b)));
    tol = (a > b ? a - b : b - a);
    T tol2 = atol + (b > 0 ? rtol * b : (-rtol) * b);
    printf("dif is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n", dif);
    printf("tol is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n", tol);
    printf("tol2 is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n",
           tol2);
    printf("rtol is>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %.15f\n",
           rtol);
    val = a == b ||
          (a > b ? a - b : b - a) < atol + (b > 0 ? rtol * b : (-rtol) * b) ||
          dif < threshold;
    printf("val is >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>: %d\n", val);
  }
  out_data[tid] = val;
}

template <typename T>
struct AllcloseFunctor<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& ctx,
                  const framework::Tensor& out, const framework::Tensor& in,
                  const framework::Tensor& other, const float rtol,
                  const float atol, bool equal_nan) {
    auto in_dims = in.dims().size();
    auto other_dims = other.dims().size();

    PADDLE_ENFORCE_EQ(in_dims == other_dims, true,
                      platform::errors::InvalidArgument(
                          "Dims of input(a) and dims of other(b) should"
                          "be equal, but received the dims of input is : %d ,"
                          "received the dims of other is :%d. ",
                          in_dims, other_dims));
    const T* in_data = in.data<T>();
    const T* other_data = other.data<T>();
    T* out_data = out.mutable_data<T>(ctx.GetPlace());

    int grid = in_dims;
    int block = 1;

    AllcloseCUDAKernel<T><<<grid, block, 0, ctx.stream()>>>(
        out_data, in_data, other_data, rtol, atol, equal_nan);
  }
};

// template struct AllcloseFunctor<platform::CUDADeviceContext, float>;
template struct AllcloseFunctor<platform::CUDADeviceContext, double>;

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(allclose, ops::AllcloseKernel<CUDA, float>,
                        ops::AllcloseKernel<CUDA, double>);
