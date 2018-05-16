/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#define EIGEN_USE_GPU
#include <vector>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/math_function_impl.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {
namespace math {

using float16 = paddle::platform::float16;

template struct SetConstant<platform::CUDADeviceContext, platform::float16>;
template struct SetConstant<platform::CUDADeviceContext, float>;
template struct SetConstant<platform::CUDADeviceContext, double>;
template struct SetConstant<platform::CUDADeviceContext, int>;
template struct SetConstant<platform::CUDADeviceContext, int64_t>;
template struct SetConstant<platform::CUDADeviceContext, bool>;

#define DEFINE_GPU_TRANS(RANK)                                          \
  template struct Transpose<platform::CUDADeviceContext, float, RANK>;  \
  template struct Transpose<platform::CUDADeviceContext, double, RANK>; \
  template struct Transpose<platform::CUDADeviceContext, float16, RANK>;

DEFINE_GPU_TRANS(1);
DEFINE_GPU_TRANS(2);
DEFINE_GPU_TRANS(3);
DEFINE_GPU_TRANS(4);
DEFINE_GPU_TRANS(5);
DEFINE_GPU_TRANS(6);

struct TensorSetConstantGPU {
  TensorSetConstantGPU(const platform::DeviceContext& context,
                       framework::Tensor* tensor, float value)
      : context_(context), tensor_(tensor), value_(value) {}

  template <typename T>
  void operator()() const {
    SetConstant<platform::CUDADeviceContext, T> functor;
    functor(reinterpret_cast<const platform::CUDADeviceContext&>(context_),
            tensor_, static_cast<T>(value_));
  }

  const platform::DeviceContext& context_;
  framework::Tensor* tensor_;
  float value_;
};

template <>
void set_constant_with_place<platform::CUDAPlace>(
    const platform::DeviceContext& context, framework::Tensor* tensor,
    float value) {
  framework::VisitDataType(framework::ToDataType(tensor->type()),
                           TensorSetConstantGPU(context, tensor, value));
}

template <typename T>
__global__ void RowwiseAddKernel(const T* a, const T* b, T* c, int width,
                                 int num) {
  T tmp = 1.0 / width;
  for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < num;
       i += blockDim.x * gridDim.x) {
    int h = i * tmp;
    int w = i - h * width;
    c[i] = a[i] + b[w];
  }
}

template <typename T>
struct RowwiseAdd<platform::CUDADeviceContext, T> {
  void operator()(const platform::CUDADeviceContext& context,
                  const framework::Tensor& input,
                  const framework::Tensor& vector, framework::Tensor* output) {
    auto in_dims = input.dims();
    auto size = input.numel() / in_dims[0];
    PADDLE_ENFORCE_EQ(vector.numel(), size);
    PADDLE_ENFORCE_EQ(output->dims(), in_dims);
    int blocks = 512;
    int grids = (input.numel() + blocks - 1) / blocks;
    RowwiseAddKernel<T><<<grids, blocks, 0, context.stream()>>>(
        input.data<T>(), vector.data<T>(), output->data<T>(),
        static_cast<int>(in_dims[1]), static_cast<int>(input.numel()));
  }
};

template struct RowwiseAdd<platform::CUDADeviceContext, float>;
template struct RowwiseAdd<platform::CUDADeviceContext, double>;
template struct ColwiseSum<platform::CUDADeviceContext, float>;
template struct ColwiseSum<platform::CUDADeviceContext, int>;
template struct ColwiseSum<platform::CUDADeviceContext, int64_t>;
// template struct ColwiseSum<platform::CUDADeviceContext, double>;
// The ColwiseSum<platform::CUDADeviceContext, double> failed in debug mode,
// and only failed for this case. So reimplemented it.
template <>
void ColwiseSum<platform::CUDADeviceContext, double>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor& input,
    framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), size);
  framework::Tensor one;
  one.mutable_data<double>({in_dims[0]}, context.GetPlace());
  SetConstant<platform::CUDADeviceContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  GetBlas<platform::CUDADeviceContext, double>(context).GEMV(
      true, static_cast<int>(in_dims[0]), static_cast<int>(in_dims[1]), 1.0,
      input.data<double>(), one.data<double>(), 0.0, vector->data<double>());
}

template struct RowwiseSum<platform::CUDADeviceContext, float>;
// template struct RowwiseSum<platform::CUDADeviceContext, double>;
// TODO(zcd): Following ColwiseSum format, need to confirm.
// The RowwiseSum<platform::CUDADeviceContext, double> failed in debug mode,
// and only failed for this case. So reimplemented it.
template <>
void RowwiseSum<platform::CUDADeviceContext, double>::operator()(
    const platform::CUDADeviceContext& context, const framework::Tensor& input,
    framework::Tensor* vector) {
  auto in_dims = input.dims();
  auto size = input.numel() / in_dims[0];
  PADDLE_ENFORCE_EQ(vector->numel(), in_dims[0]);
  framework::Tensor one;
  one.mutable_data<double>({size}, context.GetPlace());
  SetConstant<platform::CUDADeviceContext, double> set;
  set(context, &one, static_cast<double>(1.0));
  GetBlas<platform::CUDADeviceContext, double>(context).GEMV(
      true, static_cast<int>(in_dims[1]), static_cast<int>(in_dims[0]), 1.0,
      one.data<double>(), input.data<double>(), 0.0, vector->data<double>());
}

template struct RowwiseMean<platform::CUDADeviceContext, float>;
template struct RowwiseMean<platform::CUDADeviceContext, double>;

}  // namespace math
}  // namespace operators
}  // namespace paddle
