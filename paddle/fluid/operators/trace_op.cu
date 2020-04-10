// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/operators/trace_op.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

template <typename T>
struct TraceFunctor {
  TraceFunctor(const T* input, int64_t* output_stride, int64_t* ret_strides,
               int64_t diag_size, int64_t pos, int64_t stride1, int64_t stride2,
               int64_t dim_size, bool is_grad, T* output)
      : input_(input),
        output_stride_(output_stride),
        ret_strides_(ret_strides),
        diag_size_(diag_size),
        pos_(pos),
        stride1_(stride1),
        stride2_(stride2),
        dim_size_(dim_size),
        is_grad_(is_grad),
        output_(output) {}

  HOSTDEVICE void operator()(size_t idx) const {
    int64_t position = pos_;
    int64_t num = idx;
    for (size_t i = 0; i < dim_size_; i++) {
      position += num / output_stride_[i] * ret_strides_[i];
      num = num % output_stride_[i];
    }
    for (int j = 0; j < diag_size_; j++) {
      if (is_grad_) {
        output_[position] = input_[idx];
      } else {
        output_[idx] += input_[position];
      }
      position += stride1_ + stride2_;
    }
  }

  const T* input_;
  int64_t* output_stride_;
  int64_t* ret_strides_;
  int64_t diag_size_;
  int64_t pos_;
  int64_t stride1_;
  int64_t stride2_;
  int64_t dim_size_;
  bool is_grad_;
  T* output_;
};

template <typename DeviceContext, typename T>
void TraceGrad(framework::DDim input_dims, framework::DDim input_stride,
               framework::DDim output_stride, const DeviceContext& dev_ctx,
               int64_t numel, const T* input_data, T* out_data, int64_t offset,
               int64_t dim1, int64_t dim2, bool is_grad) {
  auto dim1_ = dim1 < 0 ? input_dims.size() + dim1 : dim1;
  auto dim2_ = dim2 < 0 ? input_dims.size() + dim2 : dim2;
  auto len1 = input_dims[std::min(dim1_, dim2_)];
  auto len2 = input_dims[std::max(dim1_, dim2_)];
  auto stride1 = input_stride[std::min(dim1_, dim2_)];
  auto stride2 = input_stride[std::max(dim1_, dim2_)];

  int offset_stride = 0;
  if (offset >= 0) {
    offset_stride = stride2;
    len2 -= offset;
  } else {
    offset_stride = stride1;
    len1 += offset;
  }
  int diag_size = len2 < len1 ? len2 : len1;

  auto ret_strides = vectorize(input_stride);
  ret_strides.erase(ret_strides.begin() + std::max(dim1_, dim2_));
  ret_strides.erase(ret_strides.begin() + std::min(dim1_, dim2_));

  if (diag_size > 0) {
    int64_t pos = std::abs(offset) * offset_stride;
    thrust::device_vector<int64_t> strides_vec(vectorize(output_stride));
    int64_t* strides_arr = thrust::raw_pointer_cast(strides_vec.data());
    thrust::device_vector<int64_t> ret_vec(ret_strides);
    int64_t* ret_arr = thrust::raw_pointer_cast(ret_vec.data());
    int64_t dim_size = ret_strides.size();

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    TraceFunctor<T> functor(input_data, strides_arr, ret_arr, diag_size, pos,
                            stride1, stride2, dim_size, is_grad, out_data);
    for_range(functor);
  }
}

template <typename DeviceContext, typename T>
class TraceCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* input = context.Input<framework::Tensor>("Input");
    auto* out = context.Output<framework::Tensor>("Out");

    const int64_t offset = context.Attr<int>("offset");
    const int64_t dim1 = context.Attr<int>("dim1");
    const int64_t dim2 = context.Attr<int>("dim2");

    auto* input_data = input->data<T>();
    auto input_dims = input->dims();
    auto input_stride = framework::stride(input_dims);
    auto output_dims = out->dims();
    auto output_stride = framework::stride(output_dims);
    auto numel = out->numel();

    T* out_data = out->mutable_data<T>(context.GetPlace());

    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.cuda_device_context();
    set_zero(dev_ctx, out, static_cast<T>(0.0));

    TraceGrad<DeviceContext, T>(input_dims, input_stride, output_stride,
                                dev_ctx, numel, input_data, out_data, offset,
                                dim1, dim2, false);
  }
};

template <typename DeviceContext, typename T>
class TraceGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    const auto* d_out =
        context.Input<framework::Tensor>(framework::GradVarName("Out"));
    auto* d_x =
        context.Output<framework::Tensor>(framework::GradVarName("Input"));

    int64_t offset = context.Attr<int>("offset");
    int64_t dim1 = context.Attr<int>("dim1");
    int64_t dim2 = context.Attr<int>("dim2");

    auto input_dims = d_x->dims();
    auto input_stride = framework::stride(input_dims);
    auto output_dims = d_out->dims();
    auto output_stride = framework::stride(output_dims);
    auto numel = d_out->numel();

    auto* input_data = d_out->data<T>();
    T* out_data = d_x->mutable_data<T>(context.GetPlace());
    math::SetConstant<DeviceContext, T> set_zero;
    auto& dev_ctx = context.template device_context<DeviceContext>();
    set_zero(dev_ctx, d_x, static_cast<T>(0.0));

    TraceGrad<DeviceContext, T>(input_dims, input_stride, output_stride,
                                dev_ctx, numel, input_data, out_data, offset,
                                dim1, dim2, true);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace platform = paddle::platform;
REGISTER_OP_CUDA_KERNEL(
    trace, ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext,
                         platform::float16>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TraceCUDAKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    trace_grad,
    ops::TraceGradCUDAKernel<paddle::platform::CUDADeviceContext, int>,
    ops::TraceGradCUDAKernel<paddle::platform::CUDADeviceContext, int64_t>,
    ops::TraceGradCUDAKernel<paddle::platform::CUDADeviceContext,
                             platform::float16>,
    ops::TraceGradCUDAKernel<paddle::platform::CUDADeviceContext, float>,
    ops::TraceGradCUDAKernel<paddle::platform::CUDADeviceContext, double>);
