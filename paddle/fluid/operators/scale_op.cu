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

#include "paddle/fluid/operators/elementwise/elementwise_op_impl.cu.h"
#include "paddle/fluid/operators/scale_op.h"
#include "paddle/fluid/platform/float16.h"

namespace plat = paddle::platform;

namespace paddle {
namespace operators {

template <typename T>
struct CudaScaleFunctor {
  T one = static_cast<T>(1.0f);
  float scale;
  float bias;
  bool bias_after_scale;

  CudaScaleFunctor(float s, float b, bool bas)
      : scale(s), bias(b), bias_after_scale(bas) {}

  // scale(x) = scale * x + bias, if bias_after_scale is True
  //          = scale * (x + bias), if bias_after_scale is False
  // Inputs: args[0], the input x
  __device__ __forceinline__ T operator()(const T* args) const {
    T s = static_cast<T>(scale);
    T b = static_cast<T>(bias);
    T bas = static_cast<T>(!bias_after_scale);
    return args[0] * s + b + (s - one) * b * bas;
  }
};

template <typename T>
class ScaleKernel<platform::CUDADeviceContext, T>
    : public framework::OpKernel<T> {
 public:
  virtual void Compute(const framework::ExecutionContext& ctx) const {
    auto* in_var = ctx.InputVar("X");
    auto* in = framework::GetLoDTensorOrSelectedRowsValueFromVar(*in_var);

    auto bias = static_cast<T>(ctx.Attr<float>("bias"));
    auto bias_after_scale = ctx.Attr<bool>("bias_after_scale");

    auto scale = static_cast<T>(ctx.Attr<float>("scale"));
    if (ctx.HasInput("ScaleTensor")) {
      auto* scale_tensor = ctx.Input<framework::Tensor>("ScaleTensor");
      scale = GetAttrFromTensor<T>(scale_tensor);
    }

    auto* out_var = ctx.OutputVar("Out");
    if (in_var->IsType<framework::SelectedRows>() && in_var != out_var) {
      auto& in_slr = in_var->Get<framework::SelectedRows>();
      auto* out_slr = out_var->GetMutable<framework::SelectedRows>();
      out_slr->set_rows(in_slr.rows());
      out_slr->set_height(in_slr.height());
    }

    auto* out =
        framework::GetMutableLoDTensorOrSelectedRowsValueFromVar(out_var);
    out->mutable_data<T>(in->place());

    PADDLE_ENFORCE_EQ(in->dims(), out->dims(),
                      plat::errors::InvalidArgument(
                          "the input and output should have the same dim"
                          "but input dim is %s, output dim is %s",
                          in->dims(), out->dims()));

    auto& dev = ctx.template device_context<platform::CUDADeviceContext>();

    std::vector<const framework::Tensor*> ins = {in};
    std::vector<framework::Tensor*> outs = {out};
    auto functor =
        CudaScaleFunctor<T>(static_cast<float>(scale), static_cast<float>(bias),
                            static_cast<bool>(bias_after_scale));
    LaunchElementwiseCudaKernel<ElementwiseType::kUnary, T>(dev, ins, &outs,
                                                            functor);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OP_CUDA_KERNEL(
    scale, paddle::operators::ScaleKernel<plat::CUDADeviceContext, float>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, double>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, uint8_t>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, int8_t>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, int16_t>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, int>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, int64_t>,
    paddle::operators::ScaleKernel<plat::CUDADeviceContext, plat::float16>);
