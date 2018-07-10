/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"
#include "paddle/fluid/platform/dynload/cudnn.h"
#include "paddle/fluid/platform/float16.h"

namespace paddle {
namespace operators {

inline static void GetNCHW(const framework::DDim& dim,
                           std::array<int, 4>* nchw) {
  size_t sz = dim.size();
  PADDLE_ENFORCE(sz > 0, "Dims of Input cannot be 0");
  if (sz <= nchw->size()) {
    for (size_t i = 0; i < sz; ++i) (*nchw)[i] = dim[i];
    for (size_t i = sz; i < nchw->size(); ++i) (*nchw)[i] = 1;
  } else {
    for (size_t i = 0; i < nchw->size() - 1; ++i) (*nchw)[i] = dim[i];
    int prod = 1;
    for (size_t i = nchw->size() - 1; i < sz; ++i) prod *= dim[i];
    (*nchw)[nchw->size() - 1] = prod;
  }
}

template <typename T>
class CUDNNDropoutKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::cerr << "Computing..." << std::endl;
    auto* x = context.Input<Tensor>("X");
    auto* y = context.Output<Tensor>("Out");

    float dropout_prob = context.Attr<float>("dropout_prob");
    T alpha(1.0f - dropout_prob), beta(0.0f);

    auto* x_data = x->data<T>();
    auto* y_data = y->mutable_data<T>(context.GetPlace());

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    std::array<int, 4> nchw;
    GetNCHW(x->dims(), &nchw);

    cudnnTensorDescriptor_t x_desc, y_desc;
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&x_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&y_desc));

    CUDNN_ENFORCE(platform::dynload::cudnnSetTensor4dDescriptor(
        x_desc, format, platform::CudnnDataType<T>::type, nchw[0], nchw[1],
        nchw[2], nchw[3]));

    CUDNN_ENFORCE(platform::dynload::cudnnSetTensor4dDescriptor(
        y_desc, format, platform::CudnnDataType<T>::type, nchw[0], nchw[1],
        nchw[2], nchw[3]));

    if (!context.Attr<bool>("is_test")) {
      cudnnDropoutDescriptor_t dropoutDesc;
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateDropoutDescriptor(&dropoutDesc));

      std::random_device rnd;
      int seed =
          context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

      auto* states = context.Output<Tensor>("States");
      auto* reserve_space = context.Output<Tensor>("ReserveSpace");

      size_t states_size, reserve_space_size;
      CUDNN_ENFORCE(
          platform::dynload::cudnnDropoutGetStatesSize(handle, &states_size));
      CUDNN_ENFORCE(platform::dynload::cudnnDropoutGetReserveSpaceSize(
          x_desc, &reserve_space_size));

      states->Resize({static_cast<int64_t>(states_size)});
      reserve_space->Resize({static_cast<int64_t>(reserve_space_size)});

      auto* states_data = states->mutable_data<uint8_t>(context.GetPlace());
      auto* reserve_space_data =
          reserve_space->mutable_data<uint8_t>(context.GetPlace());

      cudnnDropoutDescriptor_t dropout_desc;
      CUDNN_ENFORCE(
          platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc));
      CUDNN_ENFORCE(platform::dynload::cudnnSetDropoutDescriptor(
          dropout_desc, handle, dropout_prob, states_data, states_size, seed));

      CUDNN_ENFORCE(platform::dynload::cudnnDropoutForward(
          handle, dropout_desc, x_desc, x_data, y_desc, y_data,
          reserve_space_data, reserve_space_size));

      CUDNN_ENFORCE(
          platform::dynload::cudnnScaleTensor(handle, y_desc, y_data, &alpha));

      CUDNN_ENFORCE(
          platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc));
    } else {
      if (x_data != y_data) {
        CUDNN_ENFORCE(platform::dynload::cudnnTransformTensor(
            handle, &alpha, x_desc, x_data, &beta, y_desc, y_data));
      } else {
        CUDNN_ENFORCE(platform::dynload::cudnnScaleTensor(handle, y_desc,
                                                          y_data, &alpha));
      }
    }
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(x_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(y_desc));
  }
};

template <typename T>
class CUDNNDropoutGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    std::cerr << "Computing grad..." << std::endl;

    PADDLE_ENFORCE(!context.Attr<bool>("is_test"),
                   "GradOp is only callable when is_test is false");

    auto* grad_x = context.Output<Tensor>(framework::GradVarName("X"));
    auto* grad_y = context.Input<Tensor>(framework::GradVarName("Out"));

    auto* grad_x_data = grad_x->mutable_data<T>(context.GetPlace());
    auto* grad_y_data = grad_y->data<T>();

    auto& dev_ctx =
        context.template device_context<platform::CUDADeviceContext>();
    auto handle = dev_ctx.cudnn_handle();

    cudnnTensorFormat_t format = CUDNN_TENSOR_NCHW;
    std::array<int, 4> nchw;
    GetNCHW(grad_y->dims(), &nchw);

    std::random_device rnd;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();

    float dropout_prob = context.Attr<float>("dropout_prob");

    auto* states = context.Input<Tensor>("States");
    auto* reserve_space = context.Input<Tensor>("ReserveSpace");

    cudnnTensorDescriptor_t grad_x_desc, grad_y_desc;
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&grad_x_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnCreateTensorDescriptor(&grad_y_desc));

    CUDNN_ENFORCE(platform::dynload::cudnnSetTensor4dDescriptor(
        grad_x_desc, format, platform::CudnnDataType<T>::type, nchw[0], nchw[1],
        nchw[2], nchw[3]));

    CUDNN_ENFORCE(platform::dynload::cudnnSetTensor4dDescriptor(
        grad_y_desc, format, platform::CudnnDataType<T>::type, nchw[0], nchw[1],
        nchw[2], nchw[3]));

    auto* states_data = const_cast<uint8_t*>(states->data<uint8_t>());
    auto* reserve_space_data =
        const_cast<uint8_t*>(reserve_space->data<uint8_t>());

    cudnnDropoutDescriptor_t dropout_desc;
    CUDNN_ENFORCE(
        platform::dynload::cudnnCreateDropoutDescriptor(&dropout_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnSetDropoutDescriptor(
        dropout_desc, handle, dropout_prob, states_data, states->numel(),
        seed));

    CUDNN_ENFORCE(platform::dynload::cudnnDropoutBackward(
        handle, dropout_desc, grad_y_desc, grad_y_data, grad_x_desc,
        grad_x_data, reserve_space_data, reserve_space->numel()));

    T alpha(1.0f - dropout_prob);
    CUDNN_ENFORCE(platform::dynload::cudnnScaleTensor(handle, grad_x_desc,
                                                      grad_x_data, &alpha));

    CUDNN_ENFORCE(
        platform::dynload::cudnnDestroyDropoutDescriptor(dropout_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(grad_x_desc));
    CUDNN_ENFORCE(platform::dynload::cudnnDestroyTensorDescriptor(grad_y_desc));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;

REGISTER_OP_KERNEL(dropout, CUDNN, plat::CUDAPlace,
                   ops::CUDNNDropoutKernel<float>,
                   ops::CUDNNDropoutKernel<plat::float16>);
REGISTER_OP_KERNEL(dropout_grad, CUDNN, plat::CUDAPlace,
                   ops::CUDNNDropoutGradKernel<float>,
                   ops::CUDNNDropoutGradKernel<plat::float16>);
