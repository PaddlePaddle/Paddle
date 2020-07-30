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

#include <string>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/prelu.h"
#include "paddle/fluid/operators/prelu_op.h"
#include "paddle/fluid/operators/reduce_ops/cub_reduce.h"
#include "paddle/fluid/platform/cuda_primitives.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;

#define CUDA_NUM_THREADS 1024

inline static int PADDLE_GET_BLOCKS(const int N) {
  return (N + CUDA_NUM_THREADS - 1) / CUDA_NUM_THREADS;
}

template <typename DeviceContext, typename T>
class CUDAPReluKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* out = context.Output<Tensor>("Out");

    const T* x_ptr = x->data<T>();
    T* o_ptr = out->mutable_data<T>(context.GetPlace());

    const T* alpha_ptr = alpha->data<T>();
    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();

    VLOG(4) << "dim[0]:" << dim[0] << ", dim[1]:" << dim[1]
            << ", numel:" << numel;

    if (mode == "channel") {
      math::PreluChannelWiseDirectCUDAFunctor<T> prelu_channel_wise;
      prelu_channel_wise(context.cuda_device_context().stream(), x_ptr,
                         alpha_ptr, o_ptr, dim[0], dim[1], numel);
    } else if (mode == "element") {
      math::PreluElementWiseDirectCUDAFunctor<T> prelu_element_wise;
      prelu_element_wise(context.cuda_device_context().stream(), x_ptr,
                         alpha_ptr, o_ptr, dim[0], numel);
    } else {
      math::PreluScalarDirectCUDAFunctor<T> prelu_scalar;
      prelu_scalar(context.cuda_device_context().stream(), x_ptr, alpha_ptr,
                   o_ptr, numel);
    }
  }
};

enum PRELU_MODE { Element, Channel, Scalar };

template <typename T>
__global__ void PReluOpGradKernel(const T* x_ptr, const T* alpha_ptr,
                                  const T* dy_ptr, T* dx_ptr, T* dalpha_ptr,
                                  size_t channel_num, size_t plane_size,
                                  size_t spatial_size, size_t numel,
                                  PRELU_MODE mode) {
  CUDA_KERNEL_LOOP(index, numel) {
    T scale;
    if (mode == Element) {
      size_t element_index = index % spatial_size;
      scale = alpha_ptr[element_index];
    } else if (mode == Channel) {
      size_t temp = index / plane_size;
      size_t channel_index = temp % channel_num;
      scale = alpha_ptr[channel_index];
    } else {
      scale = alpha_ptr[0];
    }
    T x = x_ptr[index];
    T dy = dy_ptr[index];
    if (dx_ptr != nullptr) dx_ptr[index] = (x > 0) ? dy : scale * dy;
    if (dalpha_ptr != nullptr) dalpha_ptr[index] = (x > 0) ? 0 : x * dy;
  }
}

template <typename T>
class PreluOpGradFunctor {
 public:
  void operator()(cudaStream_t stream, const T* x, const T* alpha, const T* dy,
                  T* dx, T* dalpha, const framework::DDim& input_dims,
                  PRELU_MODE mode) {
    size_t numel = 1;
    for (size_t i = 0; i < input_dims.size(); ++i) {
      numel *= input_dims[i];
    }
    size_t plane_size = numel / input_dims[0] / input_dims[1];
    size_t spatial_size = numel / input_dims[0];

    PReluOpGradKernel<
        T><<<PADDLE_GET_BLOCKS(numel), CUDA_NUM_THREADS, 0, stream>>>(
        x, alpha, dy, dx, dalpha, input_dims[1], plane_size, spatial_size,
        numel, mode);
  }
};

template <typename T>
struct IdentityFunctor {
  HOSTDEVICE inline T operator()(const T& x) const { return x; }
};

template <typename DeviceContext, typename T>
class CUDAPReluGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* x = context.Input<Tensor>("X");
    auto* alpha = context.Input<Tensor>("Alpha");
    auto* dx = context.Output<Tensor>(framework::GradVarName("X"));
    auto* dy = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* dalpha = context.Output<Tensor>(framework::GradVarName("Alpha"));

    const T* x_ptr = x->data<T>();
    const T* alpha_ptr = alpha->data<T>();
    const T* dy_ptr = dy->data<T>();
    T* dx_ptr = dx ? dx->mutable_data<T>(context.GetPlace()) : nullptr;
    T* dalpha_ptr =
        dalpha ? dalpha->mutable_data<T>(context.GetPlace()) : nullptr;

    if (!dx && !dalpha) return;

    auto& mode = context.Attr<std::string>("mode");

    int numel = x->numel();
    auto dim = x->dims();
    std::vector<int> input_shape = framework::vectorize<int>(dim);
    auto stream = context.cuda_device_context().stream();

    T* dalpha_tmp_ptr;
    Tensor dalpha_tmp;
    if (dalpha_ptr == nullptr) {
      dalpha_tmp_ptr = dalpha_ptr;
    } else {
      auto& dev_ctx = context.template device_context<DeviceContext>();
      dalpha_tmp = context.AllocateTmpTensor<T, DeviceContext>(dim, dev_ctx);
      dalpha_tmp_ptr = dalpha_tmp.mutable_data<T>(context.GetPlace());
    }

    PRELU_MODE m;
    if (mode == "element") {
      m = Element;
    } else if (mode == "channel") {
      m = Channel;
    } else {
      m = Scalar;
    }
    PreluOpGradFunctor<T> prelu_grad;
    prelu_grad(stream, x_ptr, alpha_ptr, dy_ptr, dx_ptr, dalpha_tmp_ptr, dim,
               m);

    if (dalpha_tmp_ptr == nullptr) return;

    std::vector<int> reduce_dims;
    for (size_t i = 0; i < dim.size(); i++) {
      if (mode == "channel" && i == 1) continue;
      if (mode == "element" && i != 0) continue;
      reduce_dims.push_back(i);
    }

    TensorReduce<T, T, cub::Sum, IdentityFunctor<T>>(
        dalpha_tmp, dalpha, reduce_dims, static_cast<T>(0), cub::Sum(),
        IdentityFunctor<T>(), stream);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_CUDA_KERNEL(
    prelu, ops::CUDAPReluKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAPReluKernel<paddle::platform::CUDADeviceContext, double>);
REGISTER_OP_CUDA_KERNEL(
    prelu_grad,
    ops::CUDAPReluGradKernel<paddle::platform::CUDADeviceContext, float>,
    ops::CUDAPReluGradKernel<paddle::platform::CUDADeviceContext, double>);
