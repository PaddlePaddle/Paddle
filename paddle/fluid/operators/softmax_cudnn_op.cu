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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/math_cuda_utils.h"
#include "paddle/fluid/operators/softmax_op.h"
#include "paddle/fluid/platform/cudnn_helper.h"

namespace paddle {
namespace platform {
struct CUDAPlace;
struct float16;
}  // namespace platform
}  // namespace paddle

namespace paddle {
namespace operators {

using ScopedTensorDescriptor = platform::ScopedTensorDescriptor;
using DataLayout = platform::DataLayout;
using Tensor = framework::Tensor;

static inline int SizeOutAxis(const int axis, DDim dims) {
  int size = 1;
  for (int i = axis + 1; i < dims.size(); i++) {
    size *= dims[i];
  }
  return size;
}

template <typename T, int VLEN>
union vec_t {
  static_assert(sizeof(T) == -1, "vec_t is only available by specialization.");
};

template <>
union vec_t<float, 4> {
  float4 s;
  float v[4];
};

template <>
union vec_t<platform::float16, 4> {
  int2 s;
  platform::float16 v[4];
};

template <typename T, typename VECT, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxForward(T* dst, const T* src, const int batch_size,
                                  const int softmax_ele) {
  int offset = blockIdx.x * softmax_ele * WARP_PER_BLOCK;
  int idx = threadIdx.x * VPT;

  VECT buf = reinterpret_cast<const VECT*>(&src[offset + idx])[0];
  T* bufp = reinterpret_cast<T*>(&buf);
  float4 val4;
  float* val4p = reinterpret_cast<float*>(&val4);
  for (int i = 0; i < VPT; ++i) {
    val4p[i] = static_cast<float>(bufp[i]);
  }
  float val = val4.x + val4.y + val4.z + val4.w;
  float max_val = math::warpReduceMax<float>(
      max(max(val4.x, val4.y), max(val4.z, val4.w)), 0xffffffff);
  float4 tmp4 = make_float4(__expf(val4.x - max_val), __expf(val4.y - max_val),
                            __expf(val4.z - max_val), __expf(val4.w - max_val));
  float* tmp4p = reinterpret_cast<float*>(&tmp4);
  float invsum = 1.f / (math::warpReduceSum<float>(
                            tmp4.x + tmp4.y + tmp4.z + tmp4.w, 0xffffffff) +
                        1e-6f);
  for (int i = 0; i < VPT; ++i) {
    bufp[i] = static_cast<T>(tmp4p[i] * invsum);
  }
  reinterpret_cast<VECT*>(&dst[offset + idx])[0] = buf;
}

template <typename T, int VPT, int WARP_PER_BLOCK>
__global__ void VecSoftmaxBackward(T* dst, const T* grad, const T* src,
                                   const int batch_size,
                                   const int softmax_ele) {
  const int offset =
      blockIdx.x * softmax_ele * WARP_PER_BLOCK + threadIdx.x * VPT;

  float local_sum_gy = 0.f;
  vec_t<T, VPT> local_grad;
  vec_t<T, VPT> local_src;

  local_grad.s =
      reinterpret_cast<const decltype(local_grad.s)*>(&grad[offset])[0];
  local_src.s = reinterpret_cast<const decltype(local_src.s)*>(&src[offset])[0];

  for (int i = 0; i < VPT; ++i) {
    local_sum_gy += static_cast<float>(local_grad.v[i]) *
                    static_cast<float>(local_src.v[i]);
  }
  float sum_gy = math::warpReduceSum<float>(local_sum_gy, 0xffffffff);

  vec_t<T, VPT> local_dst;
  for (int i = 0; i < VPT; ++i) {
    local_dst.v[i] =
        static_cast<T>(static_cast<float>(local_src.v[i]) *
                       (static_cast<float>(local_grad.v[i]) - sum_gy));
  }
  reinterpret_cast<decltype(local_dst.s)*>(&dst[offset])[0] = local_dst.s;
}

template <typename T>
class SoftmaxCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* x = ctx.Input<Tensor>("X");
    auto* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    auto* out_data = out->data<T>();

    auto dims = x->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int warps_per_block = 4;
    if (D == 1 && dim == 128 && N % warps_per_block == 0 && sizeof(T) <= 4) {
      // a warp for a batch, 4 elements for a thread, only support the softmax
      // dim size = 128 currently
      if (sizeof(T) == 2) {
        VecSoftmaxForward<
            T, int2, 4,
            warps_per_block><<<N / warps_per_block, warps_per_block * WARP_SIZE,
                               0, ctx.cuda_device_context().stream()>>>(
            out_data, x->data<T>(), N, dim);
      } else if (sizeof(T) == 4) {
        VecSoftmaxForward<
            T, int4, 4,
            warps_per_block><<<N / warps_per_block, warps_per_block * WARP_SIZE,
                               0, ctx.cuda_device_context().stream()>>>(
            out_data, x->data<T>(), N, dim);
      } else {
        assert(false && "not support");
      }
    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxForward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, x->data<T>(),
          platform::CudnnDataType<T>::kZero(), desc_, out_data));
    }
  }
};

template <typename T>
class SoftmaxGradCUDNNKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* out = ctx.Input<Tensor>("Out");
    auto* dout = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto* dx = ctx.Output<Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());
    auto* dx_data = dx->data<T>();

    auto dims = out->dims();
    const int rank = dims.size();
    const int axis = CanonicalAxis(ctx.Attr<int>("axis"), rank);
    const int dim = dims[axis];
    const int N = SizeToAxis(axis, dims);
    const int D = SizeOutAxis(axis, dims);

    constexpr int warps_per_block = 4;
    constexpr bool warp_softmax_available =
        std::is_same<T, float>::value ||
        std::is_same<T, platform::float16>::value;
    if (D == 1 && dim == 128 && N % warps_per_block == 0 &&
        warp_softmax_available) {
      if (std::is_same<T, float>::value) {
        VecSoftmaxBackward<
            float, 4,
            warps_per_block><<<N / warps_per_block, warps_per_block * WARP_SIZE,
                               0, ctx.cuda_device_context().stream()>>>(
            dx->data<float>(), dout->data<float>(), out->data<float>(), N, dim);
      } else if (std::is_same<T, platform::float16>::value) {
        VecSoftmaxBackward<
            platform::float16, 4,
            warps_per_block><<<N / warps_per_block, warps_per_block * WARP_SIZE,
                               0, ctx.cuda_device_context().stream()>>>(
            dx->data<platform::float16>(), dout->data<platform::float16>(),
            out->data<platform::float16>(), N, dim);
      } else {
        PADDLE_ENFORCE_EQ(
            warp_softmax_available, true,
            platform::errors::Unimplemented(
                "Warp softmax backward is only available for fp32 and fp16"));
      }
    } else {
      ScopedTensorDescriptor desc;
      std::vector<int> tensor_dims = {N, dim, D, 1};
      DataLayout layout = DataLayout::kNCHW;
      cudnnTensorDescriptor_t desc_ = desc.descriptor<T>(layout, tensor_dims);

      auto& dev_ctx =
          ctx.template device_context<platform::CUDADeviceContext>();
      auto handle = dev_ctx.cudnn_handle();
      auto mode = axis == rank - 1 ? CUDNN_SOFTMAX_MODE_INSTANCE
                                   : CUDNN_SOFTMAX_MODE_CHANNEL;

      PADDLE_ENFORCE_CUDA_SUCCESS(platform::dynload::cudnnSoftmaxBackward(
          handle, CUDNN_SOFTMAX_ACCURATE, mode,
          platform::CudnnDataType<T>::kOne(), desc_, out->data<T>(), desc_,
          dout->data<T>(), platform::CudnnDataType<T>::kZero(), desc_,
          dx_data));
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
namespace plat = paddle::platform;
REGISTER_OP_KERNEL(softmax, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxCUDNNKernel<float>,
                   ops::SoftmaxCUDNNKernel<double>,
                   ops::SoftmaxCUDNNKernel<plat::float16>);
REGISTER_OP_KERNEL(softmax_grad, CUDNN, plat::CUDAPlace,
                   ops::SoftmaxGradCUDNNKernel<float>,
                   ops::SoftmaxGradCUDNNKernel<double>,
                   ops::SoftmaxGradCUDNNKernel<plat::float16>);
