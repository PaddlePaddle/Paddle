/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
Indicesou may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef __NVCC__
#include "cub/cub.cuh"
#endif

#ifdef __HIPCC__
#include <hipcub/hipcub.hpp>
namespace cub = hipcub;
#endif

#include "paddle/fluid/framework/data_layout.h"
#include "paddle/fluid/framework/op_registry.h"
<<<<<<< HEAD
#include "paddle/phi/backends/gpu/gpu_primitives.h"
=======
#include "paddle/fluid/platform/device/gpu/gpu_primitives.h"
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81

namespace paddle {
namespace operators {

<<<<<<< HEAD
template <typename T, phi::DataLayout layout, bool HasBias>
=======
template <typename T, framework::DataLayout layout, bool HasBias>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
__global__ void KeAffineChannelCUDA(const T* x,
                                    const T* scale,
                                    const T* bias,
                                    const int C,
                                    const int HxW,
                                    const int num,
                                    T* y) {
  int gid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int i = gid; i < num; i += stride) {
<<<<<<< HEAD
    const int c = layout == phi::DataLayout::kNCHW ? i / HxW % C : i % C;
=======
    const int c = layout == framework::DataLayout::kNCHW ? i / HxW % C : i % C;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    if (HasBias) {
      y[i] = scale[c] * x[i] + bias[c];
    } else {
      y[i] = scale[c] * x[i];
    }
  }
}

template <typename DeviceContext, typename T>
class AffineChannelCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
<<<<<<< HEAD
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");

    auto* y = ctx.Output<phi::DenseTensor>("Out");
    y->mutable_data<T>(ctx.GetPlace());

    const phi::DataLayout layout =
        phi::StringToDataLayout(ctx.Attr<std::string>("data_layout"));
=======
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto* bias = ctx.Input<framework::Tensor>("Bias");

    auto* y = ctx.Output<framework::Tensor>("Out");
    y->mutable_data<T>(ctx.GetPlace());

    const framework::DataLayout layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto dims = x->dims();
    const int num = x->numel();
    int N = dims[0];
<<<<<<< HEAD
    int C = layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
=======
    int C = layout == framework::DataLayout::kNCHW ? dims[1]
                                                   : dims[dims.size() - 1];
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    int HxW = num / N / C;

    const T* x_d = x->data<T>();
    const T* scale_d = scale->data<T>();
    const T* bias_d = bias->data<T>();
    T* y_d = y->data<T>();

#ifdef PADDLE_WITH_HIP
    int block = 256;
#else
    int block = 1024;
#endif  // PADDLE_WITH_HIP
    int grid = (num + block - 1) / block;

    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    grid = std::min(std::max(max_threads / block, 1), grid);
<<<<<<< HEAD
    if (layout == phi::DataLayout::kNCHW) {
      KeAffineChannelCUDA<T, phi::DataLayout::kNCHW, true>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              x_d, scale_d, bias_d, C, HxW, num, y_d);
    } else {
      KeAffineChannelCUDA<T, phi::DataLayout::kNHWC, true>
=======
    if (layout == framework::DataLayout::kNCHW) {
      KeAffineChannelCUDA<T, framework::DataLayout::kNCHW, true>
          <<<grid, block, 0, dev_ctx.stream()>>>(
              x_d, scale_d, bias_d, C, HxW, num, y_d);
    } else {
      KeAffineChannelCUDA<T, framework::DataLayout::kNHWC, true>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
          <<<grid, block, 0, dev_ctx.stream()>>>(
              x_d, scale_d, bias_d, C, HxW, num, y_d);
    }
  }
};

<<<<<<< HEAD
template <typename T, int BlockDim, phi::DataLayout layout>
=======
template <typename T, int BlockDim, framework::DataLayout layout>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
__global__ void AffineChannelScaleBiasGradientCUDAKernel(const T* dy,
                                                         const T* x,
                                                         const int N,
                                                         const int C,
                                                         const int HxW,
                                                         T* dscale,
                                                         T* dbias) {
  const int outer_size = C;
  const int inner_size = N * HxW;
  typedef cub::BlockReduce<double, BlockDim> BlockReduce;
  __shared__ typename BlockReduce::TempStorage ds_storage;
  __shared__ typename BlockReduce::TempStorage db_storage;

  for (int i = blockIdx.x; i < outer_size; i += gridDim.x) {
    T ds_sum = 0;
    T db_sum = 0;
    for (int j = threadIdx.x; j < inner_size; j += blockDim.x) {
<<<<<<< HEAD
      const int index = layout == phi::DataLayout::kNCHW
=======
      const int index = layout == framework::DataLayout::kNCHW
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
                            ? (j / HxW * C + i) * HxW + j % HxW
                            : j * outer_size + i;
      ds_sum += dy[index] * x[index];
      db_sum += dy[index];
    }
    __syncthreads();
    auto ds_out =
        BlockReduce(ds_storage).Reduce(static_cast<double>(ds_sum), cub::Sum());
    auto db_out =
        BlockReduce(db_storage).Reduce(static_cast<double>(db_sum), cub::Sum());
    __syncthreads();
    if (threadIdx.x == 0) {
      dscale[i] = ds_out;
      dbias[i] = db_out;
    }
  }
}

template <typename DeviceContext, typename T>
class AffineChannelGradCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
<<<<<<< HEAD
    auto* x = ctx.Input<phi::DenseTensor>("X");
    auto* scale = ctx.Input<phi::DenseTensor>("Scale");
    auto* bias = ctx.Input<phi::DenseTensor>("Bias");
    auto* dy = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<phi::DenseTensor>(framework::GradVarName("Bias"));

    const phi::DataLayout layout =
        phi::StringToDataLayout(ctx.Attr<std::string>("data_layout"));
=======
    auto* x = ctx.Input<framework::Tensor>("X");
    auto* scale = ctx.Input<framework::Tensor>("Scale");
    auto* bias = ctx.Input<framework::Tensor>("Bias");
    auto* dy = ctx.Input<framework::Tensor>(framework::GradVarName("Out"));

    auto* dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto* dscale =
        ctx.Output<framework::Tensor>(framework::GradVarName("Scale"));
    auto* dbias = ctx.Output<framework::Tensor>(framework::GradVarName("Bias"));

    const framework::DataLayout layout =
        framework::StringToDataLayout(ctx.Attr<std::string>("data_layout"));
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto dims = dy->dims();
    const int num = dy->numel();
    int N = dims[0];
<<<<<<< HEAD
    int C = layout == phi::DataLayout::kNCHW ? dims[1] : dims[dims.size() - 1];
=======
    int C = layout == framework::DataLayout::kNCHW ? dims[1]
                                                   : dims[dims.size() - 1];
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    int HxW = num / N / C;

    const T* dy_d = dy->data<T>();
    const T* s_d = scale->data<T>();

    T* dx_d = dx ? dx->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* ds_d = dscale ? dscale->mutable_data<T>(ctx.GetPlace()) : nullptr;
    T* db_d = dbias ? dbias->mutable_data<T>(ctx.GetPlace()) : nullptr;

#ifdef PADDLE_WITH_HIP
    const int block = 256;
#else
    const int block = 1024;
#endif  // PADDLE_WITH_HIP
    int max_threads = dev_ctx.GetMaxPhysicalThreadCount();
    const int max_blocks = std::max(max_threads / block, 1);
    int grid1 = (num + block - 1) / block;
    int grid2 = std::min(C, max_blocks);
<<<<<<< HEAD
    if (layout == phi::DataLayout::kNCHW) {
=======
    if (layout == framework::DataLayout::kNCHW) {
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      if (dscale && dbias) {
        const T* x_d = x->data<T>();
        AffineChannelScaleBiasGradientCUDAKernel<T,
                                                 block,
<<<<<<< HEAD
                                                 phi::DataLayout::kNCHW>
=======
                                                 framework::DataLayout::kNCHW>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            <<<grid2, block, 0, dev_ctx.stream()>>>(
                dy_d, x_d, N, C, HxW, ds_d, db_d);
      }
      if (dx) {
<<<<<<< HEAD
        KeAffineChannelCUDA<T, phi::DataLayout::kNCHW, false>
=======
        KeAffineChannelCUDA<T, framework::DataLayout::kNCHW, false>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            <<<grid1, block, 0, dev_ctx.stream()>>>(
                dy_d, s_d, nullptr, C, HxW, num, dx_d);
      }
    } else {
      if (dscale && dbias) {
        const T* x_d = x->data<T>();
        AffineChannelScaleBiasGradientCUDAKernel<T,
                                                 block,
<<<<<<< HEAD
                                                 phi::DataLayout::kNHWC>
=======
                                                 framework::DataLayout::kNHWC>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            <<<grid2, block, 0, dev_ctx.stream()>>>(
                dy_d, x_d, N, C, HxW, ds_d, db_d);
      }

      if (dx) {
<<<<<<< HEAD
        KeAffineChannelCUDA<T, phi::DataLayout::kNHWC, false>
=======
        KeAffineChannelCUDA<T, framework::DataLayout::kNHWC, false>
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
            <<<grid1, block, 0, dev_ctx.stream()>>>(
                dy_d, s_d, nullptr, C, HxW, num, dx_d);
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using CUDA = phi::GPUContext;

REGISTER_OP_CUDA_KERNEL(affine_channel,
                        ops::AffineChannelCUDAKernel<CUDA, float>,
                        ops::AffineChannelCUDAKernel<CUDA, double>);
REGISTER_OP_CUDA_KERNEL(affine_channel_grad,
                        ops::AffineChannelGradCUDAKernel<CUDA, float>,
                        ops::AffineChannelGradCUDAKernel<CUDA, double>);
