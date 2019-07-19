/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <iostream>
#include "paddle/fluid/operators/center_loss_op.h"
#include "paddle/fluid/platform/cuda_primitives.h"
#include "paddle/fluid/platform/gpu_info.h"
namespace paddle {
namespace operators {

using platform::PADDLE_CUDA_NUM_THREADS;

template <typename T>
__global__ void ComputeDistance(int64_t elements_num, const T *X,
                                const int x_width, const int64_t *label,
                                const T *centers_data, T *diff_xc) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < elements_num;
       index += blockDim.x * gridDim.x) {
    int row = index / x_width;
    int col = index % x_width;
    int64_t center_idx = label[row];
    diff_xc[index] = X[index] - centers_data[center_idx * x_width + col];
  }
}

template <typename T>
__global__ void UpdateCenters(int cluster_num, const int samples_num,
                              const int x_width, const int64_t *label,
                              const T *diff_xc, T *acc_sum, const T *alpha,
                              T *centers_out) {
  for (int64_t cidx = blockIdx.x * blockDim.x + threadIdx.x; cidx < cluster_num;
       cidx += blockDim.x * gridDim.x) {
    int count = 0;
    for (int sidx = 0; sidx < samples_num; sidx++) {
      if (label[sidx] == cidx) {
        count++;
        for (int col = 0; col < x_width; col++) {
          acc_sum[cidx * x_width + col] += diff_xc[sidx * x_width + col];
        }
      }
    }
    for (int col = 0; col < x_width; col++) {
      centers_out[cidx * x_width + col] +=
          alpha[0] * acc_sum[cidx * x_width + col] / (count + 1);
    }
  }
}

template <typename DeviceContext, typename T>
class CenterLossCUDAKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &device_context = ctx.template device_context<DeviceContext>();
    auto stream = device_context.stream();
    auto *X = ctx.Input<Tensor>("X");  // deep feature
    auto *labels = ctx.Input<Tensor>("Label");
    auto *centers = ctx.Input<Tensor>("Centers");
    auto *update_rate = ctx.Input<Tensor>("CenterUpdateRate");
    int cluster_num = ctx.Attr<int>("cluster_num");
    auto *lr_center = update_rate->data<T>();
    bool need_update = static_cast<T>(ctx.Attr<bool>("need_update"));

    auto x_data = X->data<T>();
    auto label_data = labels->data<int64_t>();

    auto x_dims = X->dims();
    int batch_size = x_dims[0];
    const int deep_feat_dim = x_dims[1];

    auto *centers_diff = ctx.Output<Tensor>("CentersDiff");
    auto centers_diff_data = centers_diff->mutable_data<T>(ctx.GetPlace());

    auto centers_data = centers->data<T>();
    auto centers_dim = centers->dims();
    auto *out_loss = ctx.Output<Tensor>("Loss");
    auto loss_data = out_loss->mutable_data<T>(ctx.GetPlace());

    auto *centers_out = ctx.Output<Tensor>("CentersOut");
    auto *centers_out_data = centers_out->mutable_data<T>(ctx.GetPlace());

    auto ctx_place = ctx.GetPlace();
    if (centers != centers_out) {
      framework::TensorCopy(
          *static_cast<const framework::Tensor *>(centers), ctx_place,
          *platform::DeviceContextPool::Instance().Get(ctx_place),
          static_cast<framework::Tensor *>(centers_out));
    }

    int64_t numel = X->numel();

    ComputeDistance<
        T><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
             PADDLE_CUDA_NUM_THREADS, 0, stream>>>(numel, x_data, deep_feat_dim,
                                                   label_data, centers_data,
                                                   centers_diff_data);
    auto &place = *ctx.template device_context<DeviceContext>().eigen_device();
    auto sub_result = EigenMatrix<T>::From(*centers_diff);
    auto sub_res_pow2 = (sub_result * sub_result) / T(2.0);
    auto z = EigenVector<T>::Flatten(*out_loss);
    z.device(place) = sub_res_pow2.sum(Eigen::array<int, 1>({{1}}));
    if (need_update) {
      Tensor centers_diffacc;  // used to accumulate all diff of center and x
      auto *centers_diffacc_data =
          centers_diffacc.mutable_data<T>(centers_dim, ctx.GetPlace());
      numel = centers_diffacc.numel();
      cudaMemsetAsync(centers_diffacc_data, 0, sizeof(T) * numel, stream);
      UpdateCenters<T><<<(cluster_num + PADDLE_CUDA_NUM_THREADS - 1) /
                             PADDLE_CUDA_NUM_THREADS,
                         PADDLE_CUDA_NUM_THREADS, 0, stream>>>(
          cluster_num, batch_size, deep_feat_dim, label_data, centers_diff_data,
          centers_diffacc_data, lr_center, centers_out_data);
    }
  }
};
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
using GPUCtx = paddle::platform::CUDADeviceContext;
REGISTER_OP_CUDA_KERNEL(center_loss, ops::CenterLossCUDAKernel<GPUCtx, float>,
                        ops::CenterLossCUDAKernel<GPUCtx, double>);

REGISTER_OP_CUDA_KERNEL(center_loss_grad,
                        ops::CenterLossGradKernel<GPUCtx, float>,
                        ops::CenterLossGradKernel<GPUCtx, double>);
