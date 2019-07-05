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
__global__ void ComputeDistance(int64_t total_num, const int K, const T *X,
                                const int64_t *label, const T *centers_data,
                                T *distance) {
  for (int index = blockIdx.x * blockDim.x + threadIdx.x; index < total_num;
       index += blockDim.x * gridDim.x) {
    int m = index / K;
    int k = index % K;
    int64_t label_value = label[m];
    distance[index] = X[index] - centers_data[label_value * K + k];
  }
}

template <typename T>
__global__ void ComputeCenterDiffGpu(int num, const int M, const int K,
                                     const int64_t *label, const T *distance,
                                     T *variation_sum, const T *alpha,
                                     T *centers_out) {
  for (int64_t index = blockIdx.x * blockDim.x + threadIdx.x; index < num;
       index += blockDim.x * gridDim.x) {
    int count = 0;
    for (int m = 0; m < M; m++) {
      int64_t label_value = label[m];
      if (label_value == index) {
        count++;
        for (int k = 0; k < K; k++) {
          variation_sum[index * K + k] += distance[m * K + k];
        }
      }
    }
    for (int k = 0; k < K; k++) {
      centers_out[index * K + k] +=
          alpha[0] * variation_sum[index * K + k] / (count + (T)1.);
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

    if (centers_out_data != centers_data) {
      int size = centers_out->numel() * sizeof(T);
      cudaMemcpyAsync(centers_out_data, centers_data, size,
                      cudaMemcpyDeviceToDevice, stream);
    }

    int64_t numel = X->numel();

    ComputeDistance<
        T><<<(numel + PADDLE_CUDA_NUM_THREADS - 1) / PADDLE_CUDA_NUM_THREADS,
             PADDLE_CUDA_NUM_THREADS, 0, stream>>>(numel, deep_feat_dim, x_data,
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
      ComputeCenterDiffGpu<T><<<(cluster_num + PADDLE_CUDA_NUM_THREADS - 1) /
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
REGISTER_OP_CUDA_KERNEL(center_loss, ops::CenterLossCUDAKernel<GPUCtx, float>);

REGISTER_OP_CUDA_KERNEL(center_loss_grad,
                        ops::CenterLossGradKernel<GPUCtx, float>);
