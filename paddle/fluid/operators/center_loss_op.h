/*Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <algorithm>
#include <cstring>
#include <limits>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/transform.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {

template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T,
          int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename T>
struct SubFunctor {
  inline HOSTDEVICE T operator()(T a, T b) const { return a - b; }
};

template <typename DeviceContext, typename T>
class CenterLossKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *X = ctx.Input<phi::DenseTensor>("X");  // deep feature
    auto *labels = ctx.Input<phi::DenseTensor>("Label");
    auto *centers = ctx.Input<phi::DenseTensor>("Centers");
    auto *update_rate = ctx.Input<phi::DenseTensor>("CenterUpdateRate");
    int cluster_num = ctx.Attr<int>("cluster_num");
    auto *lr_center = update_rate->data<T>();
    T alpha = lr_center[0];
    bool need_update = static_cast<T>(ctx.Attr<bool>("need_update"));

    auto x_data = X->data<T>();
    auto label_data = labels->data<int64_t>();

    auto centers_dim = centers->dims();
    auto centers_data = centers->data<T>();

    auto x_dims = X->dims();
    int batch_size = x_dims[0];
    int deep_feat_dim = x_dims[1];

    auto centers_diff = ctx.Output<phi::DenseTensor>("SampleCenterDiff");
    auto centers_diff_data = centers_diff->mutable_data<T>(ctx.GetPlace());
    auto *out_loss = ctx.Output<phi::DenseTensor>("Loss");

    auto *centers_out = ctx.Output<phi::DenseTensor>("CentersOut");
    auto *centers_out_data = centers_out->mutable_data<T>(ctx.GetPlace());

    if (centers_out_data != centers_data) {
      int size = centers_out->numel() * sizeof(T);
      memcpy(centers_out_data, centers_data, size);
    }

    std::vector<int> center_update_count(cluster_num, 1);
    auto &dev_ctx = ctx.template device_context<DeviceContext>();

    auto loss_data = out_loss->mutable_data<T>(ctx.GetPlace());

    phi::DenseTensor centers_diffacc;  // used to accumulate all diff
    auto centers_diffacc_data =
        centers_diffacc.mutable_data<T>(centers_dim, ctx.GetPlace());
    int numel = centers_diffacc.numel();
    std::memset(centers_diffacc_data, 0, sizeof(T) * numel);

    auto blas = phi::funcs::GetBlas<DeviceContext, T>(ctx);
    int tLabel;

    const T *x_index;
    const T *center_index;
    T *center_out_index;
    T *center_loss_diff_index;
    T *acc_index;
    platform::Transform<DeviceContext> trans;

    for (int i = 0; i < batch_size; ++i) {
      tLabel = label_data[i];
      center_update_count[tLabel]++;
      x_index = x_data + i * deep_feat_dim;                  // xi index
      center_index = centers_data + tLabel * deep_feat_dim;  // center index
      center_loss_diff_index = centers_diff_data + i * deep_feat_dim;
      trans(dev_ctx,
            x_index,
            x_index + deep_feat_dim,
            center_index,
            center_loss_diff_index,
            SubFunctor<T>());

      acc_index = centers_diffacc_data + tLabel * deep_feat_dim;
      blas.VADD(deep_feat_dim,
                center_loss_diff_index,
                acc_index,
                acc_index);  // accumulate
      loss_data[i] =
          blas.DOT(
              deep_feat_dim, center_loss_diff_index, center_loss_diff_index) /
          T(2.0);
    }

    // update centers data
    if (need_update == true) {
      for (int i = 0; i < cluster_num; i++) {
        acc_index = centers_diffacc_data + i * deep_feat_dim;
        center_out_index = centers_out_data + i * deep_feat_dim;
        T scale = alpha / center_update_count[i];
        blas.SCAL(deep_feat_dim, scale, acc_index);
        blas.VADD(deep_feat_dim, acc_index, center_out_index, center_out_index);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class CenterLossGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *in0 = context.Input<phi::DenseTensor>("SampleCenterDiff");
    auto *in1 = context.Input<phi::DenseTensor>(framework::GradVarName("Loss"));
    auto *x_g = context.Output<phi::DenseTensor>(framework::GradVarName("X"));
    auto sub_result = EigenMatrix<T>::From(*in0);
    auto out_grad = EigenMatrix<T>::From(*in1);

    auto x_dims = x_g->dims();
    int cols = x_g->numel() / x_dims[0];
    // calculate gradient
    auto grad_mat =
        (out_grad.broadcast(Eigen::array<int, 2>({{1, cols}}))) * sub_result;

    // propagate back to input
    auto &eigen_place =
        *context.template device_context<DeviceContext>().eigen_device();
    x_g->mutable_data<T>(context.GetPlace());
    // eigen matrix
    auto x_grad = EigenMatrix<T>::From(*x_g, phi::make_ddim({x_dims[0], cols}));
    x_grad.device(eigen_place) = grad_mat;
  }
};

}  // namespace operators
}  // namespace paddle
