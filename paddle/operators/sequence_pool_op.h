/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/math/math_function.h"
#include "paddle/operators/math/sequence_pooling.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenVector = framework::EigenVector<T, MajorType, IndexType>;
template <typename T, int MajorType = Eigen::RowMajor,
          typename IndexType = Eigen::DenseIndex>
using EigenMatrix = framework::EigenMatrix<T, MajorType, IndexType>;

template <typename Place, typename T>
class SequencePoolKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out = context.Output<Tensor>("Out");
    std::string pooltype = context.Attr<std::string>("pooltype");

    auto dims = in->dims();
    auto lod = in->lod();
    int64_t w = in->numel() / dims[0];

    // InferShape by lod
    PADDLE_ENFORCE_EQ(lod.size(), 1UL, "Only support one level sequence now.");
    PADDLE_ENFORCE_GE(
        dims[0],
        /*batch size = */ static_cast<int64_t>(lod[0].size() - 1),
        "The first dimension of Input(X) must be large than batch size.");
    dims[0] = lod[0].size() - 1;
    out->Resize({dims});

    auto lod_level_0 = lod[0];

    out->mutable_data<T>(context.GetPlace());

    if (pooltype == "MAX") {
      math::MaxSeqPoolFunctor<Place, T> max_pool;
      auto* index = context.Output<Tensor>("MaxIndex");
      index->Resize({dims});
      index->mutable_data<int>(context.GetPlace());
      max_pool(context.device_context(), *in, out, index);
      return;
    }

    auto place = context.GetEigenDevice<Place>();
    for (int i = 0; i < static_cast<int>(lod_level_0.size()) - 1; ++i) {
      Tensor in_t = in->Slice(static_cast<int>(lod_level_0[i]),
                              static_cast<int>(lod_level_0[i + 1]));
      Tensor out_t = out->Slice(i, i + 1);
      int64_t h = static_cast<int64_t>(lod_level_0[i + 1] - lod_level_0[i]);
      auto in_e = EigenMatrix<T>::From(in_t, framework::make_ddim({h, w}));
      auto out_e = EigenVector<T>::Flatten(out_t);

      if (pooltype == "AVERAGE") {
        out_e.device(place) = in_e.mean(Eigen::array<int, 1>({{0}}));
      } else if (pooltype == "SUM") {
        out_e.device(place) = in_e.sum(Eigen::array<int, 1>({{0}}));
      } else if (pooltype == "SQRT") {
        out_e.device(place) = in_e.sum(Eigen::array<int, 1>({{0}})) /
                              std::sqrt(static_cast<T>(h));
      } else if (pooltype == "LAST") {
        out_e.device(place) = in_e.chip(h - 1, 0);
      } else if (pooltype == "FIRST") {
        out_e.device(place) = in_e.chip(0, 0);
      } else {
        PADDLE_THROW("unsupported pooling pooltype");
      }
    }
  }
};

template <typename Place, typename T>
class SequencePoolGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& context) const override {
    auto* in = context.Input<LoDTensor>("X");
    auto* out_g = context.Input<Tensor>(framework::GradVarName("Out"));
    auto* in_g = context.Output<LoDTensor>(framework::GradVarName("X"));
    std::string pooltype = context.Attr<std::string>("pooltype");

    auto dims = in->dims();
    auto lod = in->lod()[0];
    int64_t w = in->numel() / dims[0];

    in_g->mutable_data<T>(context.GetPlace());

    if (pooltype == "MAX") {
      math::MaxSeqPoolGradFunctor<Place, T> max_pool_grad;
      auto* index = context.Input<Tensor>("MaxIndex");
      max_pool_grad(context.device_context(), *out_g, *index, in_g);
      return;
    }

    if (pooltype == "LAST" || pooltype == "FIRST") {
      // set X@Grad be zero at first when pooltype is LAST/FIRST
      math::SetConstant<Place, T> functor;
      functor(context.device_context(), in_g, 0);
    }
    auto place = context.GetEigenDevice<Place>();
    for (int i = 0; i < static_cast<int>(lod.size()) - 1; ++i) {
      auto in_g_t =
          in_g->Slice(static_cast<int>(lod[i]), static_cast<int>(lod[i + 1]));
      auto out_g_t = out_g->Slice(i, i + 1);
      int64_t h = static_cast<int64_t>(lod[i + 1] - lod[i]);
      auto in_g_e = EigenMatrix<T>::From(in_g_t, {h, w});
      auto out_g_e = EigenMatrix<T>::From(out_g_t, {1, w});
      auto out_g_e_v = EigenVector<T>::Flatten(out_g_t);
      Eigen::DSizes<int, 2> bcast(h, 1);

      if (pooltype == "AVERAGE") {
        in_g_e.device(place) = (out_g_e / static_cast<T>(h)).broadcast(bcast);
      } else if (pooltype == "SUM") {
        in_g_e.device(place) = (out_g_e).broadcast(bcast);
      } else if (pooltype == "SQRT") {
        in_g_e.device(place) =
            (out_g_e / std::sqrt(static_cast<T>(h))).broadcast(bcast);
      } else if (pooltype == "LAST") {
        in_g_e.chip(h - 1, 0).device(place) = out_g_e_v;
      } else if (pooltype == "FIRST") {
        in_g_e.chip(0, 0).device(place) = out_g_e_v;
      } else {
        PADDLE_THROW("unsupported pooling pooltype");
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
