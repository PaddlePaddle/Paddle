/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.

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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/lu_op.h"
#include "paddle/fluid/operators/tril_triu_op.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensorArray = framework::LoDTensorArray;

template <typename DeviceContext, typename T>
class LU_UnpackKernel : public framework::OpKernel<T> {
 public:
  void Compute(const paddle::framework::ExecutionContext& ctx) const override {
    auto xin = ctx.Input<framework::Tensor>("X");
    auto P = ctx.Input<framework::Tensor>("Pivots");

    auto ltensor = ctx.Output<framework::Tensor>("L");
    auto utensor = ctx.Output<framework::Tensor>("U");
    auto ptensor = ctx.Output<framework::Tensor>("Pmat");

    auto unpack_ludata = ctx.Attr<bool>("unpack_ludata");
    auto unpack_pivots = ctx.Attr<bool>("unpack_pivots");

    const auto& dev_ctx = ctx.template device_context<DeviceContext>();

    auto xdims = xin->dims();
    int xrank = xdims.size();
    int64_t m = xdims[xrank - 2];
    int64_t n = xdims[xrank - 1];
    int64_t k = std::min(m, n);

    if (unpack_ludata) {
      ltensor->mutable_data<T>(ctx.GetPlace());
      utensor->mutable_data<T>(ctx.GetPlace());

      framework::Tensor L, U;
      LU_Unpack<DeviceContext, T>(dev_ctx, xin, &L, &U);

      if (m >= n) {
        framework::TensorCopy(L, ctx.GetPlace(), ltensor);
        Tensor_narrow<DeviceContext, T>(ctx, &U, utensor, 0, k, 0, k);
      } else {
        framework::TensorCopy(U, ctx.GetPlace(), utensor);
        Tensor_narrow<DeviceContext, T>(ctx, &L, ltensor, 0, k, 0, k);
      }
    }

    if (unpack_pivots) {
      ptensor->mutable_data<T>(ctx.GetPlace());
      Unpack_Pivot<DeviceContext, T>(dev_ctx, *P, ptensor, m, k);
    }
  }
};

template <typename DeviceContext, typename T>
class LU_UnpackGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto dl = ctx.Input<framework::Tensor>(framework::GradVarName("L"));
    auto du = ctx.Input<framework::Tensor>(framework::GradVarName("U"));
    auto dx = ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    dx->mutable_data<T>(ctx.GetPlace());

    const auto& dev_ctx = ctx.template device_context<DeviceContext>();

    framework::Tensor dl_tril, du_triu;
    const auto ldims = dl->dims();
    dl_tril.Resize(ldims);
    auto H = ldims[ldims.size() - 2];
    auto W = ldims[ldims.size() - 1];
    auto L_dataptr = dl_tril.mutable_data<T>(dev_ctx.GetPlace());
    platform::ForRange<DeviceContext> l_for_range(dev_ctx, dl->numel());
    TrilTriuCompute<T> tril_computer(dl->data<T>(), -1, true, H, W, L_dataptr);
    l_for_range(tril_computer);

    const auto udims = du->dims();
    du_triu.Resize(udims);
    H = udims[udims.size() - 2];
    W = udims[udims.size() - 1];
    auto U_dataptr = du_triu.mutable_data<T>(dev_ctx.GetPlace());
    platform::ForRange<DeviceContext> u_for_range(dev_ctx, du->numel());
    TrilTriuCompute<T> triu_computer(du->data<T>(), 0, false, H, W, U_dataptr);
    u_for_range(triu_computer);

    auto xdims = dx->dims();
    int xrank = xdims.size();
    int64_t m = xdims[xrank - 2];
    int64_t n = xdims[xrank - 1];
    int64_t k = std::min(m, n);

    std::vector<int64_t> axes = {xrank - 2, xrank - 1};
    std::vector<int64_t> slice_starts(2, 0);
    std::vector<int64_t> slice_ends(2, 0);
    auto valuedims = vectorize(xdims);

    pten::funcs::SetConstant<DeviceContext, T> setter;
    setter(dev_ctx, dx, static_cast<T>(0));
    if (m <= n) {
      slice_starts[0] = 0;
      slice_starts[1] = 0;
      slice_ends[0] = k;
      slice_ends[1] = k;
      valuedims[xrank - 2] = k;
      valuedims[xrank - 1] = k;
      SetValueCompute_dispatch<DeviceContext, T>(ctx, dx, &dl_tril, dx, axes,
                                                 &slice_starts, &slice_ends,
                                                 valuedims, xrank);

      Tensor_Add<DeviceContext, T>(dev_ctx, *dx, du_triu, dx);
    } else {
      slice_starts[0] = 0;
      slice_starts[1] = 0;
      slice_ends[0] = k;
      slice_ends[1] = k;
      valuedims[xrank - 2] = k;
      valuedims[xrank - 1] = k;
      SetValueCompute_dispatch<DeviceContext, T>(ctx, dx, &du_triu, dx, axes,
                                                 &slice_starts, &slice_ends,
                                                 valuedims, xrank);

      Tensor_Add<DeviceContext, T>(dev_ctx, *dx, dl_tril, dx);
    }
  }
};

}  // namespace operators
}  // namespace paddle
