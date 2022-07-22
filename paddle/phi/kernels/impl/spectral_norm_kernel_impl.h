// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#pragma once

#include "paddle/phi/core/dense_tensor.h"

namespace phi {

template <typename T, typename Context>
void SpectrumNormKernel(const Context& dev_ctx
                        const DenseTensor& weight,
                        const DenseTensor& u,
                        const DenseTensor& v,
                        int dim,
                        int power_iters,
                        float eps,
                        DenseTensor* out)
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    auto weight = ctx.Input<Tensor>("Weight");
    auto u = ctx.Input<Tensor>("U");
    auto v = ctx.Input<Tensor>("V");
    auto out = ctx.Output<Tensor>("Out");

    int dim = ctx.Attr<int>("dim");
    int power_iters = ctx.Attr<int>("power_iters");
    float eps = ctx.Attr<float>("eps");

    const int h = u->dims()[0];
    const int w = v->dims()[0];

    Tensor weight_mat;
    auto dims = weight->dims();
    const int rank = dims.size();
    std::vector<int> real_dims;
    if (dim != 0) {
        std::vector<int> perm;
        perm.push_back(dim);
        real_dims.push_back(dims[dim]);
        for (int i = 0; i < rank; i++) {
        if (i != dim) {
            perm.push_back(i);
            real_dims.push_back(dims[i]);
        }
        }
        weight_mat.mutable_data<T>(phi::make_ddim(real_dims), ctx.GetPlace());
        TransCompute<DeviceContext, T>(rank, *weight, &weight_mat, perm, dev_ctx);
    } else {
        for (int i = 0; i < rank; i++) {
        real_dims.push_back(i);
        }
        paddle::framework::TensorCopySync(*weight, ctx.GetPlace(), &weight_mat);
    }
    weight_mat = weight_mat.Resize({h, w});

    Tensor sigma;
    sigma.mutable_data<T>(weight_mat.dims(), ctx.GetPlace());
    Tensor uu, vv;
    paddle::framework::TensorCopySync(*u, ctx.GetPlace(), &uu);
    paddle::framework::TensorCopySync(*v, ctx.GetPlace(), &vv);
    CalcMatrixSigmaAndNormWeight<DeviceContext, T>(&sigma,
                                                    &(uu.Resize({h, 1})),
                                                    &(vv.Resize({w, 1})),
                                                    &weight_mat,
                                                    power_iters,
                                                    eps,
                                                    ctx);

    if (dim != 0) {
        std::vector<int> perm;
        for (int i = 0; i < rank; i++) {
        if (i < dim) {
            perm.push_back(i + 1);
        } else if (i == dim) {
            perm.push_back(0);
        } else {
            perm.push_back(i);
        }
        }
        out->mutable_data<T>(dims, ctx.GetPlace());
        TransCompute<DeviceContext, T>(
            rank,
            weight_mat.Resize(phi::make_ddim(real_dims)),
            out,
            perm,
            dev_ctx);
    } else {
        paddle::framework::TensorCopySync(
            weight_mat.Resize(dims), ctx.GetPlace(), out);
    }
}

}  // namespace phi
