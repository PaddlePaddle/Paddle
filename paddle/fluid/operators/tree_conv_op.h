// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/tree2col.h"
#include "paddle/phi/kernels/funcs/blas/blas.h"

namespace paddle {
namespace operators {
<<<<<<< HEAD
=======
using Tensor = framework::Tensor;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
using DDim = framework::DDim;
template <typename DeviceContext, typename T>
class TreeConvKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    math::Tree2ColFunctor<DeviceContext, T> tree2col;
    phi::funcs::SetConstant<DeviceContext, T> constant;

<<<<<<< HEAD
    auto *Edges = ctx.Input<phi::DenseTensor>("EdgeSet");
    auto *Embeddings = ctx.Input<phi::DenseTensor>("NodesVector");
    auto *Filter = ctx.Input<phi::DenseTensor>("Filter");
    auto *output_emb = ctx.Output<phi::DenseTensor>("Out");
=======
    auto *Edges = ctx.Input<Tensor>("EdgeSet");
    auto *Embeddings = ctx.Input<Tensor>("NodesVector");
    auto *Filter = ctx.Input<Tensor>("Filter");
    auto *output_emb = ctx.Output<Tensor>("Out");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    int max_depth = ctx.Attr<int>("max_depth");

    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);

<<<<<<< HEAD
    phi::DenseTensor W;
=======
    Tensor W;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    W.ShareDataWith(*Filter);
    W.Resize(phi::flatten_to_2d(Filter->dims(), 2));

    int batch_size = static_cast<int>(Edges->dims()[0]);
    int n = static_cast<int>(Embeddings->dims()[1]);
    int out_size = static_cast<int>(Filter->dims()[2]);
    int num_filters = static_cast<int>(Filter->dims()[3]);
    output_emb->mutable_data<T>({batch_size, n, out_size, num_filters},
                                ctx.GetPlace());

    auto edge_set_slicedim = phi::slice_ddim(
        Edges->dims(), 1, static_cast<int>(Edges->dims().size()));

    auto embedding_slicedim = phi::slice_ddim(
        Embeddings->dims(), 1, static_cast<int>(Embeddings->dims().size()));

    auto output_slicedim = phi::slice_ddim(
        output_emb->dims(), 1, static_cast<int>(output_emb->dims().size()));

    output_slicedim = phi::flatten_to_2d(output_slicedim, 1);

    for (int idx = 0; idx < batch_size; idx++) {
      auto edge_set = Edges->Slice(idx, idx + 1).Resize(edge_set_slicedim);
      auto embeddings =
          Embeddings->Slice(idx, idx + 1).Resize(embedding_slicedim);
      auto out_vec = output_emb->Slice(idx, idx + 1).Resize(output_slicedim);
<<<<<<< HEAD
      phi::DenseTensor patch;
=======
      Tensor patch;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      tree2col(dev_ctx, edge_set, embeddings, &patch, max_depth);
      constant(dev_ctx, &out_vec, 0);
      blas.MatMul(patch, W, &out_vec);
    }
  }
};
template <typename DeviceContext, typename T>
class TreeConvGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
<<<<<<< HEAD
    auto *out_g = ctx.Input<phi::DenseTensor>(framework::GradVarName("Out"));
    auto *in_g =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("NodesVector"));
    auto *filter_g =
        ctx.Output<phi::DenseTensor>(framework::GradVarName("Filter"));
    int max_depth = ctx.Attr<int>("max_depth");
    auto *Embeddings = ctx.Input<phi::DenseTensor>("NodesVector");
    auto *edges = ctx.Input<phi::DenseTensor>("EdgeSet");
    auto *Filter = ctx.Input<phi::DenseTensor>("Filter");
=======
    auto *out_g = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto *in_g = ctx.Output<Tensor>(framework::GradVarName("NodesVector"));
    auto *filter_g = ctx.Output<Tensor>(framework::GradVarName("Filter"));
    int max_depth = ctx.Attr<int>("max_depth");
    auto *Embeddings = ctx.Input<Tensor>("NodesVector");
    auto *edges = ctx.Input<Tensor>("EdgeSet");
    auto *Filter = ctx.Input<Tensor>("Filter");
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    math::Tree2ColFunctor<DeviceContext, T> tree2col;
    math::Col2TreeFunctor<DeviceContext, T> col2tree;
    phi::funcs::SetConstant<DeviceContext, T> constant;
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    auto blas = phi::funcs::GetBlas<DeviceContext, T>(dev_ctx);

<<<<<<< HEAD
    phi::DenseTensor W;
=======
    Tensor W;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
    W.ShareDataWith(*Filter);
    W.Resize(phi::flatten_to_2d(Filter->dims(), 1));

    int batch_size = static_cast<int>(Embeddings->dims()[0]);

    auto edge_set_slicedim = phi::slice_ddim(
        edges->dims(), 1, static_cast<int>(edges->dims().size()));

    auto embedding_slicedim = phi::slice_ddim(
        Embeddings->dims(), 1, static_cast<int>(Embeddings->dims().size()));

    auto out_grad_dims = phi::slice_ddim(
        out_g->dims(), 1, static_cast<int>(out_g->dims().size()));
    out_grad_dims = phi::flatten_to_2d(out_grad_dims, 1);
    if (filter_g) {
      filter_g->mutable_data<T>(Filter->dims(), ctx.GetPlace());
<<<<<<< HEAD
      phi::DenseTensor f_g;
=======
      Tensor f_g;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
      f_g.ShareDataWith(*filter_g);
      f_g.Resize(phi::flatten_to_2d(Filter->dims(), 2));
      constant(dev_ctx, filter_g, 0);
      for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto edge_set =
            edges->Slice(batch_id, batch_id + 1).Resize(edge_set_slicedim);
        auto embeddings = Embeddings->Slice(batch_id, batch_id + 1)
                              .Resize(embedding_slicedim);
        auto out_grad =
            out_g->Slice(batch_id, batch_id + 1).Resize(out_grad_dims);
<<<<<<< HEAD
        phi::DenseTensor patch;
=======
        Tensor patch;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        tree2col(dev_ctx, edge_set, embeddings, &patch, max_depth);
        blas.MatMul(patch, true, out_grad, false, T(1.0), &f_g, T(1.0));
      }
    }
    if (in_g) {
      auto input_grad_dims = phi::slice_ddim(
          in_g->dims(), 1, static_cast<int>(in_g->dims().size()));
      in_g->mutable_data<T>(Embeddings->dims(), ctx.GetPlace());
      constant(dev_ctx, in_g, 0);
      for (int batch_id = 0; batch_id < batch_size; batch_id++) {
        auto edge_set =
            edges->Slice(batch_id, batch_id + 1).Resize(edge_set_slicedim);
        auto out_grad =
            out_g->Slice(batch_id, batch_id + 1).Resize(out_grad_dims);
        auto in_grad =
            in_g->Slice(batch_id, batch_id + 1).Resize(input_grad_dims);
<<<<<<< HEAD
        phi::DenseTensor in_grad_temp;
=======
        Tensor in_grad_temp;
>>>>>>> 0699afb112355f7e0a08b05030bb7fe613554d81
        col2tree(dev_ctx, edge_set, out_grad, &in_grad_temp, max_depth);
        blas.MatMul(in_grad_temp, false, W, true, &in_grad);
      }
    }
  }
};
}  // namespace operators
}  // namespace paddle
