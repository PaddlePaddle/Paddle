// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/operators/math/seq2col.h"
#include "paddle/fluid/operators/transpose_op.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename DeviceContext, typename T>
struct OverlapAddFunctor {
  void operator()(const DeviceContext& dev_ctx, const Tensor* input,
                  Tensor* output, size_t seq_length, size_t frame_length,
                  size_t n_frames, size_t hop_length,
                  bool is_grad = false) const {
    auto numel = output->numel();
    const auto* input_data = input->data<T>();
    auto* output_data = output->data<T>();

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    if (!is_grad) {
      math::Col2SeqFunctor<T> functor(input_data, output_data, seq_length,
                                      frame_length, n_frames, hop_length);
      for_range(functor);
    } else {
      math::Seq2ColFunctor<T> functor(input_data, output_data, seq_length,
                                      frame_length, n_frames, hop_length);
      for_range(functor);
    }
  }
};

template <typename DeviceContext, typename T>
class OverlapAddKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    const size_t x_rank = x->dims().size();
    const size_t out_rank = out->dims().size();

    const int hop_length = ctx.Attr<int>("hop_length");
    const int axis = ctx.Attr<int>("axis");
    const int n_frames = (axis == 0) ? x->dims()[0] : x->dims()[x_rank - 1];
    const int frame_length = (axis == 0) ? x->dims()[1] : x->dims()[x_rank - 2];
    const int seq_length =
        (axis == 0) ? out->dims()[0] : out->dims()[out_rank - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    Tensor x_(x->type());
    x_ = *x;

    framework::DDim preserved_dims;
    if (out_rank > 2) {
      // Save dims used to flatten both input and output tensors and restore
      // output tensor.
      framework::DDim x_resized_dims;
      framework::DDim out_resized_dims;
      if (axis == 0) {
        preserved_dims = phi::slice_ddim(out->dims(), 1, out_rank);
        x_resized_dims = {n_frames, frame_length, phi::product(preserved_dims)};
        out_resized_dims = {seq_length, phi::product(preserved_dims)};
      } else {
        preserved_dims = phi::slice_ddim(out->dims(), 0, out_rank - 1);
        x_resized_dims = {phi::product(preserved_dims), frame_length, n_frames};
        out_resized_dims = {phi::product(preserved_dims), seq_length};
      }
      x_.Resize(x_resized_dims);
      out->Resize(out_resized_dims);
    }

    Tensor trans_x(x_.type());
    Tensor trans_out(out->type());

    // Transpose input and output in case that axis is 0.
    if (axis == 0) {
      if (out_rank == 1U) {
        trans_out = *out;

        std::vector<int> perm_x{1, 0};
        auto x_dims_vec = phi::vectorize(x_.dims());
        for (int i = 0; i < x_.dims().size(); ++i) {
          x_dims_vec[i] = x_.dims()[perm_x[i]];
        }
        trans_x.Resize(phi::make_ddim(x_dims_vec));
        trans_x.mutable_data<T>(ctx.GetPlace());
        TransCompute<DeviceContext, T>(perm_x.size(), dev_ctx, x_, &trans_x,
                                       perm_x);
      } else {
        std::vector<int> perm_out{1, 0};
        auto out_dims_vec = phi::vectorize(out->dims());
        for (int i = 0; i < out->dims().size(); ++i) {
          out_dims_vec[i] = out->dims()[perm_out[i]];
        }
        trans_out.Resize(phi::make_ddim(out_dims_vec));
        trans_out.mutable_data<T>(ctx.GetPlace());
        TransCompute<DeviceContext, T>(perm_out.size(), dev_ctx, *out,
                                       &trans_out, perm_out);

        std::vector<int> perm_x{2, 1, 0};
        auto x_dims_vec = phi::vectorize(x_.dims());
        for (int i = 0; i < x_.dims().size(); ++i) {
          x_dims_vec[i] = x_.dims()[perm_x[i]];
        }
        trans_x.Resize(phi::make_ddim(x_dims_vec));
        trans_x.mutable_data<T>(ctx.GetPlace());
        TransCompute<DeviceContext, T>(perm_x.size(), dev_ctx, x_, &trans_x,
                                       perm_x);
      }
    } else {
      trans_x = x_;
      trans_out = *out;
    }

    OverlapAddFunctor<DeviceContext, T>()(dev_ctx, &trans_x, &trans_out,
                                          seq_length, frame_length, n_frames,
                                          hop_length, /*is_grad*/ false);

    // Transpose output in case axis is 0.
    if (axis == 0 && out_rank > 1U) {
      std::vector<int> perm_out{1, 0};
      TransCompute<DeviceContext, T>(perm_out.size(), dev_ctx, trans_out, out,
                                     perm_out);
    }

    // Restore output dims when the number of dims is larger than 2.
    if (out_rank > 2) {
      std::vector<int64_t> restored_out_shape;
      for (int i = 0; i < preserved_dims.size(); i++) {
        restored_out_shape.push_back(preserved_dims[i]);
      }

      if (axis == 0) {
        // (seq_length, ...)
        restored_out_shape.insert(restored_out_shape.begin(), seq_length);
      } else {
        // (..., seq_length)
        restored_out_shape.push_back(seq_length);
      }

      out->Resize(phi::make_ddim(restored_out_shape));
    }
  }
};

template <typename DeviceContext, typename T>
class OverlapAddGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* d_out = ctx.Input<Tensor>(framework::GradVarName("Out"));
    Tensor* d_x = ctx.Output<Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(ctx.GetPlace());
    const size_t d_out_rank = d_out->dims().size();
    const size_t d_x_rank = d_x->dims().size();

    const int hop_length = ctx.Attr<int>("hop_length");
    const int axis = ctx.Attr<int>("axis");
    const int n_frames =
        (axis == 0) ? d_x->dims()[0] : d_x->dims()[d_x_rank - 1];
    const int frame_length =
        (axis == 0) ? d_x->dims()[1] : d_x->dims()[d_x_rank - 2];
    const int seq_length =
        (axis == 0) ? d_out->dims()[0] : d_out->dims()[d_out_rank - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // When the number of input dims is larger than 2, it needs to copy
    // from x to resize input into 2d and output into 3d. Morevoer, output
    // dims will be restored at the last step.
    Tensor d_out_(d_out->type());
    d_out_ = *d_out;

    framework::DDim preserved_dims;
    if (d_out_rank > 2) {
      // Save dims used to flatten both input and output tensors and restore
      // output tensor.
      framework::DDim d_x_resized_dims;
      framework::DDim d_out_resized_dims;
      if (axis == 0) {
        preserved_dims = phi::slice_ddim(d_out_.dims(), 1, d_out_rank);
        d_x_resized_dims = {n_frames, frame_length,
                            phi::product(preserved_dims)};
        d_out_resized_dims = {seq_length, phi::product(preserved_dims)};
      } else {
        preserved_dims = phi::slice_ddim(d_out_.dims(), 0, d_out_rank - 1);
        d_x_resized_dims = {phi::product(preserved_dims), frame_length,
                            n_frames};
        d_out_resized_dims = {phi::product(preserved_dims), seq_length};
      }
      d_x->Resize(d_x_resized_dims);
      d_out_.Resize(d_out_resized_dims);
    }

    Tensor trans_d_x(d_x->type());
    Tensor trans_d_out(d_out_.type());

    // Transpose input and output in case that axis is 0.
    if (axis == 0) {
      if (d_out_rank == 1U) {
        trans_d_out = d_out_;

        std::vector<int> perm_d_x{1, 0};
        auto d_x_dims_vec = phi::vectorize(d_x->dims());
        for (int i = 0; i < d_x->dims().size(); ++i) {
          d_x_dims_vec[i] = d_x->dims()[perm_d_x[i]];
        }
        trans_d_x.Resize(phi::make_ddim(d_x_dims_vec));
        trans_d_x.mutable_data<T>(ctx.GetPlace());
        TransCompute<DeviceContext, T>(perm_d_x.size(), dev_ctx, *d_x,
                                       &trans_d_x, perm_d_x);
      } else {
        std::vector<int> perm_d_out{1, 0};
        auto d_out_dims_vec = phi::vectorize(d_out_.dims());
        for (int i = 0; i < d_out_.dims().size(); ++i) {
          d_out_dims_vec[i] = d_out_.dims()[perm_d_out[i]];
        }
        trans_d_out.Resize(phi::make_ddim(d_out_dims_vec));
        trans_d_out.mutable_data<T>(ctx.GetPlace());
        TransCompute<DeviceContext, T>(perm_d_out.size(), dev_ctx, d_out_,
                                       &trans_d_out, perm_d_out);

        std::vector<int> perm_d_x{2, 1, 0};
        auto d_x_dims_vec = phi::vectorize(d_x->dims());
        for (int i = 0; i < d_x->dims().size(); ++i) {
          d_x_dims_vec[i] = d_x->dims()[perm_d_x[i]];
        }
        trans_d_x.Resize(phi::make_ddim(d_x_dims_vec));
        trans_d_x.mutable_data<T>(ctx.GetPlace());
        TransCompute<DeviceContext, T>(perm_d_x.size(), dev_ctx, *d_x,
                                       &trans_d_x, perm_d_x);
      }
    } else {
      trans_d_x = *d_x;
      trans_d_out = d_out_;
    }

    OverlapAddFunctor<DeviceContext, T>()(dev_ctx, &trans_d_out, &trans_d_x,
                                          seq_length, frame_length, n_frames,
                                          hop_length,
                                          /*is_grad*/ true);

    // Transpose output in case axis is 0.
    if (axis == 0) {
      if (d_out_rank == 1U) {
        std::vector<int> perm_d_x{1, 0};
        TransCompute<DeviceContext, T>(perm_d_x.size(), dev_ctx, trans_d_x, d_x,
                                       perm_d_x);
      } else {
        std::vector<int> perm_d_x{2, 1, 0};
        TransCompute<DeviceContext, T>(perm_d_x.size(), dev_ctx, trans_d_x, d_x,
                                       perm_d_x);
      }
    }

    // Restore output dims when the number of dims is larger than 2.
    if (d_out_rank > 2) {
      std::vector<int64_t> restored_d_x_shape;
      for (int i = 0; i < preserved_dims.size(); i++) {
        restored_d_x_shape.push_back(preserved_dims[i]);
      }

      if (axis == 0) {
        // (n_frames, frame_length, ...)
        restored_d_x_shape.insert(restored_d_x_shape.begin(), frame_length);
        restored_d_x_shape.insert(restored_d_x_shape.begin(), n_frames);
      } else {
        // (..., frame_length, n_frames)
        restored_d_x_shape.push_back(frame_length);
        restored_d_x_shape.push_back(n_frames);
      }

      d_x->Resize(phi::make_ddim(restored_d_x_shape));
    }
  }
};

}  // namespace operators
}  // namespace paddle
