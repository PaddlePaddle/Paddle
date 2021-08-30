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
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;

template <typename T>
struct DataMappingFunctor {
  DataMappingFunctor(const T* input, T* output, size_t seq_length,
                     size_t frame_length, size_t n_frames, size_t hop_length,
                     int axis)
      : input_(input),
        output_(output),
        seq_length_(seq_length),
        frame_length_(frame_length),
        n_frames_(n_frames),
        hop_length_(hop_length),
        axis_(axis) {}

  /*
    Convert sequences to frames.

    1. Dimension infomation:

       Sequences                   Frames
    (seq_length, N)  ->  (n_frames, frame_length, N)  // axis = 0
    (N, seq_length)  ->  (N, frame_length, n_frames)  // axis = -1

    2. Mapping from `i` to  `src_idx` and `trg_idx` can be derived from:

      a. Notion
        - `i` stands for the flattened index of a bunch of frames.
        - `src_idx` and `trg_idx` are the 1D indices of seqs and frames
          respectivly.

      b. Sample idx
        ```cpp
        sample_idx = i / (n_frames_ * frame_length_);
        ```

      c. Maps `i` to `f` and `n`.
        ```cpp
        f = (axis_ == 0) ? i % (n_frames_ * frame_length_) % frame_length_ :
                           i % (n_frames_ * frame_length_) / n_frames_;
        n = (axis_ == 0) ? i % (n_frames_ * frame_length_) / frame_length_ :
                           i % (n_frames_ * frame_length_) % n_frames_;
        ```

      d. Replace `sample_idx`, `f` and `n` in the eqations followed.
        ```cpp
        src_idx = sample_idx * seq_length_ + n * hop_length_ + f;
        trg_idx =
            (axis_ == 0) ? sample_idx * n_frames_ * frame_length_ +
                           n * frame_length_ + f :
                           sample_idx * n_frames_ * frame_length_ +
                           f * n_frames_ + n;
        output_[trg_idx] = input_[src_idx];
        ```

      e. Result can be deduced shown in the function body below that differs
         from axis_.
  */
  HOSTDEVICE void operator()(size_t i) const {
    size_t src_idx;
    size_t trg_idx;
    if (axis_ == 0) {
      src_idx = i / (n_frames_ * frame_length_) * seq_length_ +
                i % (n_frames_ * frame_length_) / frame_length_ * hop_length_ +
                i % (n_frames_ * frame_length_) % frame_length_;
      trg_idx =
          i / (n_frames_ * frame_length_) * n_frames_ * frame_length_ +
          i % (n_frames_ * frame_length_) / frame_length_ * frame_length_ +
          i % (n_frames_ * frame_length_) % frame_length_;
    } else {
      src_idx = i / (n_frames_ * frame_length_) * seq_length_ +
                i % (n_frames_ * frame_length_) % n_frames_ * hop_length_ +
                i % (n_frames_ * frame_length_) / n_frames_;
      trg_idx = i / (n_frames_ * frame_length_) * n_frames_ * frame_length_ +
                i % (n_frames_ * frame_length_) / n_frames_ * n_frames_ +
                i % (n_frames_ * frame_length_) % n_frames_;
    }
    output_[trg_idx] = input_[src_idx];
  }

  const T* input_;
  T* output_;
  size_t seq_length_;
  size_t frame_length_;
  size_t n_frames_;
  size_t hop_length_;
  int axis_;
};

template <typename DeviceContext, typename T>
struct FrameFunctor {
  void operator()(const DeviceContext& dev_ctx, const Tensor* input,
                  Tensor* output, size_t seq_length, size_t frame_length,
                  size_t n_frames, size_t hop_length, int axis) const {
    auto numel = output->numel();
    auto* input_data = input->data<T>();
    auto* output_data = output->data<T>();

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    DataMappingFunctor<T> functor(input_data, output_data, seq_length,
                                  frame_length, n_frames, hop_length, axis);
    for_range(functor);
  }
};

template <typename DeviceContext, typename T>
static inline void TransCompute(const framework::ExecutionContext& ctx,
                                const Tensor& x, Tensor* out,
                                const std::vector<int>& perm) {
  int rank = x.dims().size();
  if (rank <= 1 || rank > 3) {
    PADDLE_THROW(paddle::platform::errors::Fatal(
        "Input rank should be 2 or 3, but got %d.", rank));
  }

  if (!out->IsInitialized()) {
    auto dims_vec = framework::vectorize(x.dims());
    for (int i = 0; i < rank; ++i) {
      dims_vec[i] = x.dims()[perm[i]];
    }
    out->Resize(framework::make_ddim(dims_vec));
    out->mutable_data<T>(ctx.GetPlace());
  }

  auto& dev_ctx = ctx.device_context<DeviceContext>();

  switch (rank) {
    case 2:
      math::Transpose<DeviceContext, T, 2> trans2;
      trans2(dev_ctx, x, out, perm);
      break;
    case 3:
      math::Transpose<DeviceContext, T, 3> trans3;
      trans3(dev_ctx, x, out, perm);
      break;
    default:
      break;
  }
}

template <typename DeviceContext, typename T>
class FrameKernel : public framework::OpKernel<T> {
 public:
  /*
    This kernel requires input dims in [1, +∞). The main steps
    as follow:

      1. Case 1 - input dims <= 1:
        Call a FrameFunctor to compute it directly.

      2. Case 2 - input dims == 2:
        - axis is -1: Call a FrameFunctor like Case 1.
        - axis is  0: Transpose both input and output firstly, and then it falls
          into case axis is -1. Finally, it restores the dims of output tensor.

      3. Case 3 - input dims > 2:
        Flatten the input and output dims to 2D and 3D respectively so that it
        falls into Case 2. Finally, it restores the dims of output tensor.
  */
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());

    const int frame_length = ctx.Attr<int>("frame_length");
    const int hop_length = ctx.Attr<int>("hop_length");
    const int axis = ctx.Attr<int>("axis");
    const int n_frames =
        (axis == 0) ? out->dims()[0] : out->dims()[out->dims().size() - 1];
    const int seq_length =
        (axis == 0) ? x->dims()[0] : x->dims()[x->dims().size() - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    if (x->dims().size() == 1U) {
      FrameFunctor<DeviceContext, T>()(dev_ctx, x, out, seq_length,
                                       frame_length, n_frames, hop_length,
                                       axis);
    } else {
      // When the number of input dims is larger than 2, it needs to copy
      // from x to resize input into 2d and output into 3d. Morevoer, output
      // dims will be restored at the last step.
      Tensor x_(x->type());
      x_ = *x;

      framework::DDim preserved_dims;
      if (x->dims().size() > 2) {
        // Save dims used to flatten both input and output tensors and restore
        // output tensor.
        framework::DDim x_resized_dims;
        framework::DDim out_resized_dims;
        if (axis == 0) {
          preserved_dims =
              framework::slice_ddim(x_.dims(), 1, x_.dims().size());
          x_resized_dims = {seq_length, framework::product(preserved_dims)};
          out_resized_dims = {n_frames, frame_length,
                              framework::product(preserved_dims)};
        } else {
          preserved_dims =
              framework::slice_ddim(x_.dims(), 0, x_.dims().size() - 1);
          x_resized_dims = {framework::product(preserved_dims), seq_length};
          out_resized_dims = {framework::product(preserved_dims), frame_length,
                              n_frames};
        }

        x_.Resize(x_resized_dims);
        out->Resize(out_resized_dims);
      }

      Tensor trans_x(x_.type());
      Tensor trans_out(out->type());

      // Transpose input and output in case that axis is 0.
      if (axis == 0) {
        std::vector<int> perm_x{1, 0};
        TransCompute<DeviceContext, T>(ctx, x_, &trans_x, perm_x);

        std::vector<int> perm_out{2, 1, 0};
        TransCompute<DeviceContext, T>(ctx, *out, &trans_out, perm_out);
      } else {
        trans_x = x_;
        trans_out = *out;
      }

      FrameFunctor<DeviceContext, T>()(dev_ctx, &trans_x, &trans_out,
                                       seq_length, frame_length, n_frames,
                                       hop_length, /*axis*/ -1);

      // Transpose output in case axis is 0.
      if (axis == 0) {
        std::vector<int> perm_out{2, 1, 0};
        TransCompute<DeviceContext, T>(ctx, trans_out, out, perm_out);
      }

      // Restore output when the number of dims is larger than 2.
      if (x->dims().size() > 2) {
        std::vector<int64_t> restored_out_shape;
        for (int i = 0; i < preserved_dims.size(); i++) {
          restored_out_shape.push_back(preserved_dims[i]);
        }

        if (axis == 0) {
          // (n_frames, frame_length, ...)
          restored_out_shape.insert(restored_out_shape.begin(), frame_length);
          restored_out_shape.insert(restored_out_shape.begin(), n_frames);
        } else {
          // (..., frame_length, n_frames)
          restored_out_shape.push_back(frame_length);
          restored_out_shape.push_back(n_frames);
        }

        out->Resize(framework::make_ddim(restored_out_shape));
      }
    }
  }
};

template <typename DeviceContext, typename T>
class FrameGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    const framework::Tensor* x = ctx.Input<framework::Tensor>("X");
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    auto& dev_ctx = ctx.device_context<DeviceContext>();

    VLOG(0) << d_out;
    VLOG(0) << x;
    VLOG(0) << d_x;
    VLOG(0) << &dev_ctx;
  }
};

}  // namespace operators
}  // namespace paddle
