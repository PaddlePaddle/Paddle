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
  DataMappingFunctor(const T* x, T* out, size_t seq_length, size_t frame_length,
                     size_t n_frames, size_t hop_length)
      : x_(x),
        out_(out),
        seq_length_(seq_length),
        frame_length_(frame_length),
        n_frames_(n_frames),
        hop_length_(hop_length) {}

  /*
    Convert sequences to frames.

    1. Dimension infomation:

       Sequences                   Frames
    (N, seq_length)  ->  (N, frame_length, n_frames)

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
        f = i % (n_frames_ * frame_length_) / n_frames_;
        n = i % (n_frames_ * frame_length_) % n_frames_;
        ```

      d. Replace `sample_idx`, `f` and `n` in the eqations followed.
        ```cpp
        src_idx = sample_idx * seq_length_ + n * hop_length_ + f;
        trg_idx = sample_idx * n_frames_ * frame_length_ + f * n_frames_ + n;
        out_[trg_idx] = x_[src_idx];
        ```

      e. Result can be deduced shown in the function body below.
  */
  HOSTDEVICE void operator()(size_t i) const {
    size_t src_idx;
    size_t trg_idx;
    src_idx = i / (n_frames_ * frame_length_) * seq_length_ +
              i % (n_frames_ * frame_length_) % n_frames_ * hop_length_ +
              i % (n_frames_ * frame_length_) / n_frames_;
    trg_idx = i / (n_frames_ * frame_length_) * n_frames_ * frame_length_ +
              i % (n_frames_ * frame_length_) / n_frames_ * n_frames_ +
              i % (n_frames_ * frame_length_) % n_frames_;
    out_[trg_idx] = x_[src_idx];
  }

  const T* x_;
  T* out_;
  size_t seq_length_;
  size_t frame_length_;
  size_t n_frames_;
  size_t hop_length_;
};

template <typename T>
struct DataMappingGradFunctor {
  DataMappingGradFunctor(const T* d_out, T* d_x, size_t seq_length,
                         size_t frame_length, size_t n_frames,
                         size_t hop_length)
      : d_out_(d_out),
        d_x_(d_x),
        seq_length_(seq_length),
        frame_length_(frame_length),
        n_frames_(n_frames),
        hop_length_(hop_length) {}

  /*
    Accumulate output gradient d_out to d_x.

    1. Dimension infomation:

              d_out                        d_x
    (N, frame_length, n_frames)  ->  (N, seq_length)

    2. Using a sliding window to find source indices from `d_out` according to
       `i`:

      a. Notion
        - `i` stands for the flattened index of `d_x`.
        - `seq_i` stands for a relative index of a `d_x` sample.
        - `left`: Starting index of a frame window.
        - `right`: Ending index of a frame window.

      b. Sample idx
        ```cpp
        sample_idx = i / seq_length_;
        ```

      c. Slides a window with length of `frame_length` to find `f` and `n`.
        - `n`: The idx of num_frames_, increases in each hop.
        - `f`: The idx of frame_lengths_, relative idx from left of a sliding
               window.

      d. Accumulate all grads from d_out.
        ```cpp
        d_x_[i] +=
            d_out_[sample_idx * frame_length_ * n_frames_ + f * n_frames_ + n];
        ```
  */
  HOSTDEVICE void operator()(size_t i) const {
    size_t sample_idx = i / seq_length_;
    size_t seq_i = i % seq_length_;

    // Sliding window
    d_x_[i] = 0;  // Init d_x_[i] to 0, and a while loop followed to sum up all
                  // grads from d_out_.

    size_t n = get_start_frame_idx(seq_i);
    size_t f;
    size_t left = n * hop_length_;
    size_t right = left + frame_length_ - 1;

    while (left <= seq_i && right < seq_length_) {
      f = seq_i - left;
      d_x_[i] +=
          d_out_[sample_idx * frame_length_ * n_frames_ + f * n_frames_ + n];
      // Next frame.
      left += hop_length_;
      right += hop_length_;
      n += 1;
    }
  }

  /*
    Calculate minimum value of frame index `n` to satisfy the inequality:

      seq_i <= right
      ==> seq_i <= left + frame_length - 1
      ==> seq_i <= hop_length_ * n + frame_length_ - 1
  */
  HOSTDEVICE size_t get_start_frame_idx(size_t seq_i) const {
    int64_t tmp = seq_i + 1 - frame_length_;
    if (tmp > 0) {
      size_t n = tmp / hop_length_;
      if (tmp % hop_length_ == 0) {
        return n;
      } else {
        return n + 1;
      }
    } else {
      return 0;
    }
  }

  const T* d_out_;
  T* d_x_;
  size_t seq_length_;
  size_t frame_length_;
  size_t n_frames_;
  size_t hop_length_;
};

template <typename DeviceContext, typename T>
struct FrameFunctor {
  void operator()(const DeviceContext& dev_ctx, const Tensor* input,
                  Tensor* output, size_t seq_length, size_t frame_length,
                  size_t n_frames, size_t hop_length,
                  bool is_grad = false) const {
    auto numel = output->numel();
    auto* input_data = input->data<T>();
    auto* output_data = output->data<T>();

    platform::ForRange<DeviceContext> for_range(dev_ctx, numel);
    if (!is_grad) {
      DataMappingFunctor<T> functor(input_data, output_data, seq_length,
                                    frame_length, n_frames, hop_length);
      for_range(functor);
    } else {
      DataMappingGradFunctor<T> functor(input_data, output_data, seq_length,
                                        frame_length, n_frames, hop_length);
      for_range(functor);
    }
  }
};

template <typename DeviceContext, typename T>
static inline void TransCompute(const framework::ExecutionContext& ctx,
                                const Tensor& x, Tensor* out,
                                const std::vector<int>& perm) {
  int rank = x.dims().size();
  PADDLE_ENFORCE_EQ((rank == 2 || rank == 3), true,
                    platform::errors::InvalidArgument(
                        "Rank of x should be 2 or 3, but got %d.", rank));

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
    Frame kernel slice frames from input sequences. The main steps as follow:

      - Case 1 - input dims == 1:
        - axis is -1: Call a FrameFunctor to compute directly.
        - axis is  0: Transpose output firstly, and then it falls into
                      case axis is -1. Finally, it restores the dims of
                      output tensor.

      - Case 2 - input dims == 2:
        - axis is -1: Call a FrameFunctor to compute directly.
        - axis is  0: Transpose both input and output firstly, and then it falls
                      into case axis is -1. Finally, it restores the dims of
                      output tensor.

      - Case 3 - input dims > 2:
        Flatten the input and output to 2D and 3D respectively so that it
        falls into Case 2. Finally, it restores the dims of output tensor.
  */
  void Compute(const framework::ExecutionContext& ctx) const override {
    const Tensor* x = ctx.Input<Tensor>("X");
    Tensor* out = ctx.Output<Tensor>("Out");
    out->mutable_data<T>(ctx.GetPlace());
    const size_t x_rank = x->dims().size();
    const size_t out_rank = out->dims().size();

    const int frame_length = ctx.Attr<int>("frame_length");
    const int hop_length = ctx.Attr<int>("hop_length");
    const int axis = ctx.Attr<int>("axis");
    const int n_frames =
        (axis == 0) ? out->dims()[0] : out->dims()[out_rank - 1];
    const int seq_length = (axis == 0) ? x->dims()[0] : x->dims()[x_rank - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    // When the number of input dims is larger than 2, it needs to copy
    // from x to resize input into 2d and output into 3d. Morevoer, output
    // dims will be restored at the last step.
    Tensor x_(x->type());
    x_ = *x;

    framework::DDim preserved_dims;
    if (x_rank > 2) {
      // Save dims used to flatten both input and output tensors and restore
      // output tensor.
      framework::DDim x_resized_dims;
      framework::DDim out_resized_dims;
      if (axis == 0) {
        preserved_dims = framework::slice_ddim(x_.dims(), 1, x_rank);
        x_resized_dims = {seq_length, framework::product(preserved_dims)};
        out_resized_dims = {n_frames, frame_length,
                            framework::product(preserved_dims)};
      } else {
        preserved_dims = framework::slice_ddim(x_.dims(), 0, x_rank - 1);
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
      if (x_rank == 1U) {
        trans_x = x_;
        std::vector<int> perm_out{1, 0};
        TransCompute<DeviceContext, T>(ctx, *out, &trans_out, perm_out);
      } else {
        std::vector<int> perm_x{1, 0};
        TransCompute<DeviceContext, T>(ctx, x_, &trans_x, perm_x);
        std::vector<int> perm_out{2, 1, 0};
        TransCompute<DeviceContext, T>(ctx, *out, &trans_out, perm_out);
      }
    } else {
      trans_x = x_;
      trans_out = *out;
    }

    FrameFunctor<DeviceContext, T>()(dev_ctx, &trans_x, &trans_out, seq_length,
                                     frame_length, n_frames, hop_length,
                                     /*is_grad*/ false);

    // Transpose output in case axis is 0.
    if (axis == 0) {
      if (x_rank == 1U) {
        std::vector<int> perm_out{1, 0};
        TransCompute<DeviceContext, T>(ctx, trans_out, out, perm_out);
      } else {
        std::vector<int> perm_out{2, 1, 0};
        TransCompute<DeviceContext, T>(ctx, trans_out, out, perm_out);
      }
    }

    // Restore output dims when the number of dims is larger than 2.
    if (x_rank > 2) {
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
};

template <typename DeviceContext, typename T>
class FrameGradKernel : public framework::OpKernel<T> {
 public:
  /*
    Frame gradient kernel accumulate gradient `d_x` from `d_out`. The
    main steps as follow:

      - Case 1 - d_x dims == 1:
        - axis is -1: Call a FrameFunctor to compute directly. Notes that
                      `is_grad` is set to true to select gradient data functor.
        - axis is  0: Transpose `d_out` firstly, and then it falls into
                      case axis is -1.

      - Case 2 - d_x dims == 2:
        - axis is -1: Call a FrameFunctor to compute directly.
        - axis is  0: Transpose both `d_x` and `d_out` firstly, and then it
                      falls into case axis is -1. Finally, it restores the
                      dims of `d_x`.

      - Case 3 - d_x dims > 2:
        Flatten the `d_x` and `d_out` to 2D and 3D respectively so that it
        falls into Case 2. Finally, it restores the dims of `d_x` tensor.
  */
  void Compute(const framework::ExecutionContext& ctx) const {
    const framework::Tensor* d_out =
        ctx.Input<framework::Tensor>(framework::GradVarName("Out"));
    framework::Tensor* d_x =
        ctx.Output<framework::Tensor>(framework::GradVarName("X"));
    d_x->mutable_data<T>(ctx.GetPlace());
    const size_t d_out_rank = d_out->dims().size();
    const size_t d_x_rank = d_x->dims().size();

    const int frame_length = ctx.Attr<int>("frame_length");
    const int hop_length = ctx.Attr<int>("hop_length");
    const int axis = ctx.Attr<int>("axis");
    const int n_frames =
        (axis == 0) ? d_out->dims()[0] : d_out->dims()[d_out_rank - 1];
    const int seq_length =
        (axis == 0) ? d_x->dims()[0] : d_x->dims()[d_x_rank - 1];

    auto& dev_ctx = ctx.device_context<DeviceContext>();

    Tensor d_out_(d_out->type());
    d_out_ = *d_out;

    framework::DDim preserved_dims;
    if (d_x_rank > 2) {
      // Save dims used to flatten both input and output tensors and restore
      // output tensor.
      framework::DDim d_x_resized_dims;
      framework::DDim d_out_resized_dims;
      if (axis == 0) {
        preserved_dims = framework::slice_ddim(d_x->dims(), 1, d_x_rank);
        d_x_resized_dims = {seq_length, framework::product(preserved_dims)};
        d_out_resized_dims = {n_frames, frame_length,
                              framework::product(preserved_dims)};
      } else {
        preserved_dims = framework::slice_ddim(d_x->dims(), 0, d_x_rank - 1);
        d_x_resized_dims = {framework::product(preserved_dims), seq_length};
        d_out_resized_dims = {framework::product(preserved_dims), frame_length,
                              n_frames};
      }
      d_x->Resize(d_x_resized_dims);
      d_out_.Resize(d_out_resized_dims);
    }

    Tensor trans_d_x(d_x->type());
    Tensor trans_d_out(d_out_.type());

    // Transpose input and output in case that axis is 0.
    if (axis == 0) {
      if (d_x_rank == 1U) {
        trans_d_x = *d_x;
        std::vector<int> perm_d_out{1, 0};
        TransCompute<DeviceContext, T>(ctx, d_out_, &trans_d_out, perm_d_out);
      } else {
        std::vector<int> perm_d_x{1, 0};
        TransCompute<DeviceContext, T>(ctx, *d_x, &trans_d_x, perm_d_x);
        std::vector<int> perm_d_out{2, 1, 0};
        TransCompute<DeviceContext, T>(ctx, d_out_, &trans_d_out, perm_d_out);
      }
    } else {
      trans_d_x = *d_x;
      trans_d_out = d_out_;
    }

    FrameFunctor<DeviceContext, T>()(dev_ctx, &trans_d_out, &trans_d_x,
                                     seq_length, frame_length, n_frames,
                                     hop_length,
                                     /*is_grad*/ true);

    // Transpose output in case axis is 0.
    if (axis == 0 && d_x_rank > 1U) {
      std::vector<int> perm_d_x{1, 0};
      TransCompute<DeviceContext, T>(ctx, trans_d_x, d_x, perm_d_x);
    }

    // Restore output dims when the number of dims is larger than 2.
    if (d_x_rank > 2) {
      std::vector<int64_t> restored_d_x_shape;
      for (int i = 0; i < preserved_dims.size(); i++) {
        restored_d_x_shape.push_back(preserved_dims[i]);
      }

      if (axis == 0) {
        // (seq_length, ...)
        restored_d_x_shape.insert(restored_d_x_shape.begin(), seq_length);
      } else {
        // (..., seq_length)
        restored_d_x_shape.push_back(seq_length);
      }

      d_x->Resize(framework::make_ddim(restored_d_x_shape));
    }
  }
};

}  // namespace operators
}  // namespace paddle
