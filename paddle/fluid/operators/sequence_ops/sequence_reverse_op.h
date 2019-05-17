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

#include <memory>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/math/algorithm.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

class SequenceReverseOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must exist");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist");

    auto x_dim = ctx->GetInputDim("X");
    PADDLE_ENFORCE_GE(x_dim.size(), 2,
                      "Rank of Input(X) must be not less than 2.");

    ctx->SetOutputDim("Y", x_dim);
    ctx->ShareLoD("X", "Y");
  }
};

class SequenceReverseOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input LoDTensor of sequence_reverse op.");
    AddOutput("Y", "The output LoDTensor of sequence_reverse op.");
    AddComment(R"DOC(
SequenceReverse Operator.

Reverse each sequence in input X along dim 0.

Assuming X is a LoDTensor with dims [5, 4] and lod [[0, 2, 5]], where:

X.data() = [
  [1, 2, 3, 4],
  [5, 6, 7, 8], # the 0-th sequence with length 2
  [9, 10, 11, 12],
  [13, 14, 15, 16],
  [17, 18, 19, 20] # the 1-st sequence with length 3
]

The output Y would be a LoDTensor sharing the same dims and lod with input X,
and:

Y.data() = [
  [5, 6, 7, 8],
  [1, 2, 3, 4], # the reversed 0-th sequence with length 2
  [17, 18, 19, 20],
  [13, 14, 15, 16],
  [9, 10, 11, 12] # the reversed 1-st sequence with length 3
]

This Operator is useful to build a reverse dynamic RNN network.

This Operator only supports one-level lod currently.
    )DOC");
  }
};

template <typename T>
struct SequenceReverseFunctor {
  SequenceReverseFunctor(const T *x, T *y, const size_t *lod, size_t lod_count,
                         size_t row_numel)
      : x_(x), y_(y), lod_(lod), lod_count_(lod_count), row_numel_(row_numel) {}

  HOSTDEVICE void operator()(size_t idx_x) const {
    auto row_idx_x = idx_x / row_numel_;
    auto lod_idx = math::UpperBound(lod_, lod_count_, row_idx_x);
    auto row_idx_y = lod_[lod_idx - 1] + (lod_[lod_idx] - 1 - row_idx_x);
    auto idx_y = row_idx_y * row_numel_ + idx_x % row_numel_;
    y_[idx_y] = x_[idx_x];
  }

  const T *x_;
  T *y_;
  const size_t *lod_;
  size_t lod_count_;
  size_t row_numel_;
};

template <typename DeviceContext, typename T>
class SequenceReverseOpKernel : public framework::OpKernel<T> {
  using LoDTensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto &x = *ctx.Input<LoDTensor>("X");
    auto *y = ctx.Output<LoDTensor>("Y");

    PADDLE_ENFORCE_EQ(x.lod().size(), 1,
                      "SequenceReverse Op only support one level lod.");

    const size_t *lod;
    size_t lod_count = x.lod()[0].size();

#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(ctx.GetPlace())) {
      lod = x.lod()[0].CUDAData(ctx.GetPlace());
    } else {
#endif
      lod = x.lod()[0].data();
#ifdef PADDLE_WITH_CUDA
    }
#endif

    size_t limit = static_cast<size_t>(x.numel());
    size_t row_numel = static_cast<size_t>(limit / x.dims()[0]);
    auto *x_data = x.data<T>();
    auto *y_data = y->mutable_data<T>(ctx.GetPlace());

    PADDLE_ENFORCE_NE(x_data, y_data,
                      "SequenceReverse Op does not support in-place operation");

    if (platform::is_cpu_place(ctx.GetPlace())) {
      for (size_t idx = 0; idx < lod_count - 1; idx++) {
        auto start_pos = lod[idx];
        auto end_pos = lod[idx + 1];
        for (auto pos = start_pos; pos < end_pos; pos++) {
          auto cur_pos = end_pos - pos - 1 + start_pos;
          std::memcpy(y_data + pos * row_numel, x_data + cur_pos * row_numel,
                      row_numel * sizeof(T));
        }
      }
    } else {
      auto &dev_ctx = ctx.template device_context<DeviceContext>();

      SequenceReverseFunctor<T> functor(x_data, y_data, lod, lod_count,
                                        row_numel);
      platform::ForRange<DeviceContext> for_range(dev_ctx, limit);
      for_range(functor);
    }
  }
};

class SequenceReverseGradOpDescMaker : public framework::SingleGradOpDescMaker {
 public:
  using framework::SingleGradOpDescMaker::SingleGradOpDescMaker;

 protected:
  std::unique_ptr<framework::OpDesc> Apply() const override {
    std::unique_ptr<framework::OpDesc> op(new framework::OpDesc());
    op->SetType("sequence_reverse");
    op->SetInput("X", OutputGrad("Y"));
    op->SetOutput("Y", InputGrad("X"));
    op->SetAttrMap(Attrs());
    return op;
  }
};

}  // namespace operators
}  // namespace paddle
