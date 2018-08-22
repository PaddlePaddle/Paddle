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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

class SequenceMaskOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must exist");
    auto max_len = ctx->Attrs().Get<int>("max_len");
    PADDLE_ENFORCE_GT(max_len, 1, "Attr(max_len) must be larger than 1");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist");
    auto dim = framework::vectorize2int(ctx->GetInputDim("X"));
    dim.push_back(max_len);
    ctx->SetOutputDim("Y", framework::make_ddim(dim));
  }
};

class SequenceMaskOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input of sequence_mask op.");
    AddOutput("Y", "The output mask of sequence_mask op.");
    AddAttr<int>("max_len", "The maximum length of the sequence.")
        .GreaterThan(1);
    AddAttr<int>("out_dtype", "Output data type");
    AddComment(R"DOC(
SequenceMask Operator

This operator outputs a Mask according to Input(X) and Attr(max_len).
Supposing Input(X) is a Tensor with shape [d_1, d_2, ..., d_n], the
Output(Y) is a mask with shape [d_1, d_2, ..., d_n, max_len], where:

Y(i_1, i_2, ..., i_n, j) = (j < X(i_1, i_2, ..., i_n)) 
    )DOC");
  }
};

template <typename Tx, typename Ty>
struct SequenceMaskForRangeFunctor {
  HOSTDEVICE SequenceMaskForRangeFunctor(const Tx *x, Ty *y, int max_len)
      : x_(x), y_(y), max_len_(max_len) {}

  HOSTDEVICE void operator()(int y_idx) const {
    int x_idx = y_idx / max_len_;
    int j = y_idx % max_len_;
    y_[y_idx] = static_cast<Ty>(j < x_[x_idx] ? 1 : 0);
  }

 private:
  const Tx *x_;
  Ty *y_;
  int max_len_;
};

template <typename DeviceContext, typename Tx>
struct SequenceMaskFunctor {
  using Tensor = framework::LoDTensor;

  SequenceMaskFunctor(const DeviceContext &ctx, const Tx *x, Tensor *y,
                      int limits, int max_len)
      : ctx_(ctx), x_(x), y_(y), limits_(limits), max_len_(max_len) {}

  template <typename Ty>
  void operator()() const {
    auto *y_data = y_->mutable_data<Ty>(ctx_.GetPlace());
    platform::ForRange<DeviceContext> for_range(ctx_, limits_);
    for_range(SequenceMaskForRangeFunctor<Tx, Ty>(x_, y_data, max_len_));
  }

 private:
  const DeviceContext &ctx_;
  const Tx *x_;
  Tensor *y_;
  int limits_;
  int max_len_;
};

template <typename DeviceContext, typename Tx>
class SequenceMaskKernel : public framework::OpKernel<Tx> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Y");
    auto max_len = ctx.Attr<int>("max_len");
    auto out_dtype = static_cast<framework::proto::VarType::Type>(
        ctx.Attr<int>("out_dtype"));
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    framework::VisitDataType(out_dtype, SequenceMaskFunctor<DeviceContext, Tx>(
                                            dev_ctx, x->data<Tx>(), y,
                                            x->numel() * max_len, max_len));
  }
};

}  // namespace operators
}  // namespace paddle
