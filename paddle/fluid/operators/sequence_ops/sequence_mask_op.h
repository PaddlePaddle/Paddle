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

#ifdef __NVCC__
#include <thrust/device_ptr.h>
#include <thrust/functional.h>
#include <thrust/reduce.h>
#else
#include <algorithm>
#endif

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/for_range.h"

namespace paddle {
namespace operators {

class SequenceMaskOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"), "Input(X) must exist");
    PADDLE_ENFORCE(ctx->HasOutput("Y"), "Output(Y) must exist");

    int maxlen = ctx->Attrs().Get<int>("maxlen");
    auto dim = framework::vectorize2int(ctx->GetInputDim("X"));
    dim.push_back(maxlen > 0 ? maxlen : -1);
    ctx->SetOutputDim("Y", framework::make_ddim(dim));
  }
};

class SequenceMaskOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input tensor of sequence_mask op.");
    AddOutput("Y", "The output mask of sequence_mask op.");
    AddAttr<int>("maxlen",
                 "The maximum length of the sequence. If maxlen < 0, maxlen "
                 "= max(Input(X)).")
        .SetDefault(-1)
        .AddCustomChecker([](const int &v) {
          PADDLE_ENFORCE(v < 0 || v >= 1,
                         "Attr(maxlen) must be less than 0 or larger than 1");
        });
    AddAttr<int>("out_dtype", "Output data type");
    AddComment(R"DOC(
SequenceMask Operator

This operator outputs a Mask according to Input(X) and Attr(maxlen).
Supposing Input(X) is a Tensor with shape [d_1, d_2, ..., d_n], the
Output(Y) is a mask with shape [d_1, d_2, ..., d_n, maxlen], where:

Y(i_1, i_2, ..., i_n, j) = (j < X(i_1, i_2, ..., i_n)) 

If maxlen < 0, maxlen = max(X)
    )DOC");
  }
};

template <typename Tx, typename Ty>
struct SequenceMaskForRangeFunctor {
  HOSTDEVICE SequenceMaskForRangeFunctor(const Tx *x, Ty *y, int maxlen)
      : x_(x), y_(y), maxlen_(maxlen) {}

  HOSTDEVICE void operator()(int y_idx) const {
    int x_idx = y_idx / maxlen_;
    int j = y_idx % maxlen_;
    y_[y_idx] = static_cast<Ty>(j < x_[x_idx] ? 1 : 0);
  }

 private:
  const Tx *x_;
  Ty *y_;
  int maxlen_;
};

template <typename DeviceContext, typename Tx>
struct SequenceMaskFunctor {
  using Tensor = framework::LoDTensor;

  SequenceMaskFunctor(const DeviceContext &ctx, const Tx *x, Tensor *y,
                      int limits, int maxlen)
      : ctx_(ctx), x_(x), y_(y), limits_(limits), maxlen_(maxlen) {}

  template <typename Ty>
  void apply() const {
    auto *y_data = y_->mutable_data<Ty>(ctx_.GetPlace());
    platform::ForRange<DeviceContext> for_range(ctx_, limits_);
    for_range(SequenceMaskForRangeFunctor<Tx, Ty>(x_, y_data, maxlen_));
  }

 private:
  const DeviceContext &ctx_;
  const Tx *x_;
  Tensor *y_;
  int limits_;
  int maxlen_;
};

template <typename DeviceContext, typename Tx>
class SequenceMaskKernel : public framework::OpKernel<Tx> {
  using Tensor = framework::LoDTensor;

 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    auto *x = ctx.Input<Tensor>("X");
    auto *y = ctx.Output<Tensor>("Y");
    auto maxlen = ctx.Attr<int>("maxlen");

    auto *x_data = x->data<Tx>();
    auto x_numel = x->numel();
    if (maxlen < 0) {
#ifdef __NVCC__
      VLOG(10)
          << "SequenceMaskOp on GPU may be slow when maxlen is not provided.";
      maxlen = static_cast<int>(
          thrust::reduce(thrust::device_pointer_cast(x_data),
                         thrust::device_pointer_cast(x_data) + x_numel,
                         static_cast<Tx>(0), thrust::maximum<Tx>()));
#else
      maxlen = static_cast<int>(*std::max_element(x_data, x_data + x_numel));
#endif
      auto y_dim = framework::vectorize2int(x->dims());
      y_dim.push_back(maxlen);
      y->Resize(framework::make_ddim(y_dim));
    }

    auto out_dtype = static_cast<framework::proto::VarType::Type>(
        ctx.Attr<int>("out_dtype"));
    auto &dev_ctx = ctx.template device_context<DeviceContext>();
    framework::VisitDataType(out_dtype,
                             SequenceMaskFunctor<DeviceContext, Tx>(
                                 dev_ctx, x_data, y, x_numel * maxlen, maxlen));
  }
};

}  // namespace operators
}  // namespace paddle
