/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/platform/device_context.h"

namespace pten {
class DenseTensor;
}  // namespace pten

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

struct CopyRange {
  size_t begin;
  size_t end;
};

using LoD = framework::LoD;

class SplitLoDTensorOp : public framework::OperatorBase {
 public:
  SplitLoDTensorOp(const std::string &type,
                   const framework::VariableNameMap &inputs,
                   const framework::VariableNameMap &outputs,
                   const framework::AttributeMap &attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &dev_place) const override {
    auto &x = scope.FindVar(Input("X"))->Get<framework::LoDTensor>();
    auto &mask = scope.FindVar(Input("Mask"))->Get<framework::LoDTensor>();
    auto *out_true =
        scope.FindVar(Output("OutTrue"))->GetMutable<framework::LoDTensor>();
    auto *out_false =
        scope.FindVar(Output("OutFalse"))->GetMutable<framework::LoDTensor>();
    auto level = static_cast<size_t>(Attr<int>("level"));
    auto &x_lod = x.lod();
    auto &mask_dim = mask.dims();

    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(dev_place);

    std::unique_ptr<framework::LoDTensor> cpu_mask{new framework::LoDTensor()};
    if (platform::is_cpu_place(mask.place())) {
      cpu_mask->ShareDataWith(mask);
    } else if (platform::is_gpu_place(mask.place())) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
      framework::TensorCopy(mask, platform::CPUPlace(), dev_ctx,
                            cpu_mask.get());
#else
      PADDLE_THROW(paddle::platform::errors::Fatal(
          "Not support GPU, Please compile WITH_GPU option"));
#endif
    }
    auto *mask_data = cpu_mask->data<bool>();

    std::vector<std::vector<CopyRange>> copy_ranges(2);

    // set out_true/out_false lod
    for (size_t t = 0; t < 2; t++) {
      LoD *lod = nullptr;
      if (t == 0) {
        lod = out_false->mutable_lod();
      } else {
        lod = out_true->mutable_lod();
      }
      lod->clear();
      for (size_t i = 0; i < static_cast<size_t>(mask_dim[0]); i++) {
        if (static_cast<size_t>(mask_data[i]) == t) {
          size_t start_idx = i;
          auto lod_and_offset = framework::GetSubLoDAndAbsoluteOffset(
              x_lod, start_idx, start_idx + 1, level);

          auto &lod_length = lod_and_offset.first;
          framework::AppendLoD(lod, lod_length);

          size_t start_offset = lod_and_offset.second.first;
          size_t end_offset = lod_and_offset.second.second;
          copy_ranges[t].emplace_back(CopyRange{start_offset, end_offset});
        }
      }
    }

    for (size_t t = 0; t < 2; ++t) {
      framework::LoDTensor *out;
      if (t == 0) {
        out = out_false;
      } else {
        out = out_true;
      }
      auto &ranges = copy_ranges[t];
      size_t height = std::accumulate(
          ranges.begin(), ranges.end(), 0UL,
          [](size_t a, const CopyRange &b) { return a + b.end - b.begin; });
      auto x_dim = x.dims();
      x_dim[0] = static_cast<int64_t>(height);
      out->Resize(x_dim);
      out->mutable_data(x.place(), x.type());
      size_t offset = 0;
      for (auto &each_range : ranges) {
        size_t len = each_range.end - each_range.begin;
        if (len == 0) {
          continue;
        }
        // out[offset: offset+len] = x[each_range.begin: each_range.end]
        auto slice = out->Slice(static_cast<int>(offset),
                                static_cast<int>(offset + len));
        framework::TensorCopy(x.Slice(static_cast<int>(each_range.begin),
                                      static_cast<int>(each_range.end)),
                              x.place(), dev_ctx, &slice);
        offset += len;
      }
    }
  }
};

class SplitLoDTensorOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "The input LoDTensor");
    AddInput("Mask", "A bool column vector which mask the input");
    AddOutput("OutTrue", "True branch of input LoDTensor");
    AddOutput("OutFalse", "False branch of input LoDTensor");
    AddAttr<int>("level", "(int) the specific lod level to split.")
        .SetDefault(0)
        .EqualGreaterThan(0);
    AddComment(
        R"DOC(
        Split a LoDTensor with a Mask at certain level. The input LoDTensor
        has 3 sequence at certain lod level. The Mask is a bool column vector,
        such as [0, 1, 0] at the same level. The first and third sequence will
        be send to False Output LoDTensor; whereas the second sequence will
        be send to True Output LoDTensor. Please refer to MergeLoDTensorOp.)DOC");
  }
};

class SplitLoDTensorInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *context) const override {
    OP_INOUT_CHECK(context->HasInput("X"), "Input", "X", "SplitLoDTensor");
    OP_INOUT_CHECK(context->HasInput("Mask"), "Input", "Mask",
                   "SplitLoDTensor");
    OP_INOUT_CHECK(context->HasOutput("OutTrue"), "Output", "OutTrue",
                   "SplitLoDTensor");
    OP_INOUT_CHECK(context->HasOutput("OutFalse"), "Output", "OutFalse",
                   "SplitLoDTensor");

    auto mask_dim = context->GetInputDim("Mask");
    PADDLE_ENFORCE_EQ(
        mask_dim.size(), 2,
        platform::errors::InvalidArgument(
            "If you are using IfElse OP:"
            "\n\nie = fluid.layers.IfElse(cond=cond)\nwith "
            "ie.true_block():\n    out_1 = ie.input(x)\n\n"
            "Please ensure that the cond should be a 2-D tensor and "
            "the second dim size of cond should be 1. "
            "But now the cond's shape is [",
            *mask_dim.Get(), "].\n"));
    PADDLE_ENFORCE_EQ(mask_dim[1], 1,
                      platform::errors::InvalidArgument(
                          "If you are using IfElse OP:"
                          "\n\nie = fluid.layers.IfElse(cond=cond)\nwith "
                          "ie.true_block():\n    out_1 = ie.input(x)\n\n"
                          "Please ensure that the cond should be a 2-D tensor "
                          "and the second dim size of cond should be 1. "
                          "But now the cond's shape is [",
                          *mask_dim.Get(), "].\n"));

    context->SetOutputDim("OutTrue", context->GetInputDim("X"));
    context->SetOutputDim("OutFalse", context->GetInputDim("X"));
  }
};

template <typename T>
class SplitLoDTensorArrayGradMaker : public framework::SingleGradOpMaker<T> {
 public:
  using framework::SingleGradOpMaker<T>::SingleGradOpMaker;

 protected:
  void Apply(GradOpPtr<T> grad_op) const override {
    grad_op->SetType("merge_lod_tensor");
    grad_op->SetInput("InTrue", this->OutputGrad("OutTrue"));
    grad_op->SetInput("InFalse", this->OutputGrad("OutFalse"));
    grad_op->SetInput("Mask", this->Input("Mask"));
    grad_op->SetInput("X", this->Input("X"));
    grad_op->SetOutput("Out", this->InputGrad("X"));
    grad_op->SetAttrMap(this->Attrs());
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(
    split_lod_tensor, ops::SplitLoDTensorOp, ops::SplitLoDTensorOpProtoMaker,
    ops::SplitLoDTensorInferShape,
    ops::SplitLoDTensorArrayGradMaker<paddle::framework::OpDesc>,
    ops::SplitLoDTensorArrayGradMaker<paddle::imperative::OpBase>);
