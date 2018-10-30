/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

class ExtractRowsOpInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext *ctx) const override {
    PADDLE_ENFORCE(ctx->HasInput("X"),
                   "Input(X) of ExtractRowsOp should not be null.");
    PADDLE_ENFORCE(ctx->HasOutput("Out"),
                   "Output(Out) of ExtractRowsOp should not be null.");
    PADDLE_ENFORCE_EQ(ctx->GetInputsVarType("X")[0],
                      framework::proto::VarType::SELECTED_ROWS,
                      "The type of input(X) must be SelectedRows.");
    auto in_dims = ctx->GetInputDim("X");

    ctx->SetOutputDim(
        "Out", framework::make_ddim(std::vector<int64_t>{in_dims[0], 1}));
  }
};

class ExtractRowsOp : public framework::OperatorBase {
 public:
  ExtractRowsOp(const std::string &type,
                const framework::VariableNameMap &inputs,
                const framework::VariableNameMap &outputs,
                const framework::AttributeMap &attrs)
      : framework::OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope &scope,
               const platform::Place &place) const override {
    auto &in = scope.FindVar(Input("X"))->Get<framework::SelectedRows>();
    auto out = scope.FindVar(Output("Out"))->GetMutable<framework::LoDTensor>();

    auto &in_rows = in.rows();
    auto out_dim = framework::make_ddim(
        std::vector<int64_t>{static_cast<int64_t>(in_rows.size()), 1});
    auto dst_ptr = out->mutable_data<int64_t>(out_dim, in.place());

    if (paddle::platform::is_gpu_place(in.place())) {
#ifdef PADDLE_WITH_CUDA
      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto *dev_ctx = pool.Get(in.place());
      auto src_ptr = in_rows.Data(in.place());
      auto stream =
          reinterpret_cast<const platform::CUDADeviceContext &>(*dev_ctx)
              .stream();
      memory::Copy(boost::get<platform::CUDAPlace>(out->place()), dst_ptr,
                   boost::get<platform::CUDAPlace>(in.place()), src_ptr,
                   in_rows.size() * sizeof(int64_t), stream);
#else
      PADDLE_THROW("Not compiled with CUDA.");
#endif
    } else {
      memory::Copy(platform::CPUPlace(), dst_ptr, platform::CPUPlace(),
                   in_rows.data(), in_rows.size() * sizeof(int64_t));
    }
  }
};

class ExtractRowsOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X",
             "(SelectedRows). The input tensor of extract_rows operator,"
             " and its type is SelectedRows.");
    AddOutput("Out", "(Tensor). The the rows of input(X).");

    AddComment(R"DOC(
    ExtractRows Operator.

The function of extract_rows_op is extracting the rows from the input(X)
whose type is SelectedRows.

    )DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(extract_rows, ops::ExtractRowsOp, ops::ExtractRowsOpMaker,
                  ops::ExtractRowsOpInferShape);
