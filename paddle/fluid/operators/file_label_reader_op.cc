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

#include "paddle/fluid/operators/file_label_reader_op.h"

namespace paddle {
namespace operators {

FileDataReaderWrapper reader_wrapper;

template <typename T>
class CPUFileLabelKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {}
};

class FileLabelReaderOp : public framework::OperatorBase {
 public:
  // using framework::OperatorWithKernel::OperatorWithKernel;
  FileLabelReaderOp(const std::string& type,
                    const framework::VariableNameMap& inputs,
                    const framework::VariableNameMap& outputs,
                    const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const {
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Out"), true,
                      platform::errors::InvalidArgument(
                          "Output(Out) of ReadFileOp is null."));
  }

 protected:
  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(framework::proto::VarType::UINT8,
                                   platform::CPUPlace());
  }

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    LOG(ERROR) << "FileLabelReaderOp RunImpl start";
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(dev_place);
    framework::RuntimeContext run_ctx(Inputs(), Outputs(), scope);
    framework::ExecutionContext ctx(*this, scope, dev_ctx, run_ctx);

    auto* out = scope.FindVar(Output("Out"));
    auto out_queue = out->Get<LoDTensorBlockingQueueHolder>().GetQueue();
    if (out_queue == nullptr) {
      LOG(ERROR) << "FileLabelReaderOp init output queue";
      auto* holder = out->template GetMutable<LoDTensorBlockingQueueHolder>();
      holder->InitOnce(2);
      out_queue = holder->GetQueue();
    }

    if (reader_wrapper.reader == nullptr) {
      // create reader
      reader_wrapper.SetUp(ctx, out_queue.get());
    }
    // LoDTensorArray samples = reader_wrapper.reader->Next();
    // framework::LoDTensorArray out_array;
    // out_array.resize(samples.size());
    // for (size_t i = 0; i < samples.size(); ++i) {
    //   copy_tensor(samples[i], &out_array[i]);
    // }
    // out_queue->Push(out_array);
    LOG(ERROR) << "FileLabelReaderOp RunImpl finish";
  }

  void copy_tensor(const framework::LoDTensor& lod_tensor,
                   framework::LoDTensor* out) const {
    if (lod_tensor.numel() == 0) return;
    auto& out_tensor = *out;
    TensorCopy(lod_tensor, lod_tensor.place(), &out_tensor);
    out_tensor.set_lod(lod_tensor.lod());
  }

  // std::shared_ptr<FileDataReader> reader=nullptr;
};

class FileLabelReaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The output tensor of ReadFile op");
    AddComment(R"DOC(
This operator read a file.
)DOC");
    AddAttr<std::string>("root_dir", "Path of the file to be readed.")
        .SetDefault("");
    AddAttr<int>("batch_size", "Path of the file to be readed.").SetDefault(1);
    AddAttr<int>("rank", "Path of the file to be readed.").SetDefault(0);
    AddAttr<int>("world_size", "Path of the file to be readed.").SetDefault(1);
    AddAttr<std::vector<std::string>>("files", "Path of the file to be readed.")
        .SetDefault({});
    AddAttr<std::vector<int>>("labels", "Path of the file to be readed.")
        .SetDefault({});
  }
};

class FileLabelReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out",
                   "FileLabelReader");
  }
};

class FileLabelReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY,
                       framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    file_label_reader, ops::FileLabelReaderOp, ops::FileLabelReaderOpMaker,
    ops::FileLabelReaderInferShape, ops::FileLabelReaderInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(file_label_reader, ops::CPUFileLabelKernel<uint8_t>)
