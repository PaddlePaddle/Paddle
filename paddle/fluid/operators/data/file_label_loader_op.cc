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

#include "paddle/fluid/operators/data/file_label_loader_op.h"

namespace paddle {
namespace operators {
namespace data {
// FileDataReaderWrapper reader_wrapper;

// // initialization static variables out of ReaderManager
// ReaderManager *ReaderManager::rm_instance_ptr_ = nullptr;
// std::mutex ReaderManager::m_;

class FileLabelLoaderOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

 protected:
  void InferShape(framework::InferShapeContext* ctx) const {
    PADDLE_ENFORCE_EQ(ctx->HasInput("Indices"), true,
                      platform::errors::InvalidArgument(
                          "Input(Indices) of ReadFileLoaderOp is null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Image"), true,
                      platform::errors::InvalidArgument(
                          "Output(Image) of ReadFileLoaderOp is null."));
    PADDLE_ENFORCE_EQ(ctx->HasOutput("Label"), true,
                      platform::errors::InvalidArgument(
                          "Output(Label) of ReadFileLoaderOp is null."));

    auto dim_indices = ctx->GetInputDim("Indices");
    PADDLE_ENFORCE_EQ(dim_indices.size(), 1,
                      platform::errors::InvalidArgument(
                          "Input(Indices) should be a 1-D Tensor"));

    auto files = ctx->Attrs().Get<std::vector<std::string>>("files");
    auto labels = ctx->Attrs().Get<std::vector<int>>("labels");
    PADDLE_ENFORCE_GT(files.size(), 0,
                      platform::errors::InvalidArgument(
                          "length of files should be greater than 0"));
    PADDLE_ENFORCE_GT(labels.size(), 0,
                      platform::errors::InvalidArgument(
                          "length of labels should be greater than 0"));
    PADDLE_ENFORCE_EQ(files.size(), labels.size(),
                      platform::errors::InvalidArgument(
                          "length of labels and files should be equal"));
  }

  framework::OpKernelType GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const {
    return framework::OpKernelType(framework::proto::VarType::UINT8,
                                   platform::CPUPlace());
  }

//  private:
//   void RunImpl(const framework::Scope& scope,
//                const platform::Place& dev_place) const override {
//     LOG(ERROR) << "FileLabelLoaderOp RunImpl start";
//     platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
//     auto& dev_ctx = *pool.Get(dev_place);
//     framework::RuntimeContext run_ctx(Inputs(), Outputs(), scope);
//     framework::ExecutionContext ctx(*this, scope, dev_ctx, run_ctx);
//
//     auto* out = scope.FindVar(Output("Out"));
//     auto out_queue = out->Get<LoDTensorBlockingQueueHolder>().GetQueue();
//     if (out_queue == nullptr) {
//       LOG(ERROR) << "FileLabelLoaderOp init output queue";
//       auto* holder = out->template GetMutable<LoDTensorBlockingQueueHolder>();
//       holder->InitOnce(2);
//       out_queue = holder->GetQueue();
//     }
//
//     auto* out_label = scope.FindVar(Output("Label"));
//     auto out_label_queue =
//         out_label->Get<LoDTensorBlockingQueueHolder>().GetQueue();
//     if (out_label_queue == nullptr) {
//       LOG(ERROR) << "FileLabelLoaderOp init output label queue";
//       auto* label_holder =
//           out_label->template GetMutable<LoDTensorBlockingQueueHolder>();
//       label_holder->InitOnce(2);
//       out_label_queue = label_holder->GetQueue();
//     }
//
//     ReaderManager::Instance()->GetReader(
//         0, ctx, out_queue.get(), out_label_queue.get());
//     // LoDTensorArray samples = reader_wrapper.reader->Next();
//     // framework::LoDTensorArray out_array;
//     // out_array.resize(samples.size());
//     // for (size_t i = 0; i < samples.size(); ++i) {
//     //   copy_tensor(samples[i], &out_array[i]);
//     // }
//     // out_queue->Push(out_array);
//     LOG(ERROR) << "FileLabelLoaderOp RunImpl finish";
//   }

  // std::shared_ptr<FileDataReader> reader=nullptr;
};

class FileLabelLoaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Indices", "The batch indices of input samples");
    AddOutput("Image", "The output image tensor of ReadFileLoader op");
    AddOutput("Label", "The output label tensor of ReadFileLoader op");
    AddAttr<std::vector<std::string>>("files", "Path of the file to be readed.")
        .SetDefault({});
    AddAttr<std::vector<int>>("labels", "Path of the file to be readed.")
        .SetDefault({});
    AddComment(R"DOC(
This operator read a file.
)DOC");
    // AddAttr<std::string>("root_dir", "Path of the file to be readed.")
    //     .SetDefault("");
    // AddAttr<int>("batch_size", "Path of the file to be readed.").SetDefault(1);
    // AddAttr<int>("rank", "Path of the file to be readed.").SetDefault(0);
    // AddAttr<int>("world_size", "Path of the file to be readed.").SetDefault(1);
    // AddAttr<int64_t>("reader_id",
    //                  "(int64_t)"
    //                  "The unique hash id used as cache key for "
    //                  "ExecutorInfoCache").SetDefault(0);;
  }
};

// class FileLabelReaderInferShape : public framework::InferShapeBase {
//  public:
//   void operator()(framework::InferShapeContext* context) const override {
//     OP_INOUT_CHECK(context->HasOutput("Out"), "Output", "Out",
//                    "FileLabelReader");
//   }
// };
//
// class FileLabelReaderInferVarType : public framework::VarTypeInference {
//  public:
//   void operator()(framework::InferVarTypeContext* ctx) const override {
//     ctx->SetOutputType("Out", framework::proto::VarType::LOD_TENSOR_ARRAY,
//                        framework::ALL_ELEMENTS);
//   }
// };

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::data;

REGISTER_OPERATOR(
    file_label_loader, ops::FileLabelLoaderOp, ops::FileLabelLoaderOpMaker,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>)

REGISTER_OP_CPU_KERNEL(file_label_loader, ops::FileLabelLoaderCPUKernel<uint8_t>)
