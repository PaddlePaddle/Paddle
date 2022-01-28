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

#include "paddle/fluid/operators/data/data_reader_op.h"

namespace paddle {
namespace operators {
namespace data {

// initialization static variables out of ReaderManager
ReaderManager *ReaderManager::rm_instance_ptr_ = nullptr;
std::mutex ReaderManager::m_;

class DataReaderOp : public framework::OperatorBase {
 public:
  DataReaderOp(const std::string& type,
               const framework::VariableNameMap& inputs,
               const framework::VariableNameMap& outputs,
               const framework::AttributeMap& attrs)
               : OperatorBase(type, inputs, outputs, attrs) {}

  void InferShape(framework::InferShapeContext* ctx) const {
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "DataReaderOp");
  }

//  protected:
//   framework::OpKernelType GetExpectedKernelType(
//       const framework::ExecutionContext& ctx) const {
//     return framework::OpKernelType(framework::proto::VarType::FP32,
//                                    ctx.GetPlace());
//   }
//
 private:
  void RunImpl(const framework::Scope& scope,
      const platform::Place& dev_place) const override {
    auto outputs = Outputs("Out");
    std::vector<Variable*> output_vars;
    output_vars.reserve(outputs.size());
    for (auto& output : outputs) {
      output_vars.emplace_back(scope.FindVar(output));
    }

    CheckAndInitOutputQueue(output_vars, /*capacity=*/2);

    auto batch_size = Attr<int>("batch_size");
    auto num_samples = Attr<int>("num_samples");
    auto shuffle = Attr<bool>("shuffle");
    auto drop_last = Attr<bool>("drop_last");
    auto rank = Attr<int>("rank");
    auto world_size = Attr<int>("world_size");
    auto indices_var_name = Attr<std::string>("indices_var_name");
    auto output_var_names = Attr<std::vector<std::string>>("output_var_names");
    auto* reader_block = Attr<BlockDesc*>("reader_block");
    auto reader_id = Attr<int64_t>("reader_id");
    LOG(ERROR) << "DataReaderOp enter, reader_id: " << reader_id;

    auto output_queues = GetQueueVecFromVariableVec(output_vars);
    ReaderManager::Instance()->StartDataReader(
        reader_id, reader_block, &scope, platform::CPUPlace(), indices_var_name,
        output_var_names, output_queues, batch_size, num_samples,
        shuffle, drop_last, rank, world_size);
  }
};

class DataReaderInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasOutputs("Out"), "Output", "Out", "MapOp");
  }
};

class DataReaderInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {}
};

class DataReaderOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddOutput("Out", "The output queue variable of DataReader op")
        .AsDuplicable();
    AddAttr<int>("batch_size", "The batch size for reading samples")
        .SetDefault(1);
    AddAttr<int>("num_samples", "The sample number in dataset");
    AddAttr<bool>("shuffle", "Whether shuffle the dataset")
        .SetDefault(false);
    AddAttr<bool>("drop_last", "Whether drop last incomplete batch")
        .SetDefault(false);
    AddAttr<int>("rank", "The logical rank of current device.")
        .SetDefault(0);
    AddAttr<int>("world_size", "The number of running devices.")
        .SetDefault(1);
    AddAttr<int64_t>("reader_id", "The unique id to generate and get reader");
    AddAttr<BlockDesc*>("reader_block",
                        "(BlockDesc *)"
                        "The global block of executed reader program "
                        "desc.");
    AddAttr<std::string>("indices_var_name",
                     "(string)"
                     "input variable names for sample indices");
    AddAttr<std::vector<std::string>>("output_var_names",
                     "(list of string)"
                     "output variable names for reader program");
    AddComment(R"DOC(
        This operator read a file.
)DOC");
  }
};

}  // namespace data
}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators::data;

REGISTER_OPERATOR(
    data_reader, ops::DataReaderOp, ops::DataReaderOpMaker,
    ops::DataReaderInferShape, ops::DataReaderInferVarType)

REGISTER_OP_CPU_KERNEL(data_reader, ops::DataReaderCPUKernel<uint8_t>)
