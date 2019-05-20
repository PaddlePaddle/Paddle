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

#include <future>  // NOLINT
#include <memory>
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/fleet/nccl_wrapper.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#ifdef PADDLE_WITH_CUDA
#include "paddle/fluid/platform/nccl_helper.h"
#endif

namespace paddle {
namespace operators {

struct MutableDataFunctor {
  MutableDataFunctor(void** data, framework::LoDTensor* tensor,
                     const platform::Place& place)
      : data_(data), tensor_(tensor), place_(place) {}

  template <typename T>
  void apply() {
    *data_ = tensor_->mutable_data<T>(place_);
  }

  void** data_;
  framework::LoDTensor* tensor_;
  platform::Place place_;
};

class AllReduceOp : public framework::OperatorBase {
  using OperatorBase::OperatorBase;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    PADDLE_ENFORCE(is_gpu_place(place),
                   "AllReduce op can run on gpu place only for now.");
#ifdef PADDLE_WITH_CUDA
    auto in_names = Inputs("X");
    auto out_names = Outputs("Out");
    PADDLE_ENFORCE_EQ(in_names.size(), 1, "Only support one input");
    PADDLE_ENFORCE_EQ(out_names.size(), 1, "Only support one output");
    std::shared_ptr<framework::NCCLWrapper> nccl_ptr_ =
        framework::NCCLWrapper::GetInstance();
    int reduce_type = Attr<int>("reduce_type");
    nccl_ptr_->AllReduce(scope, in_names[0], out_names[0], place, reduce_type);
#endif
  }
};

class AllReduceOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "(Tensor), tensor to be allreduced.");
    AddOutput("Out", "(Tensor) the result of allreduced.");
    AddAttr<int>("reduce_type", "(int) determin the reduce type.")
        .SetDefault(0);
    AddComment(R"DOC(
***AllReduce Operator***

Call NCCL AllReduce internally. Note that this op must be used when one
thread is managing one GPU device.

For speed reasons, reduce_type should be an integer:

0: sum
1: prod
2: max
3: min

If input and output are the same variable, in-place allreduce will be used.
)DOC");
  }
};

class AllReduceOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(allreduce, ops::AllReduceOp,
                  paddle::framework::EmptyGradOpMaker, ops::AllReduceOpMaker,
                  ops::AllReduceOpShapeInference);
