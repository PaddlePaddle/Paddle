/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/operators/sendrecv.h"

namespace paddle {
namespace operators {

class SendOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "Send Tensor");
    AddAttr<std::string>("src_device", "src device name, can be rpc endpoint");
    AddAttr<std::string>("dst_device", "dst device name, can be rpc endpoint");
    AddComment(R"DOC(
Send Operator.
Send local tensor to another node/device.
if two endpoint is rpc endpoint, then it will send tensor to remote node.
otherwise, it will copy tensor from src_device to dst_device.
)DOC");
  }
};

class RecvOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(framework::OpProto* proto, framework::OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddOutput("Out", "Recv Tensor");
    AddAttr<std::string>("src_device", "src device name, can be rpc endpoint");
    AddAttr<std::string>("dst_device", "dst device name, can be rpc endpoint");
    AddComment(R"DOC(
Recv Operator.
Recv tensor from another node/device.
if two endpoint is rpc endpoint, then it will receive tensor from remote node.
otherwise, it will copy tensor from src_device to dst_device.
)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
