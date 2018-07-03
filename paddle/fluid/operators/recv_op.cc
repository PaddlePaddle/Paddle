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
#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/detail/macros.h"
#include "paddle/fluid/platform/profiler.h"

namespace paddle {
namespace operators {

class RecvOp : public framework::OperatorBase {
 public:
  RecvOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    int sync_mode = Attr<int>("sync_mode");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);
    // For profiling
    platform::RecordEvent record_event(Type(), &ctx);

    distributed::RPCClient* rpc_client =
        distributed::RPCClient::GetInstance<RPCCLIENT_T>();

    for (size_t i = 0; i < outs.size(); i++) {
      VLOG(3) << "getting " << outs[i] << " from " << epmap[i];
      rpc_client->AsyncGetVar(epmap[i], ctx, scope, outs[i]);
    }
    if (sync_mode) {
      rpc_client->Wait();
    }
  }
};

class RecvOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddOutput("Out", "(Tensor) Variables to get from server.").AsDuplicable();
    AddComment(R"DOC(
Recv operator

This operator can get variables from server side.
)DOC");
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({});
    AddAttr<int>("sync_mode",
                 "(int, default 0)"
                 "sync recv or async recv.")
        .SetDefault(0);
  }
};

class RecvOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(recv, ops::RecvOp, paddle::framework::EmptyGradOpMaker,
                  ops::RecvOpMaker, ops::RecvOpShapeInference);
