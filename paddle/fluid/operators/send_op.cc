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

#include <ostream>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"

#include <future>
#include "paddle/fluid/operators/detail/grpc_client.h"
#include "paddle/fluid/operators/detail/sendrecvop_utils.h"

#include <sys/time.h>
#include "paddle/fluid/framework/threadpool.h"

namespace paddle {
namespace operators {
static bool IsVariableInitialized(const framework::Scope& scope,
                                  const std::string& varname) {
  auto* var = scope.FindVar(varname);
  PADDLE_ENFORCE_NOT_NULL(var, "Can not find variable '%s' in the send side.",
                          varname);
  if (var->IsType<framework::LoDTensor>()) {
    return var->Get<framework::LoDTensor>().IsInitialized();
  } else if (var->IsType<framework::SelectedRows>()) {
    return var->Get<framework::SelectedRows>().value().IsInitialized();
  } else {
    PADDLE_THROW(
        "Variable type in send side should be in "
        "[LodTensor, SelectedRows]");
  }
  return false;
}

static void PrintVarDims(const framework::Scope& scope,
                         const std::string& varname) {
  auto* var = scope.FindVar(varname);
  PADDLE_ENFORCE_NOT_NULL(var, "Can not find variable '%s' in the send side.",
                          varname);
  if (var->IsType<framework::LoDTensor>()) {
    VLOG(3) << "sending LoDTensor  varname: " << varname
            << ", dims: " << var->Get<framework::LoDTensor>().dims();
  } else if (var->IsType<framework::SelectedRows>()) {
    VLOG(3) << "sending SelectedRows varname: " << varname << ", dims: "
            << var->Get<framework::SelectedRows>().GetCompleteDims();
  }
}

void* hl_malloc_host_2(size_t size) {
  void* dest_h = NULL;
  PADDLE_ENFORCE(cudaHostAlloc((void**)&dest_h, size, cudaHostAllocDefault) ==
                 0);
  return dest_h;
}

void hl_free_mem_host_2(void* dest_h) {
  PADDLE_ENFORCE(cudaFreeHost(dest_h) == 0);
}

sendrecv::VariableMessage* GetOne(const std::string& ep,
                                  const platform::DeviceContext& ctx,
                                  const framework::Scope& scope,
                                  const std::string& var_name, double* sum,
                                  char* buf) {
  const platform::DeviceContext* p_ctx = &ctx;
  const std::string ep_val = ep;
  const std::string var_name_val = var_name;
  const framework::Scope* p_scope = &scope;
  // const auto ch = GetChannel(ep_val);
  auto* var = p_scope->FindVar(var_name_val);
  sendrecv::VariableMessage* req = new sendrecv::VariableMessage();

  struct timeval t1, t0;
  gettimeofday(&t0, 0);
  paddle::operators::detail::SerializeToMessage(var_name_val, var, *p_ctx, req,
                                                buf);
  gettimeofday(&t1, 0);
  double dif = double((t1.tv_sec - t0.tv_sec) * 1000.0 +
                      (t1.tv_usec - t0.tv_usec) / 1000.0);
  printf("Test in single %s time is %.2f ms.\n", var_name.c_str(), dif);
  *sum += dif;
  return req;
}

class SendOp : public framework::OperatorBase {
 public:
  SendOp(const std::string& type, const framework::VariableNameMap& inputs,
         const framework::VariableNameMap& outputs,
         const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const platform::Place& place) const override {
    struct timeval t1, t0;
    gettimeofday(&t0, 0);

    auto ins = Inputs("X");
    auto outs = Outputs("Out");
    std::vector<std::string> epmap = Attr<std::vector<std::string>>("epmap");
    std::vector<std::string> endpoints =
        Attr<std::vector<std::string>>("endpoints");

    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& ctx = *pool.Get(place);

    auto client_var_name = Output("RPCClient");
    PADDLE_ENFORCE_NOT_NULL(scope.FindVar(client_var_name),
                            "Can not find variable '%s' in the scope.",
                            client_var_name);
    auto* client_var = scope.FindVar(client_var_name);
    detail::RPCClient* rpc_client = client_var->GetMutable<detail::RPCClient>();

    /*
    struct timeval t2;
    gettimeofday(&t1, 0);
    for (size_t i = 0; i < ins.size(); i++) {
      TestOne(epmap[i], ctx, scope, ins[i]);
    }
    gettimeofday(&t2, 0);
    double dif = double((t2.tv_sec - t1.tv_sec) * 1000.0 +
                        (t2.tv_usec - t1.tv_usec) / 1000.0);
    printf("TestOne is %.2f ms.\n", dif);
    */

    std::vector<sendrecv::VariableMessage*> req;
    double copy_time = 0;

    constexpr size_t kBufSize = 1024 * 1024 * 64;  // 64MB
    char* buf = (char*)hl_malloc_host_2(kBufSize);
    for (size_t i = 0; i < ins.size(); i++) {
      req.push_back(GetOne(epmap[i], ctx, scope, ins[i], &copy_time, buf));
      if (IsVariableInitialized(scope, ins[i])) {
        VLOG(3) << "sending " << ins[i] << " to " << epmap[i];
        rpc_client->AsyncSendVariable(epmap[i], ctx, scope, ins[i], req[i]);
      } else {
        VLOG(3) << "don't send no-initialied variable: " << ins[i];
      }
    }
    hl_free_mem_host_2(buf);

    PADDLE_ENFORCE(rpc_client->Wait());
    gettimeofday(&t1, 0);
    double dif = double((t1.tv_sec - t0.tv_sec) * 1000.0 +
                        (t1.tv_usec - t0.tv_usec) / 1000.0);
    printf("Sending time is %.2f ms, copy_time:%.2f ms .\n", dif, copy_time);

    for (size_t i = 0; i < ins.size(); i++) {
      delete req[i];
    }

    gettimeofday(&t0, 0);
    for (auto& ep : endpoints) {
      VLOG(3) << "batch barrier, ep: " << ep;
      rpc_client->AsyncSendBatchBarrier(ep);
    }
    PADDLE_ENFORCE(rpc_client->Wait());
    gettimeofday(&t1, 0);
    dif = double((t1.tv_sec - t0.tv_sec) * 1000.0 +
                 (t1.tv_usec - t0.tv_usec) / 1000.0);
    printf("barrier time is %.2f ms.\n", dif);

    gettimeofday(&t0, 0);
    if (outs.size() > 0) {
      for (size_t i = 0; i < outs.size(); i++) {
        VLOG(3) << "getting " << outs[i] << " from " << epmap[i];
        rpc_client->AsyncGetVariable(epmap[i], ctx, scope, outs[i]);
      }
      PADDLE_ENFORCE(rpc_client->Wait());
    }
    gettimeofday(&t1, 0);
    dif = double((t1.tv_sec - t0.tv_sec) * 1000.0 +
                 (t1.tv_usec - t0.tv_usec) / 1000.0);
    printf("getting time is %.2f ms.\n", dif);
  }
};

class SendOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  SendOpMaker(OpProto* proto, OpAttrChecker* op_checker)
      : OpProtoAndCheckerMaker(proto, op_checker) {
    AddInput("X", "(Tensor) Input tensor to be sent").AsDuplicable();
    AddOutput("Out", "(Tensor) Output tensor to be received from server")
        .AsDuplicable();
    AddOutput("RPCClient",
              "(RPCClient) The RPC client object which is"
              "initialized at most once.");
    AddComment(R"DOC(
Send operator

This operator will send tensor to recv_op at the parameter server.
)DOC");
    // TODO(typhoonzero): remove this attr generate de-duplicated vector from
    // epmap when initializing.
    AddAttr<std::vector<std::string>>("endpoints",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints to send variables to.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>("epmap",
                                      "(string vector, default 127.0.0.1:6164)"
                                      "Server endpoints in the order of input "
                                      "variables for mapping")
        .SetDefault({});
  }
};

class SendOpVarTypeInference : public framework::VarTypeInference {
 public:
  void operator()(const framework::OpDesc& op_desc,
                  framework::BlockDesc* block) const override {
    auto out_var_name = op_desc.Output("RPCClient").front();
    auto& out_var = block->FindRecursiveOrCreateVar(out_var_name);
    auto var_type = framework::proto::VarType::RAW;
    out_var.SetType(var_type);
  }
};

class SendOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;

REGISTER_OPERATOR(send, ops::SendOp, paddle::framework::EmptyGradOpMaker,
                  ops::SendOpMaker, ops::SendOpVarTypeInference,
                  ops::SendOpShapeInference);
