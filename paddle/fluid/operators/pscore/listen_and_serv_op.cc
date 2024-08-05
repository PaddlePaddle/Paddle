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

constexpr char kLRDecayBlockId[] = "lr_decay_block_id";      // NOLINT
constexpr char kCheckpointBlockId[] = "checkpint_block_id";  // NOLINT
constexpr char kPrefetchVarNameToBlockId[] =
    "prefetch_var_name_to_block_id";                           // NOLINT
constexpr char kOptimizeBlocks[] = "optimize_blocks";          // NOLINT
constexpr char kSparseGradToParam[] = "sparse_grad_to_param";  // NOLINT

namespace paddle::framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace paddle::framework
namespace paddle::imperative {
class OpBase;
}  // namespace paddle::imperative

namespace paddle::operators {

class ListenAndServOp : public framework::OperatorBase {
 public:
  ListenAndServOp(const std::string& type,
                  const framework::VariableNameMap& inputs,
                  const framework::VariableNameMap& outputs,
                  const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

  void RunImpl(const framework::Scope& scope,
               const phi::Place& place) const override {
    VLOG(1) << "just for recorder";
  }
};

class ListenAndServOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("X", "(Tensor) Variables that server recv.").AsDuplicable();
    AddComment(R"DOC(" + "ListenAndServ operator" + "\n" + "This operator" +
" will start a RPC server which can receive variables from send_op and send" +
"back variables to recv_op.)DOC");
    AddAttr<std::string>("endpoint",
                         "(string, default 127.0.0.1:6164)"
                         "IP address to listen on.")
        .SetDefault("127.0.0.1:6164")
        .AddCustomChecker([](const std::string& ip) { return !ip.empty(); });
    AddAttr<int>("pserver_id",
                 "(int, default -1), the parameter server index id")
        .SetDefault(-1);
    AddAttr<std::vector<std::string>>(
        "grad_to_block_id",
        "['param1@GRAD.block0:1', 'param2@GRAD.blockn:2'] "
        "a map from grad name to it's optimize block id")
        .SetDefault({});
    AddAttr<int>("distributed_mode",
                 "indicate distriubte training mode, 0 is sync, 1 is "
                 "fully-async, 2 is half-async, 3 is geo")
        .SetDefault(0);
    AddAttr<bool>("dc_asgd", "set to true will enable DC-ASGD training.")
        .SetDefault(false);
    AddAttr<std::vector<framework::BlockDesc*>>(
        kOptimizeBlocks, "Optimize blocks to run on server side.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(kPrefetchVarNameToBlockId,
                                      "prefetch blocks to run on server side.")
        .SetDefault({});
    AddAttr<std::vector<std::string>>(
        kSparseGradToParam,
        "sparse grad name to param name. like: 'emb@Grad:emb'")
        .SetDefault({});
    AddAttr<int>("Fanin", "How many clients send to this server.")
        .SetDefault(1);
    AddAttr<int>(kCheckpointBlockId,
                 "BlockID to run save checkpoint on pserver.")
        .SetDefault(-1);
    AddAttr<int>(kLRDecayBlockId, "BlockID to run lr decay on pserver.")
        .SetDefault(-1);
    AddAttr<int>("rpc_get_thread_num", "pserver get thread num.").SetDefault(1);
    AddAttr<int>("rpc_send_thread_num", "pserver send thread num.")
        .SetDefault(1);
    AddAttr<int>("rpc_prefetch_thread_num", "pserver prefetch thread num.")
        .SetDefault(1);
  }
};

class ListenAndServOpShapeInference : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* ctx) const override {}
};

}  // namespace paddle::operators

namespace ops = paddle::operators;

REGISTER_OPERATOR(
    listen_and_serv,
    ops::ListenAndServOp,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>,
    ops::ListenAndServOpMaker,
    ops::ListenAndServOpShapeInference);
