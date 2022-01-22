/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once

#include <stdint.h>
#include <atomic>
#include <memory>
#include <set>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/distributed/ps/service/brpc_utils.h"
#include "paddle/fluid/distributed/ps/service/heter_server.h"
#include "paddle/fluid/distributed/ps/service/sendrecv.pb.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/threadpool.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace distributed {
class HeterRequestHandler;
class HeterServer;
}  // namespace distributed
}  // namespace paddle

namespace paddle {
namespace framework {
class Executor;
class ProgramDesc;
class Scope;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

template <class TKey, class TValue>
class DoubleFindMap : public std::unordered_map<TKey, TValue> {
 public:
  typename std::unordered_map<TKey, TValue>::iterator find_value(TValue v) {
    return std::find_if(this->begin(), this->end(),
                        [&v](const std::pair<const std::string, int> p) {
                          return p.second == v;
                        });
  }
};

void RunServer(std::shared_ptr<paddle::distributed::HeterServer> service);

class HeterListenAndServOp : public framework::OperatorBase {
 public:
  HeterListenAndServOp(const std::string& type,
                       const framework::VariableNameMap& inputs,
                       const framework::VariableNameMap& outputs,
                       const framework::AttributeMap& attrs);
  virtual ~HeterListenAndServOp();

  void RunAsyncLoop(framework::ProgramDesc* program) const;

  void Stop() override;

  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override;

 protected:
  mutable std::shared_ptr<paddle::distributed::HeterServer> rpc_service_;
  mutable std::shared_ptr<std::thread> server_thread_;
  mutable std::shared_ptr<paddle::distributed::RequestSendAndRecvHandler>
      request_send_and_recv_handler_;
};

}  // namespace operators
}  // namespace paddle
