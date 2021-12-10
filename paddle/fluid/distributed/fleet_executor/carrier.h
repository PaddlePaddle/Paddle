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

#pragma once

#include <condition_variable>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/distributed/fleet_executor/interceptor.h"
#include "paddle/fluid/distributed/fleet_executor/interceptor_message.pb.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/errors.h"
#include "paddle/fluid/platform/macros.h"
#include "paddle/fluid/platform/place.h"

namespace paddle {
namespace framework {
class Scope;
}

namespace distributed {

class TaskNode;
class InterceptorMessageServiceImpl;
class RuntimeGraph;

// A singleton MessageBus
class Carrier final {
 public:
  static Carrier& Instance() {
    static Carrier carrier;
    return carrier;
  }

  void Init(std::shared_ptr<RuntimeGraph> runtime_graph,
            framework::Scope* root_scope, framework::Scope* minibatch_scope,
            const std::vector<framework::Scope*>& microbatch_scopes,
            const platform::Place& place);

  ~Carrier();
  void Release();

  // Enqueue a message to corresponding interceptor id
  bool EnqueueInterceptorMessage(const InterceptorMessage& interceptor_message);

  // get interceptor based on the interceptor id
  Interceptor* GetInterceptor(int64_t interceptor_id);

  // set interceptor with interceptor id
  Interceptor* SetInterceptor(int64_t interceptor_id,
                              std::unique_ptr<Interceptor>);

  void SetCreatingFlag(bool flag);

  std::condition_variable& GetCondVar();

  void Start();

  bool IsInit() const;

  // NOTE: This mutex will be used in interceptor's RunOps function.
  // This mutex is used for avoiding forward ops and backward ops run
  // simultaneously, which will lead to a random hang for some sync ops.
  std::mutex run;

  DISABLE_COPY_AND_ASSIGN(Carrier);

 private:
  Carrier() = default;

  // create each Interceptor
  void CreateInterceptors();

  void HandleTmpMessages();

  // interceptor logic id to actually interceptor
  std::unordered_map<int64_t, std::unique_ptr<Interceptor>>
      interceptor_idx_to_interceptor_;

  std::vector<int64_t> source_interceptor_ids_;

  std::vector<InterceptorMessage> message_tmp_{};
  std::mutex tmp_message_mutex_;
  bool creating_interceptors_{true};
  std::mutex creating_flag_mutex_;
  bool is_init_{false};

  std::mutex running_mutex_;
  std::condition_variable cond_var_;
  std::vector<framework::Scope*> microbatch_scopes_;
  framework::Scope* root_scope_;
  framework::Scope* minibatch_scope_;
  paddle::platform::Place place_;
  paddle::platform::DeviceContext* dev_ctx_{nullptr};
  std::shared_ptr<RuntimeGraph> runtime_graph_;
};

}  // namespace distributed
}  // namespace paddle
