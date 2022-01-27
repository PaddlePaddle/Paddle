/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>         // NOLINT
#include <unordered_map>  // NOLINT
#include <unordered_set>  // NOLINT
#include <vector>
#include "paddle/fluid/framework/heter_service.pb.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/platform/timer.h"
#endif

namespace paddle {
namespace framework {

#if defined(PADDLE_WITH_PSLIB) && !defined(PADDLE_WITH_HETERPS)
typedef std::function<int(const HeterRequest*, HeterResponse*)>
    HeterServiceHandler;
class DataFeed;

class HeterXpuService : public HeterService {
 public:
  HeterXpuService() {}
  virtual ~HeterXpuService() {}
  void service(::google::protobuf::RpcController* controller,
               const HeterRequest* request, HeterResponse* response,
               ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    int ret = 0;
    int cmd = request->cmd();
    auto itr = handler_map_.find(cmd);
    if (itr == handler_map_.end()) {
    } else {
      ret = itr->second(request, response);
    }
    // response->set_err_code(0);
    // response->set_err_msg("");
    if (ret != 0) {
      // response->set_err_code(-1);
      // response->set_err_msg("xpu service error");
    }
  }

  void RegisterServiceHandler(int cmd, HeterServiceHandler func) {
    VLOG(0) << "register heter service";
    handler_map_[cmd] = func;
  }

 private:
  std::unordered_map<int, HeterServiceHandler> handler_map_;
};

#endif

}  // namespace framework
}  // namespace paddle
