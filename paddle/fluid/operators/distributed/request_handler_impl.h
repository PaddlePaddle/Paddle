// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include <time.h>

#include <functional>
#include <string>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace paddle {
namespace operators {
namespace distributed {

class RequestSendHandler final : public RequestHandler {
 public:
  explicit RequestSendHandler(bool sync_mode) : RequestHandler(sync_mode) {}
  virtual ~RequestSendHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const std::string& out_var_name = "") override;
  void ResetSparseVarRecorder();

 private:
  std::mutex mutex_sparse_vars_;
  std::vector<framework::Variable*> sparse_vars_;
};

class RequestGetHandler final : public RequestHandler {
 public:
  explicit RequestGetHandler(bool sync_mode) : RequestHandler(sync_mode) {}
  virtual ~RequestGetHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const std::string& out_var_name = "") override;
};

class RequestPrefetchHandler final : public RequestHandler {
 public:
  explicit RequestPrefetchHandler(bool sync_mode) : RequestHandler(sync_mode) {}
  virtual ~RequestPrefetchHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const std::string& out_var_name = "") override;
};

class RequestCheckpointHandler final : public RequestHandler {
 public:
  explicit RequestCheckpointHandler(bool sync_mode, int checkpoint_notify_id)
      : RequestHandler(sync_mode) {
    this->checkpoint_notify_id = checkpoint_notify_id;
  }
  virtual ~RequestCheckpointHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const std::string& out_var_name = "") override;

 private:
  int checkpoint_notify_id;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
