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
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/var_type.h"
#include "paddle/fluid/operators/distributed/request_handler.h"

namespace paddle {
namespace framework {
class Scope;
class Variable;
}  // namespace framework
}  // namespace paddle

namespace paddle {
namespace operators {
namespace distributed {

class RequestSendHandler final : public RequestHandler {
 public:
  explicit RequestSendHandler(int distributed_mode, bool enable_dc_asgd = false)
      : RequestHandler(distributed_mode) {
    enable_dc_asgd_ = enable_dc_asgd;
  }
  virtual ~RequestSendHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;

 private:
  bool enable_dc_asgd_;
};

class RequestGetHandler final : public RequestHandler {
 public:
  explicit RequestGetHandler(int distributed_mode, bool enable_dc_asgd = false)
      : RequestHandler(distributed_mode) {
    enable_dc_asgd_ = enable_dc_asgd;
  }
  virtual ~RequestGetHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;

 private:
  bool enable_dc_asgd_;
};

class RequestGetNoBarrierHandler final : public RequestHandler {
 public:
  RequestGetNoBarrierHandler() : RequestHandler(false) {}
  virtual ~RequestGetNoBarrierHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;
};

static inline void BuildVar(const std::string& param_name,
                            std::initializer_list<const char*> arguments,
                            paddle::framework::proto::OpDesc::Var* var) {
  var->set_parameter(param_name);
  for (auto& arg_name : arguments) {
    *var->mutable_arguments()->Add() = arg_name;
  }
}

class RequestPrefetchHandler final : public RequestHandler {
 public:
  explicit RequestPrefetchHandler(int distributed_mode)
      : RequestHandler(distributed_mode) {}
  virtual ~RequestPrefetchHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;

 private:
  std::unique_ptr<paddle::framework::OperatorBase> PullLargeScaleOp(
      const std::string& table_name, const std::string& id_name,
      const std::string& out_name) {
    framework::OpDesc desc;
    desc.SetType("lookup_sparse_table_read");
    desc.SetInput("Ids", {id_name});
    desc.SetOutput("Out", std::vector<std::string>({out_name}));
    desc.SetAttr("tablename", {table_name});
    desc.SetAttr("init", true);
    desc.SetAttr("value_names", std::vector<std::string>({"Param"}));

    auto op = paddle::framework::OpRegistry::CreateOp(desc);
    return op;
  }

  std::unique_ptr<paddle::framework::OperatorBase> BuildLookupTableOp(
      const std::string& table_name, const std::string& id_name,
      const std::string& out_name) {
    paddle::framework::proto::OpDesc op_desc;
    op_desc.set_type("lookup_table");
    BuildVar("W", {table_name.data()}, op_desc.add_inputs());
    BuildVar("Ids", {id_name.data()}, op_desc.add_inputs());
    BuildVar("Out", {out_name.data()}, op_desc.add_outputs());

    auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
    return op;
  }
};

class RequestCheckpointHandler final : public RequestHandler {
 public:
  explicit RequestCheckpointHandler(int distributed_mode)
      : RequestHandler(distributed_mode) {}

  virtual ~RequestCheckpointHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;

 private:
  std::unique_ptr<paddle::framework::OperatorBase> BuildCheckpointOp(
      const std::string& varname, const std::string& file_path) {
    paddle::framework::proto::OpDesc op_desc;
    op_desc.set_type("save");
    BuildVar("X", {varname.data()}, op_desc.add_inputs());

    auto attr = op_desc.mutable_attrs()->Add();
    attr->set_name("file_path");
    attr->set_type(paddle::framework::proto::AttrType::STRING);
    attr->set_s(file_path);

    auto op = paddle::framework::OpRegistry::CreateOp(op_desc);
    return op;
  }
};

class RequestNotifyHandler final : public RequestHandler {
 public:
  explicit RequestNotifyHandler(int distributed_mode, int trainers)
      : RequestHandler(distributed_mode) {
    this->trainers = trainers;
    for (int i = 0; i < trainers; i++) {
      decay_counters[i] = 0;
    }
  }
  virtual ~RequestNotifyHandler() {}
  bool Handle(const std::string& varname, framework::Scope* scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;

 private:
  int trainers;
  std::unordered_map<int, int64_t> decay_counters;
};

class RequestSendAndRecvHandler final : public RequestHandler {
 public:
  explicit RequestSendAndRecvHandler(int distributed_mode)
      : RequestHandler(distributed_mode) {}
  virtual ~RequestSendAndRecvHandler() {}
  bool Handle(const std::string& varname, framework::Scope* Scope,
              framework::Variable* var, framework::Variable** outvar,
              const int trainer_id, const std::string& out_var_name = "",
              const std::string& table_name = "") override;
};

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
