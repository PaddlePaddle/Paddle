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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <boost/any.hpp>
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {
namespace compatible {

struct OpUpdateRecord {
  enum class Type { kInvalid = 0, kModifyAttr, kNewAttr };
  Type type_;
  std::string remark_;
};

struct ModifyAttr : OpUpdateRecord {
  ModifyAttr(const std::string& name, const std::string& remark,
             boost::any default_value)
      : OpUpdateRecord({Type::kModifyAttr, remark}),
        name_(name),
        default_value_(default_value) {
    // TODO(Shixiaowei02): Check the data type with proto::OpDesc.
  }

 private:
  std::string name_;
  boost::any default_value_;
};
struct NewAttr : OpUpdateRecord {
  NewAttr(const std::string& name, const std::string& remark)
      : OpUpdateRecord({Type::kNewAttr, remark}), name_(name) {}

 private:
  std::string name_;
};

class OpVersionDesc {
 public:
  OpVersionDesc& ModifyAttr(const std::string& name, const std::string& remark,
                            boost::any default_value) {
    infos_.push_back(std::shared_ptr<OpUpdateRecord>(
        new compatible::ModifyAttr(name, remark, default_value)));
    return *this;
  }

  OpVersionDesc& NewAttr(const std::string& name, const std::string& remark) {
    infos_.push_back(
        std::shared_ptr<OpUpdateRecord>(new compatible::NewAttr(name, remark)));
    return *this;
  }

 private:
  std::vector<std::shared_ptr<OpUpdateRecord>> infos_;
};

class OpVersion {
 public:
  OpVersion& AddCheckpoint(const std::string& note,
                           const OpVersionDesc& op_version_desc) {
    checkpoints_.push_back(Checkpoint({note, op_version_desc}));
    return *this;
  }

 private:
  struct Checkpoint {
    std::string note_;
    OpVersionDesc op_version_desc_;
  };
  std::vector<Checkpoint> checkpoints_;
};

class OpVersionRegistrar {
 public:
  static OpVersionRegistrar& GetInstance() {
    static OpVersionRegistrar instance;
    return instance;
  }
  OpVersion& Register(const std::string& op_type) {
    if (op_version_map_.find(op_type) != op_version_map_.end()) {
      PADDLE_THROW("'%s' is registered in operator version more than once.",
                   op_type);
    }
    op_version_map_.insert({op_type, OpVersion()});
    return op_version_map_[op_type];
  }

 private:
  std::unordered_map<std::string, OpVersion> op_version_map_;

  OpVersionRegistrar() = default;
  OpVersionRegistrar& operator=(const OpVersionRegistrar&) = delete;
};

}  // namespace compatible
}  // namespace framework
}  // namespace paddle

#define REGISTER_OP_VERSION(op_type)                                       \
  static paddle::framework::compatible::OpVersion                          \
      RegisterOpVersion__##op_type =                                       \
          paddle::framework::compatible::OpVersionRegistrar::GetInstance() \
              .Register(#op_type)
