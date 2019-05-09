// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include "paddle/fluid/framework/framework.pb.h"
#include "paddle/fluid/lite/utils/all.h"

namespace paddle {
namespace lite {

// We define the runtime data structure for framework.proto to support some
// other model format such as JSON if needed.
using proto_type_t = framework::proto::VarType::Type;

class TensorDesc {
 public:
  proto_type_t data_type;
  std::vector<int64_t> dims;

  TensorDesc() = default;
  explicit TensorDesc(const framework::proto::VarType_TensorDesc& proto) {
    Parse(proto);
  }

  void Parse(const framework::proto::VarType_TensorDesc& proto) {
    data_type = proto.data_type();
    for (auto& d : proto.dims()) dims.push_back(d);
  }
};

class LoDTensorDesc {
 public:
  TensorDesc tensor;
  int lod_level{-1};

  LoDTensorDesc(const framework::proto::VarType_LoDTensorDesc& proto) {
    Parse(proto);
  }

  void Parse(const framework::proto::VarType_LoDTensorDesc& proto) {
    tensor.Parse(proto.tensor());
    lod_level = proto.lod_level();
  }
};

class LoDTensorArrayDesc {
 public:
  TensorDesc tensor;
  int lod_level{-1};

  LoDTensorArrayDesc(
      const framework::proto::VarType_LoDTensorArrayDesc& proto) {
    Parse(proto);
  }

  void Parse(const framework::proto::VarType_LoDTensorArrayDesc& proto) {
    tensor.Parse(proto.tensor());
    lod_level = proto.lod_level();
  }
};

class VarType {
 public:
  framework::proto::VarType::Type type;
  variant<LoDTensorDesc, TensorDesc> desc;

  void Parse(const framework::proto::VarType& proto);
};

class VarDesc {
 public:
  void Parse(const framework::proto::VarDesc& desc);

  std::string name;
  VarType type;
  bool persistable{false};
};

class OpDesc {
 public:
  void Parse(const framework::proto::OpDesc& desc);

  std::string op_type;
  std::map<std::string, std::vector<std::string>> inputs;
  std::map<std::string, std::vector<std::string>> outputs;
  std::map<std::string, variant<int, float, std::string>> attrs;
};

class BlockDesc {
 public:
  void Parse(const framework::proto::BlockDesc& desc);

  int idx{-1};
  int parent_idx{-1};
  int forward_block_idx{-1};
  std::map<std::string, VarDesc> vars;
  std::vector<OpDesc> ops;
};

class ProgramDesc {
 public:
  void Parse(const framework::proto::ProgramDesc& desc);

  BlockDesc block;
};

}  // namespace lite
}  // namespace paddle
