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

#include "paddle/fluid/lite/model_parser/runtime.h"

namespace paddle {
namespace lite {

void VarDesc::Parse(const framework::proto::VarDesc& desc) {
  name = desc.name();
  this->persistable = desc.persistable();
  type.Parse(desc.type());
}

void OpDesc::Parse(const framework::proto::OpDesc& desc) {
  op_type = desc.type();
  // prepare inputs
  for (const auto& input : desc.inputs()) {
    for (const auto& arg : input.arguments()) {
      inputs[input.parameter()].push_back(arg);
    }
  }

  // prepare outputs
  for (const auto& output : desc.inputs()) {
    for (const auto& arg : output.arguments()) {
      inputs[output.parameter()].push_back(arg);
    }
  }

  // prepare attributes
  for (const auto& attr : desc.attrs()) {
    switch (static_cast<int>(attr.type())) {
      case framework::proto::AttrType::INT:
        attrs[attr.name()] = attr.i();
        break;
      case framework::proto::AttrType::FLOAT:
        attrs[attr.name()] = attr.f();
        break;
      case framework::proto::AttrType::STRING:
        attrs[attr.name()] = attr.s();
        break;
      case framework::proto::AttrType::INTS:
        attrs[attr.name()] = attr.ints();
        break;
      case framework::proto::AttrType::FLOATS:
        attrs[attr.name()] = attr.floats();
        break;
      case framework::proto::AttrType::STRINGS:
        attrs[attr.name()] = attr.strings();
        break;
      case framework::proto::AttrType::BOOLEAN:
        attrs[attr.name()] = attr.b();
        break;
      case framework::proto::AttrType::BOOLEANS:
        attrs[attr.name()] = attr.bools();
        break;
      case framework::proto::AttrType::LONG:
        attrs[attr.name()] = attr.l();
        break;
      case framework::proto::AttrType::LONGS:
        attrs[attr.name()] = attr.longs();
        break;
      case framework::proto::AttrType::BLOCK:
        attrs[attr.name()] = attr.block_idx();
        break;
      case framework::proto::AttrType::BLOCKS:
        attrs[attr.name()] = attr.blocks_idx();
        break;
      default:
        LOG(ERROR) << "unknown attribute type found";
    }
  }
}

void BlockDesc::Parse(const framework::proto::BlockDesc& desc) {
  idx = desc.idx();
  parent_idx = desc.parent_idx();
}

void VarType::Parse(const framework::proto::VarType& proto) {
  switch (static_cast<int>(proto.type())) {
    case framework::proto::VarType_Type::VarType_Type_LOD_TENSOR:
      desc = LoDTensorDesc(proto.lod_tensor());
      break;

    case framework::proto::VarType_Type::VarType_Type_LOD_TENSOR_ARRAY:
      desc = LoDTensorArrayDesc(proto.tensor_array());
      break;

    default:
      LOG(ERROR) << "no valid var type found";
      return;
  }
}

}  // namespace lite
}  // namespace paddle
