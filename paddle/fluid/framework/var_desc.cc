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

#include "paddle/fluid/framework/var_desc.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

proto::VarType::Type VarDesc::GetType() const { return desc_.type().type(); }

void VarDesc::SetType(proto::VarType::Type type) {
  desc_.mutable_type()->set_type(type);
}

void VarDesc::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
}

void VarDesc::SetTensorDescNum(size_t num) {
  switch (desc_.type().type()) {
    case proto::VarType::READER: {
      auto *lod_tensors_ptr =
          desc_.mutable_type()->mutable_reader()->mutable_lod_tensor();
      lod_tensors_ptr->Clear();
      for (size_t i = 0; i < num; ++i) {
        lod_tensors_ptr->Add();
      }
      return;
    } break;
    default:
      PADDLE_THROW(
          "Setting 'sub_tensor_number' is not supported by the type of var %s.",
          this->Name());
  }
}

size_t VarDesc::GetTensorDescNum() const {
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      return desc_.type().reader().lod_tensor_size();
      break;
    default:
      PADDLE_THROW(
          "Getting 'sub_tensor_number' is not supported by the type of var %s.",
          this->Name());
  }
}

void VarDesc::SetShapes(
    const std::vector<std::vector<int64_t>> &multiple_dims) {
  if (multiple_dims.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given shapes(" << multiple_dims.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_dims.size());
  }
  std::vector<proto::VarType::TensorDesc *> tensors = mutable_tensor_descs();
  for (size_t i = 0; i < multiple_dims.size(); ++i) {
    VectorToRepeated(multiple_dims[i], tensors[i]->mutable_dims());
  }
}

std::vector<int64_t> VarDesc::GetShape() const {
  return RepeatedToVector(tensor_desc().dims());
}

std::vector<std::vector<int64_t>> VarDesc::GetShapes() const {
  std::vector<proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<std::vector<int64_t>> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(RepeatedToVector(tensor_desc.dims()));
  }
  return res;
}

void VarDesc::SetDataType(proto::VarType::Type data_type) {
  switch (desc_.type().type()) {
    case proto::VarType::CHANNEL:
      mutable_channel_desc()->set_data_type(data_type);
      break;
    default:
      mutable_tensor_desc()->set_data_type(data_type);
  }
}

void VarDesc::SetDataTypes(
    const std::vector<proto::VarType::Type> &multiple_data_type) {
  if (multiple_data_type.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given data types("
            << multiple_data_type.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_data_type.size());
  }
  std::vector<proto::VarType::TensorDesc *> tensor_descs =
      mutable_tensor_descs();
  for (size_t i = 0; i < multiple_data_type.size(); ++i) {
    tensor_descs[i]->set_data_type(multiple_data_type[i]);
  }
}

proto::VarType::Type VarDesc::GetDataType() const {
  switch (desc_.type().type()) {
    case proto::VarType::CHANNEL:
      return channel_desc().data_type();
      break;
    default:
      return tensor_desc().data_type();
  }
}

std::vector<proto::VarType::Type> VarDesc::GetDataTypes() const {
  std::vector<proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<proto::VarType::Type> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(tensor_desc.data_type());
  }
  return res;
}

void VarDesc::SetCapacity(int64_t capacity) {
  switch (desc_.type().type()) {
    case proto::VarType::CHANNEL:
      desc_.mutable_type()->mutable_channel()->set_capacity(capacity);
      break;
    default:
      PADDLE_THROW("Setting 'capacity' is not supported by the type of var %s.",
                   this->Name());
  }
}

void VarDesc::SetLoDLevel(int32_t lod_level) {
  switch (desc_.type().type()) {
    case proto::VarType::LOD_TENSOR:
      desc_.mutable_type()->mutable_lod_tensor()->set_lod_level(lod_level);
      break;
    case proto::VarType::LOD_TENSOR_ARRAY:
      desc_.mutable_type()->mutable_tensor_array()->set_lod_level(lod_level);
      break;
    default:
      PADDLE_THROW(
          "Setting 'lod_level' is not supported by the type of var %s.",
          this->Name());
  }
}

void VarDesc::SetLoDLevels(const std::vector<int32_t> &multiple_lod_level) {
  if (multiple_lod_level.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given lod_levels("
            << multiple_lod_level.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_lod_level.size());
  }
  switch (desc_.type().type()) {
    case proto::VarType::READER: {
      size_t i = 0;
      for (auto &lod_tensor :
           *desc_.mutable_type()->mutable_reader()->mutable_lod_tensor()) {
        lod_tensor.set_lod_level(multiple_lod_level[i++]);
      }
    } break;
    default:
      PADDLE_THROW(
          "Setting 'lod_levels' is not supported by the type of var %s.",
          this->Name());
  }
}

int32_t VarDesc::GetLoDLevel() const {
  switch (desc_.type().type()) {
    case proto::VarType::LOD_TENSOR:
      return desc_.type().lod_tensor().lod_level();
    case proto::VarType::LOD_TENSOR_ARRAY:
      return desc_.type().tensor_array().lod_level();
    default:
      PADDLE_THROW(
          "Getting 'lod_level' is not supported by the type of var %s.",
          this->Name());
  }
}

std::vector<int32_t> VarDesc::GetLoDLevels() const {
  std::vector<int32_t> res;
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      res.reserve(desc_.type().reader().lod_tensor_size());
      for (auto &lod_tensor : desc_.type().reader().lod_tensor()) {
        res.push_back(lod_tensor.lod_level());
      }
      return res;
      break;
    default:
      PADDLE_THROW(
          "Getting 'lod_levels' is not supported by the type of var %s.",
          this->Name());
  }
}

const proto::VarType::ChannelDesc &VarDesc::channel_desc() const {
  PADDLE_ENFORCE(desc_.has_type(), "The var's type hasn't been set.");
  PADDLE_ENFORCE(desc_.type().has_type(), "The var type hasn't been set.");
  switch (desc_.type().type()) {
    case proto::VarType::CHANNEL:
      return desc_.type().channel();
    default:
      PADDLE_THROW(
          "Getting 'channel_desc' is not supported by the type of var %s.",
          this->Name());
  }
}

const proto::VarType::TensorDesc &VarDesc::tensor_desc() const {
  PADDLE_ENFORCE(desc_.has_type(), "The var's type hasn't been set.");
  PADDLE_ENFORCE(desc_.type().has_type(), "The var type hasn't been set.");
  switch (desc_.type().type()) {
    case proto::VarType::SELECTED_ROWS:
      return desc_.type().selected_rows();
    case proto::VarType::LOD_TENSOR:
      return desc_.type().lod_tensor().tensor();
    case proto::VarType::LOD_TENSOR_ARRAY:
      return desc_.type().tensor_array().tensor();
    default:
      PADDLE_THROW(
          "Getting 'tensor_desc' is not supported by the type of var %s.",
          this->Name());
  }
}

std::vector<proto::VarType::TensorDesc> VarDesc::tensor_descs() const {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  std::vector<proto::VarType::TensorDesc> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      for (const auto &lod_tensor : desc_.type().reader().lod_tensor()) {
        res.push_back(lod_tensor.tensor());
      }
      return res;
    default:
      PADDLE_THROW(
          "Getting 'tensor_descs' is not supported by the type of var "
          "%s.",
          this->Name());
  }
}

proto::VarType::ChannelDesc *VarDesc::mutable_channel_desc() {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  PADDLE_ENFORCE(desc_.type().has_type(), "The var type hasn't been set.");
  switch (desc_.type().type()) {
    case proto::VarType::CHANNEL:
      return desc_.mutable_type()->mutable_channel();
    default:
      PADDLE_THROW(
          "Getting 'mutable_channel_desc' is not supported by the type of var "
          "%s.",
          this->Name());
  }
}

proto::VarType::TensorDesc *VarDesc::mutable_tensor_desc() {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  PADDLE_ENFORCE(desc_.type().has_type(), "The var type hasn't been set.");
  switch (desc_.type().type()) {
    case proto::VarType::SELECTED_ROWS:
      return desc_.mutable_type()->mutable_selected_rows();
    case proto::VarType::LOD_TENSOR:
      return desc_.mutable_type()->mutable_lod_tensor()->mutable_tensor();
    case proto::VarType::LOD_TENSOR_ARRAY:
      return desc_.mutable_type()->mutable_tensor_array()->mutable_tensor();
    default:
      PADDLE_THROW(
          "Getting 'mutable_tensor_desc' is not supported by the type of var "
          "%s.",
          this->Name());
  }
}

std::vector<proto::VarType::TensorDesc *> VarDesc::mutable_tensor_descs() {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  PADDLE_ENFORCE(desc_.type().has_type(), "The var type hasn't been set.");
  std::vector<proto::VarType::TensorDesc *> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type().type()) {
    case proto::VarType::READER:
      for (auto &lod_tensor :
           *desc_.mutable_type()->mutable_reader()->mutable_lod_tensor()) {
        res.push_back(lod_tensor.mutable_tensor());
      }
      return res;
    default:
      PADDLE_THROW(
          "Getting 'tensor_descs' is not supported by the type of var "
          "%s.",
          this->Name());
  }
}

}  // namespace framework
}  // namespace paddle
