/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/framework/var_desc.h"
#include "paddle/platform/enforce.h"

namespace paddle {
namespace framework {

proto::VarDesc::VarType VarDesc::GetType() const { return desc_.type(); }

void VarDesc::SetType(proto::VarDesc::VarType type) { desc_.set_type(type); }

void VarDesc::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
}

void VarDesc::SetTensorDescNum(size_t num) {
  switch (desc_.type()) {
    case proto::VarDesc::READER: {
      auto *lod_tensors_ptr = desc_.mutable_reader()->mutable_lod_tensor();
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
  switch (desc_.type()) {
    case proto::VarDesc::READER:
      return desc_.reader().lod_tensor_size();
      break;
    default:
      PADDLE_THROW(
          "Getting 'sub_tensor_number' is not supported by the type of var %s.",
          this->Name());
  }
}

void VarDesc::SetShapes(
    const std::vector<std::vector<int64_t>> &multiple_dims) {
  PADDLE_ENFORCE_EQ(multiple_dims.size(), GetTensorDescNum(),
                    "The number of given shapes(%d) doesn't equal to the "
                    "number of sub tensor.",
                    multiple_dims.size(), GetTensorDescNum());
  std::vector<proto::TensorDesc *> tensors = mutable_tensor_descs();
  for (size_t i = 0; i < multiple_dims.size(); ++i) {
    VectorToRepeated(multiple_dims[i], tensors[i]->mutable_dims());
  }
}

std::vector<int64_t> VarDesc::GetShape() const {
  return RepeatedToVector(tensor_desc().dims());
}

std::vector<std::vector<int64_t>> VarDesc::GetShapes() const {
  std::vector<proto::TensorDesc> descs = tensor_descs();
  std::vector<std::vector<int64_t>> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(RepeatedToVector(tensor_desc.dims()));
  }
  return res;
}

void VarDesc::SetDataType(proto::DataType data_type) {
  mutable_tensor_desc()->set_data_type(data_type);
}

void VarDesc::SetDataTypes(
    const std::vector<proto::DataType> &multiple_data_type) {
  PADDLE_ENFORCE_EQ(multiple_data_type.size(), GetTensorDescNum(),
                    "The number of given data types(%d) doesn't equal to the "
                    "number of sub tensor.",
                    multiple_data_type.size(), GetTensorDescNum());
  std::vector<proto::TensorDesc *> tensor_descs = mutable_tensor_descs();
  for (size_t i = 0; i < multiple_data_type.size(); ++i) {
    tensor_descs[i]->set_data_type(multiple_data_type[i]);
  }
}

proto::DataType VarDesc::GetDataType() const {
  return tensor_desc().data_type();
}

std::vector<proto::DataType> VarDesc::GetDataTypes() const {
  std::vector<proto::TensorDesc> descs = tensor_descs();
  std::vector<proto::DataType> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(tensor_desc.data_type());
  }
  return res;
}

void VarDesc::SetLoDLevel(int32_t lod_level) {
  switch (desc_.type()) {
    case proto::VarDesc::LOD_TENSOR:
      desc_.mutable_lod_tensor()->set_lod_level(lod_level);
      break;
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      desc_.mutable_tensor_array()->set_lod_level(lod_level);
      break;
    default:
      PADDLE_THROW(
          "Setting 'lod_level' is not supported by the type of var %s.",
          this->Name());
  }
}

void VarDesc::SetLoDLevels(const std::vector<int32_t> &multiple_lod_level) {
  PADDLE_ENFORCE_EQ(multiple_lod_level.size(), GetTensorDescNum(),
                    "The number of given data types(%d) doesn't equal to the "
                    "number of sub tensor.",
                    multiple_lod_level.size(), GetTensorDescNum());
  switch (desc_.type()) {
    case proto::VarDesc::READER: {
      size_t i = 0;
      for (auto &lod_tensor : *desc_.mutable_reader()->mutable_lod_tensor()) {
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
  switch (desc_.type()) {
    case proto::VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().lod_level();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.tensor_array().lod_level();
    default:
      PADDLE_THROW(
          "Getting 'lod_level' is not supported by the type of var %s.",
          this->Name());
  }
}

std::vector<int32_t> VarDesc::GetLoDLevels() const {
  std::vector<int32_t> res;
  switch (desc_.type()) {
    case proto::VarDesc::READER:
      res.reserve(desc_.reader().lod_tensor_size());
      for (auto &lod_tensor : desc_.reader().lod_tensor()) {
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

const proto::TensorDesc &VarDesc::tensor_desc() const {
  PADDLE_ENFORCE(desc_.has_type(), "The var's type hasn't been set.");
  switch (desc_.type()) {
    case proto::VarDesc::SELECTED_ROWS:
      return desc_.selected_rows();
    case proto::VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().tensor();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.tensor_array().tensor();
    default:
      PADDLE_THROW(
          "Getting 'tensor_desc' is not supported by the type of var %s.",
          this->Name());
  }
}

std::vector<proto::TensorDesc> VarDesc::tensor_descs() const {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  std::vector<proto::TensorDesc> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type()) {
    case proto::VarDesc::READER:
      for (const auto &lod_tensor : desc_.reader().lod_tensor()) {
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

proto::TensorDesc *VarDesc::mutable_tensor_desc() {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  switch (desc_.type()) {
    case proto::VarDesc::SELECTED_ROWS:
      return desc_.mutable_selected_rows();
    case proto::VarDesc::LOD_TENSOR:
      return desc_.mutable_lod_tensor()->mutable_tensor();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.mutable_tensor_array()->mutable_tensor();
    default:
      PADDLE_THROW(
          "Getting 'mutable_tensor_desc' is not supported by the type of var "
          "%s.",
          this->Name());
  }
}

std::vector<proto::TensorDesc *> VarDesc::mutable_tensor_descs() {
  PADDLE_ENFORCE(desc_.has_type(), "The var type hasn't been set.");
  std::vector<proto::TensorDesc *> res;
  res.reserve(GetTensorDescNum());
  switch (desc_.type()) {
    case proto::VarDesc::READER:
      for (auto &lod_tensor : *desc_.mutable_reader()->mutable_lod_tensor()) {
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
