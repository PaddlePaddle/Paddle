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

#include "paddle/infrt/paddle/pb/var_desc.h"

#include <google/protobuf/map.h>

#include "paddle/infrt/paddle/cpp/desc_api.h"
#include "paddle/infrt/paddle/framework.pb.h"

namespace infrt {
namespace paddle {
namespace pb {

cpp::VarDescAPI::Type VarDesc::GetType() const {
  auto type = desc_->type().type();

#define GET_TYPE_CASE_ITEM(type__)       \
  case framework_proto::VarType::type__: \
    return cpp::VarDescAPI::Type::type__;

  switch (type) {
    GET_TYPE_CASE_ITEM(LOD_TENSOR);
    GET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    GET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    GET_TYPE_CASE_ITEM(SELECTED_ROWS);
    GET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    GET_TYPE_CASE_ITEM(FETCH_LIST);
    GET_TYPE_CASE_ITEM(STEP_SCOPES);
    GET_TYPE_CASE_ITEM(PLACE_LIST);
    GET_TYPE_CASE_ITEM(READER);
    default:
      LOG(FATAL) << "Unknown var type";
      return VarDescAPI::Type();
  }
#undef GET_TYPE_CASE_ITEM
}

void VarDesc::SetType(VarDescAPI::Type type) {
#define SET_TYPE_CASE_ITEM(type__)                                     \
  case VarDescAPI::Type::type__:                                       \
    desc_->mutable_type()->set_type(framework_proto::VarType::type__); \
    break;

  switch (type) {
    SET_TYPE_CASE_ITEM(LOD_TENSOR);
    SET_TYPE_CASE_ITEM(LOD_TENSOR_ARRAY);
    SET_TYPE_CASE_ITEM(LOD_RANK_TABLE);
    SET_TYPE_CASE_ITEM(SELECTED_ROWS);
    SET_TYPE_CASE_ITEM(FEED_MINIBATCH);
    SET_TYPE_CASE_ITEM(FETCH_LIST);
    SET_TYPE_CASE_ITEM(STEP_SCOPES);
    SET_TYPE_CASE_ITEM(PLACE_LIST);
    SET_TYPE_CASE_ITEM(READER);
    default:
      LOG(FATAL) << "Unknown var type";
  }
#undef SET_TYPE_CASE_ITEM
}

void VarDesc::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
}

void VarDesc::SetTensorDescNum(size_t num) {
  switch (desc_->type().type()) {
    case framework_proto::VarType::READER: {
      auto *lod_tensors_ptr =
          desc_->mutable_type()->mutable_reader()->mutable_lod_tensor();
      lod_tensors_ptr->Clear();
      for (size_t i = 0; i < num; ++i) {
        lod_tensors_ptr->Add();
      }
      return;
    } break;
    default:
      LOG(FATAL) << "Setting 'sub_tensor_number' is not supported by the type "
                    "of var %s."
                 << this->Name();
  }
}

size_t VarDesc::GetTensorDescNum() const {
  switch (desc_->type().type()) {
    case framework_proto::VarType::READER:
      return desc_->type().reader().lod_tensor_size();
      break;
    default:
      LOG(FATAL) << "Getting 'sub_tensor_number' is not supported by the type "
                    "of var %s."
                 << this->Name();
  }
  return 0;
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
  std::vector<framework_proto::VarType::TensorDesc *> tensors =
      mutable_tensor_descs();
  for (size_t i = 0; i < multiple_dims.size(); ++i) {
    VectorToRepeated(multiple_dims[i], tensors[i]->mutable_dims());
  }
}

std::vector<int64_t> VarDesc::GetShape() const {
  return RepeatedToVector(tensor_desc().dims());
}

std::vector<std::vector<int64_t>> VarDesc::GetShapes() const {
  std::vector<framework_proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<std::vector<int64_t>> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(RepeatedToVector(tensor_desc.dims()));
  }
  return res;
}

void VarDesc::SetDataType(VarDescAPI::VarDataType data_type) {
#define SET_DATA_TYPE_CASE_ITEM(type__)                                     \
  case cpp::VarDescAPI::Type::type__:                                       \
    mutable_tensor_desc()->set_data_type(framework_proto::VarType::type__); \
    break;

  switch (data_type) {
    SET_DATA_TYPE_CASE_ITEM(BOOL);
    SET_DATA_TYPE_CASE_ITEM(SIZE_T);
    SET_DATA_TYPE_CASE_ITEM(UINT8);
    SET_DATA_TYPE_CASE_ITEM(INT8);
    SET_DATA_TYPE_CASE_ITEM(INT16);
    SET_DATA_TYPE_CASE_ITEM(INT32);
    SET_DATA_TYPE_CASE_ITEM(INT64);
    SET_DATA_TYPE_CASE_ITEM(FP16);
    SET_DATA_TYPE_CASE_ITEM(FP32);
    SET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      LOG(FATAL) << "Unknown var type: " << static_cast<int>(data_type);
  }
#undef SET_DATA_TYPE_CASE_ITEM
}

void VarDesc::SetDataTypes(
    const std::vector<framework_proto::VarType::Type> &multiple_data_type) {
  if (multiple_data_type.size() != GetTensorDescNum()) {
    VLOG(3) << "WARNING: The number of given data types("
            << multiple_data_type.size()
            << ") doesn't match the existing tensor number("
            << GetTensorDescNum()
            << "). The Reader is going to be reinitialized.";
    SetTensorDescNum(multiple_data_type.size());
  }
  std::vector<framework_proto::VarType::TensorDesc *> tensor_descs =
      mutable_tensor_descs();
  for (size_t i = 0; i < multiple_data_type.size(); ++i) {
    tensor_descs[i]->set_data_type(multiple_data_type[i]);
  }
}

// proto::VarType::Type VarDesc::GetDataType() const {
//   return tensor_desc().data_type();
// }
cpp::VarDescAPI::VarDataType VarDesc::GetDataType() const {
  CHECK(desc_->has_type()) << "The var's type hasn't been set.";
  CHECK(desc_->type().has_type()) << "The var type hasn't been set.";
  if (desc_->type().type() != framework_proto::VarType::LOD_TENSOR) {
    return VarDescAPI::Type();
  }
  auto type = tensor_desc().data_type();
#define GET_DATA_TYPE_CASE_ITEM(type__)                       \
  case framework_proto::VarType::Type::VarType_Type_##type__: \
    return VarDescAPI::Type::type__

  switch (type) {
    GET_DATA_TYPE_CASE_ITEM(BOOL);
    GET_DATA_TYPE_CASE_ITEM(SIZE_T);
    GET_DATA_TYPE_CASE_ITEM(UINT8);
    GET_DATA_TYPE_CASE_ITEM(INT8);
    GET_DATA_TYPE_CASE_ITEM(INT16);
    GET_DATA_TYPE_CASE_ITEM(INT32);
    GET_DATA_TYPE_CASE_ITEM(INT64);
    GET_DATA_TYPE_CASE_ITEM(FP16);
    GET_DATA_TYPE_CASE_ITEM(FP32);
    GET_DATA_TYPE_CASE_ITEM(FP64);
    default:
      LOG(FATAL) << "Unknown var type: " << static_cast<int>(type);
      return VarDescAPI::Type();
  }
#undef GET_DATA_TYPE_CASE_ITEM
}

std::vector<framework_proto::VarType::Type> VarDesc::GetDataTypes() const {
  std::vector<framework_proto::VarType::TensorDesc> descs = tensor_descs();
  std::vector<framework_proto::VarType::Type> res;
  res.reserve(descs.size());
  for (const auto &tensor_desc : descs) {
    res.push_back(tensor_desc.data_type());
  }
  return res;
}

void VarDesc::SetLoDLevel(int32_t lod_level) {
  switch (desc_->type().type()) {
    case framework_proto::VarType::LOD_TENSOR:
      desc_->mutable_type()->mutable_lod_tensor()->set_lod_level(lod_level);
      break;
    case framework_proto::VarType::LOD_TENSOR_ARRAY:
      desc_->mutable_type()->mutable_tensor_array()->set_lod_level(lod_level);
      break;
    default:
      LOG(FATAL)
          << "Setting 'lod_level' is not supported by the type of var %s."
          << this->Name();
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
  switch (desc_->type().type()) {
    case framework_proto::VarType::READER: {
      size_t i = 0;
      for (auto &lod_tensor :
           *desc_->mutable_type()->mutable_reader()->mutable_lod_tensor()) {
        lod_tensor.set_lod_level(multiple_lod_level[i++]);
      }
    } break;
    default:
      LOG(FATAL)
          << "Setting 'lod_levels' is not supported by the type of var %s."
          << this->Name();
  }
}

int32_t VarDesc::GetLoDLevel() const {
  switch (desc_->type().type()) {
    case framework_proto::VarType::LOD_TENSOR:
      return desc_->type().lod_tensor().lod_level();
    case framework_proto::VarType::LOD_TENSOR_ARRAY:
      return desc_->type().tensor_array().lod_level();
    default:
      LOG(FATAL)
          << "Getting 'lod_level' is not supported by the type of var %s."
          << this->Name();
  }
  return 0;
}

std::vector<int32_t> VarDesc::GetLoDLevels() const {
  std::vector<int32_t> res;
  switch (desc_->type().type()) {
    case framework_proto::VarType::READER:
      res.reserve(desc_->type().reader().lod_tensor_size());
      for (auto &lod_tensor : desc_->type().reader().lod_tensor()) {
        res.push_back(lod_tensor.lod_level());
      }
      return res;
      break;
    default:
      LOG(FATAL)
          << "Getting 'lod_levels' is not supported by the type of var %s."
          << this->Name();
  }
  return std::vector<int32_t>();
}

const framework_proto::VarType::TensorDesc &VarDesc::tensor_desc() const {
  CHECK(desc_->has_type()) << "The var's type hasn't been set.";
  CHECK(desc_->type().has_type()) << "The var type hasn't been set.";
  switch (desc_->type().type()) {
    case framework_proto::VarType::SELECTED_ROWS:
      return desc_->type().selected_rows();
    case framework_proto::VarType::LOD_TENSOR:
      return desc_->type().lod_tensor().tensor();
    case framework_proto::VarType::LOD_TENSOR_ARRAY:
      return desc_->type().tensor_array().tensor();
    default:
      LOG(FATAL)
          << "Getting 'tensor_desc' is not supported by the type of var %s."
          << this->Name();
  }
  return framework_proto::VarDesc().type().lod_tensor().tensor();
}

std::vector<framework_proto::VarType::TensorDesc> VarDesc::tensor_descs()
    const {
  CHECK(desc_->has_type()) << "The var type hasn't been set.";
  std::vector<framework_proto::VarType::TensorDesc> res;
  res.reserve(GetTensorDescNum());
  switch (desc_->type().type()) {
    case framework_proto::VarType::READER:
      for (const auto &lod_tensor : desc_->type().reader().lod_tensor()) {
        res.push_back(lod_tensor.tensor());
      }
      return res;
    default:
      LOG(FATAL)
          << "Getting 'tensor_descs' is not supported by the type of var "
             "%s."
          << this->Name();
  }
  return std::vector<framework_proto::VarType::TensorDesc>();
}

framework_proto::VarType::TensorDesc *VarDesc::mutable_tensor_desc() {
  CHECK(desc_->has_type()) << "The var type hasn't been set.";
  CHECK(desc_->type().has_type()) << "The var type hasn't been set.";
  switch (desc_->type().type()) {
    case framework_proto::VarType::SELECTED_ROWS:
      return desc_->mutable_type()->mutable_selected_rows();
    case framework_proto::VarType::LOD_TENSOR:
      return desc_->mutable_type()->mutable_lod_tensor()->mutable_tensor();
    case framework_proto::VarType::LOD_TENSOR_ARRAY:
      return desc_->mutable_type()->mutable_tensor_array()->mutable_tensor();
    default:
      LOG(FATAL) << "Getting 'mutable_tensor_desc' is not supported by the "
                    "type of var "
                    "%s."
                 << this->Name();
  }
  return nullptr;
}

std::vector<framework_proto::VarType::TensorDesc *>
VarDesc::mutable_tensor_descs() {
  CHECK(desc_->has_type()) << "The var type hasn't been set.";
  CHECK(desc_->type().has_type()) << "The var type hasn't been set.";
  std::vector<framework_proto::VarType::TensorDesc *> res;
  res.reserve(GetTensorDescNum());
  switch (desc_->type().type()) {
    case framework_proto::VarType::READER:
      for (auto &lod_tensor :
           *desc_->mutable_type()->mutable_reader()->mutable_lod_tensor()) {
        res.push_back(lod_tensor.mutable_tensor());
      }
      return res;
    default:
      LOG(FATAL)
          << "Getting 'tensor_descs' is not supported by the type of var "
             "%s."
          << this->Name();
  }
  return std::vector<framework_proto::VarType::TensorDesc *>();
}

}  // namespace pb
}  // namespace paddle
}  // namespace infrt
