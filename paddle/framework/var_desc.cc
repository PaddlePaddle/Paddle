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

void VarDescBind::SetVarType(const VarDesc_VarType& var_type) {
  desc_.set_type(var_type);
  var_type_ = var_type;
}

VarDesc_VarType VarDescBind::GetVarType() const { return var_type_; }

void VarDescBind::SetShape(const std::vector<int64_t>& dims) {
  if (var_type_ == VarDesc_VarType_LOD_TENSOR) {
    VectorToRepeated(
        dims, desc_.mutable_lod_tensor()->mutable_tensor()->mutable_dims());
  } else if (var_type_ == VarDesc_VarType_SELECTED_ROWS) {
    VectorToRepeated(dims, desc_.mutable_selected_rows()->mutable_dims());
  }
}

std::vector<int64_t> VarDescBind::GetShape() const {
  if (var_type_ == VarDesc_VarType_LOD_TENSOR) {
    return RepeatedToVector(desc_.lod_tensor().tensor().dims());
  } else if (var_type_ == VarDesc_VarType_SELECTED_ROWS) {
    return RepeatedToVector(desc_.selected_rows().dims());
  } else {
    PADDLE_THROW("Unsupported Variable Type");
  }
}

void VarDescBind::SetDataType(DataType data_type) {
  if (var_type_ == VarDesc_VarType_LOD_TENSOR) {
    desc_.mutable_lod_tensor()->mutable_tensor()->set_data_type(data_type);
  } else if (var_type_ == VarDesc_VarType_SELECTED_ROWS) {
    desc_.mutable_selected_rows()->set_data_type(data_type);
  }
}

DataType VarDescBind::GetDataType() const {
  if (var_type_ == VarDesc_VarType_LOD_TENSOR) {
    return desc_.lod_tensor().tensor().data_type();
  } else if (var_type_ == VarDesc_VarType_SELECTED_ROWS) {
    return desc_.selected_rows().data_type();
  } else {
    PADDLE_THROW("Unsupported Variable Type");
  }
}

void VarDescBind::SetLoDLevel(int32_t lod_level) {
  if (var_type_ == VarDesc_VarType_LOD_TENSOR) {
    desc_.mutable_lod_tensor()->set_lod_level(lod_level);
  }
}

int32_t VarDescBind::GetLodLevel() const {
  if (var_type_ == VarDesc_VarType_LOD_TENSOR) {
    return desc_.lod_tensor().lod_level();
  } else {
    PADDLE_THROW("Can't get lod_level info from non lod type Variable");
  }
}
}  // namespace framework
}  // namespace paddle
