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

void VarDesc::SetDataType(proto::DataType data_type) {
  mutable_tensor_desc()->set_data_type(data_type);
}

std::vector<int64_t> VarDesc::Shape() const {
  return RepeatedToVector(tensor_desc().dims());
}

proto::DataType VarDesc::GetDataType() const {
  return tensor_desc().data_type();
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
      PADDLE_THROW("Tensor type=%d does not support LoDLevel",
                   desc_.tensor_array().lod_level());
  }
}

int32_t VarDesc::GetLodLevel() const {
  switch (desc_.type()) {
    case proto::VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().lod_level();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.tensor_array().lod_level();
    default:
      PADDLE_THROW("Tensor type=%d does not support LoDLevel",
                   desc_.tensor_array().lod_level());
  }
}

const proto::TensorDesc &VarDesc::tensor_desc() const {
  PADDLE_ENFORCE(desc_.has_type(), "invoke TensorDesc must after set type");
  switch (desc_.type()) {
    case proto::VarDesc::SELECTED_ROWS:
      return desc_.selected_rows();
    case proto::VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().tensor();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.tensor_array().tensor();
    default:
      PADDLE_THROW("Unexpected branch.");
  }
}

proto::TensorDesc *VarDesc::mutable_tensor_desc() {
  PADDLE_ENFORCE(desc_.has_type(),
                 "invoke MutableTensorDesc must after set type");
  switch (desc_.type()) {
    case proto::VarDesc::SELECTED_ROWS:
      return desc_.mutable_selected_rows();
    case proto::VarDesc::LOD_TENSOR:
      return desc_.mutable_lod_tensor()->mutable_tensor();
    case proto::VarDesc::LOD_TENSOR_ARRAY:
      return desc_.mutable_tensor_array()->mutable_tensor();
    default:
      PADDLE_THROW("Unexpected branch.");
  }
}
}  // namespace framework
}  // namespace paddle
