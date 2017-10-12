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

void VarDescBind::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, mutable_tensor_desc()->mutable_dims());
}

void VarDescBind::SetDataType(DataType data_type) {
  mutable_tensor_desc()->set_data_type(data_type);
}

std::vector<int64_t> VarDescBind::Shape() const {
  return RepeatedToVector(tensor_desc().dims());
}

DataType VarDescBind::GetDataType() const { return tensor_desc().data_type(); }

void VarDescBind::SetLoDLevel(int32_t lod_level) {
  PADDLE_ENFORCE(desc_.type() == VarDesc::LOD_TENSOR);
  desc_.mutable_lod_tensor()->set_lod_level(lod_level);
}

int32_t VarDescBind::GetLodLevel() const {
  PADDLE_ENFORCE(desc_.type() == VarDesc::LOD_TENSOR);
  return desc_.lod_tensor().lod_level();
}

const TensorDesc &VarDescBind::tensor_desc() const {
  PADDLE_ENFORCE(desc_.has_type(), "invoke TensorDesc must after set type");
  switch (desc_.type()) {
    case VarDesc::SELECTED_ROWS:
      return desc_.selected_rows();
    case VarDesc::LOD_TENSOR:
      return desc_.lod_tensor().tensor();
  }
}

TensorDesc *VarDescBind::mutable_tensor_desc() {
  PADDLE_ENFORCE(desc_.has_type(),
                 "invoke MutableTensorDesc must after set type");
  switch (desc_.type()) {
    case VarDesc::SELECTED_ROWS:
      return desc_.mutable_selected_rows();
    case VarDesc::LOD_TENSOR:
      return desc_.mutable_lod_tensor()->mutable_tensor();
  }
}
}  // namespace framework
}  // namespace paddle
