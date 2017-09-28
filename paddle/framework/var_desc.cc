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

namespace paddle {
namespace framework {

void VarDescBind::SetShape(const std::vector<int64_t> &dims) {
  VectorToRepeated(dims, desc_.mutable_lod_tensor()->mutable_dims());
}

void VarDescBind::SetDataType(DataType data_type) {
  desc_.mutable_lod_tensor()->set_data_type(data_type);
}

std::vector<int64_t> VarDescBind::Shape() const {
  return RepeatedToVector(desc_.lod_tensor().dims());
}

DataType VarDescBind::GetDataType() const {
  return desc_.lod_tensor().data_type();
}
}  // namespace framework
}  // namespace paddle
