// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/capi/include/c_data_type.h"

#include "paddle/phi/capi/include/common.h"

void PD_DeletePointerList(PD_List list) {
  auto data = reinterpret_cast<void**>(list.data);
  if (data) delete[] data;
}

void PD_DeleteUInt8List(PD_List list) {
  auto data = reinterpret_cast<uint8_t*>(list.data);
  if (data) delete[] data;
}

void PD_DeleteInt64List(PD_List list) {
  auto data = reinterpret_cast<int64_t*>(list.data);
  if (data) delete[] data;
}

void PD_DeleteInt32List(PD_List list) {
  auto data = reinterpret_cast<int32_t*>(list.data);
  delete[] data;
}

void PD_DeleteFloat64List(PD_List list) {
  auto data = reinterpret_cast<double*>(list.data);
  if (data) delete[] data;
}

void PD_DeleteFloat32List(PD_List list) {
  auto data = reinterpret_cast<float*>(list.data);
  if (data) delete[] data;
}

PD_REGISTER_CAPI(data_type);
