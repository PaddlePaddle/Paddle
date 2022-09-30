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

#include "paddle/phi/capi/include/c_int_array.h"

#include "paddle/phi/capi/include/common.h"
#include "paddle/phi/common/int_array.h"

PD_List PD_IntArrayGetDataPointer(PD_IntArray* int_array) {
  auto cc_int_array = reinterpret_cast<phi::IntArray*>(int_array);
  const auto& data = cc_int_array->GetData();
  PD_List list;
  list.size = data.size();
  list.data = const_cast<int64_t*>(data.data());
  return list;
}

size_t PD_IntArrayGetElementCount(PD_IntArray* int_array) {
  auto cc_int_array = reinterpret_cast<phi::IntArray*>(int_array);
  return cc_int_array->size();
}

size_t PD_IntArrayGetSize(PD_IntArray* int_array) {
  auto cc_int_array = reinterpret_cast<phi::IntArray*>(int_array);
  return cc_int_array->size();
}

PD_REGISTER_CAPI(int_array);
