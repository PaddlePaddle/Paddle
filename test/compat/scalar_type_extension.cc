// Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

#include <torch/extension.h>

int64_t scalar_type_value(c10::ScalarType dtype) {
  return static_cast<int64_t>(dtype);
}

c10::ScalarType scalar_type_round_trip(c10::ScalarType dtype) { return dtype; }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("scalar_type_value", &scalar_type_value);
  m.def("scalar_type_round_trip", &scalar_type_round_trip);
}
