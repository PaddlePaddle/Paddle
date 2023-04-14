/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */
#pragma once
#ifdef PADDLE_WITH_CUSTOM_DEVICE
#include <paddle/phi/common/data_type.h>
#include <string>
#include <unordered_map>
#include <unordered_set>
namespace phi {
namespace backends {
namespace custom_device {
bool is_in_custom_black_list(const std::string& fluid_op_name);
}  // namespace custom_device
}  // namespace backends
}  // namespace phi
#endif
