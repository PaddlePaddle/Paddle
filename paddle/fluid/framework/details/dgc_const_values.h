//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#pragma once

#include <string>

namespace paddle {
namespace framework {
namespace details {

constexpr char g_dgc_counter_name[] = "__g_dgc_counter__";
constexpr char g_dgc_rampup_begin_step[] = "__g_rampup_begin_step__";
constexpr char g_dgc_nranks[] = "__g_nranks__";
constexpr char g_dgc_k[] = "__dgc_k__";
constexpr char g_dgc_encoded[] = "__dgc_encoded__";
constexpr char g_dgc_gather[] = "__dgc_gather__";

}  // namespace details
}  // namespace framework
}  // namespace paddle
