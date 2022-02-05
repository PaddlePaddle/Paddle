/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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

#include <algorithm>
#include <memory>
#include <mutex>  // NOLINT
#include <unordered_map>
#include <utility>
#include <vector>

#include "paddle/pten/core/selected_rows.h"

#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
/*
 * Serialize/Desiralize SelectedRows to std::ostream
 * You can pass ofstream or ostringstream to serilize to file
 * or to a in memory string. GPU tensor will be copied to CPU.
 */
void SerializeToStream(std::ostream& os,
                       const pten::SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx);
void DeserializeFromStream(std::istream& is, pten::SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx);

void SerializeToStream(std::ostream& os,
                       const pten::SelectedRows& selected_rows);

void DeserializeFromStream(std::istream& os, pten::SelectedRows* selected_rows);

}  // namespace framework
}  // namespace paddle
