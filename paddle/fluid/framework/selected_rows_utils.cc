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

#include "paddle/fluid/framework/selected_rows_utils.h"

#include "paddle/phi/core/serialization.h"

namespace paddle {
namespace framework {

void SerializeToStream(std::ostream& os,
                       const phi::SelectedRows& selected_rows,
                       const platform::DeviceContext& dev_ctx) {
  phi::SerializeToStream(os, selected_rows, dev_ctx);
}

void SerializeToStream(std::ostream& os,
                       const phi::SelectedRows& selected_rows) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext* dev_ctx;
  auto place = selected_rows.place();
  dev_ctx = pool.Get(place);
  phi::SerializeToStream(os, selected_rows, *dev_ctx);
}

void DeserializeFromStream(std::istream& is, phi::SelectedRows* selected_rows) {
  platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
  const platform::DeviceContext* dev_ctx;
  dev_ctx = pool.Get(platform::CPUPlace());
  phi::DeserializeFromStream(is, selected_rows, *dev_ctx);
}

void DeserializeFromStream(std::istream& is,
                           phi::SelectedRows* selected_rows,
                           const platform::DeviceContext& dev_ctx) {
  phi::DeserializeFromStream(is, selected_rows, dev_ctx);
}

}  // namespace framework
}  // namespace paddle
