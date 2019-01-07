// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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


#include "debug_support.h"

namespace paddle {
namespace platform {

std::once_flag DebugSupport::init_flag_;
std::unique_ptr<DebugSupport> debugSupport_(nullptr);

DebugSupport *DebugSupport::GetInstance() {
  std::call_once(init_flag_,
                 [&]() { debugSupport_.reset(new DebugSupport()); });
  return debugSupport_.get();
}

std::string DebugSupport::getActiveOperator() { return infos[TOperaor]; }

void DebugSupport::setActiveOperator(std::string info) {
  infos.at(TOperaor) = info;
}

}  // namespace platform
}  // namespace paddle
