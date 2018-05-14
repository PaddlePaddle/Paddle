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

#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/platform/place.h"

int main() {
  const auto cpu_place = paddle::platform::CPUPlace();
  std::cout << std::type_index(typeid(cpu_place)).name() << std::endl;
  //  auto executor = paddle::framework::Executor(cpu_place);
  paddle::framework::Executor executor(cpu_place);
  paddle::framework::Scope scope;
  std::cout << std::type_index(typeid(scope)).name() << std::endl;
  return 0;
}
