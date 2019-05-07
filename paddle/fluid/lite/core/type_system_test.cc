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

#include "paddle/fluid/lite/core/type_system.h"
#include <gtest/gtest.h>

namespace paddle {
namespace lite {

TEST(TypeSystem, test) {
  ASSERT_TRUE(TypeSystem::Global().Contains<lite::TensorBase>());
}

TEST(TypeSystem, register_new) {
  TypeSystem::Global().Register<int>("int32");
  ASSERT_TRUE(TypeSystem::Global().Contains<int>());
  ASSERT_TRUE(TypeSystem::Global().Contains(typeid(int).hash_code()));
  ASSERT_TRUE(TypeSystem::Global().Contains("int32"));
}

}  // namespace lite
}  // namespace paddle
