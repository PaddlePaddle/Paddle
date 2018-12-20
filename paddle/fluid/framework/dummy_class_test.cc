// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/dummy_class.h"
#include <glog/logging.h>
#include <gtest/gtest.h>
#include <string>
#include "paddle/fluid/framework/dummy_class_type_index.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

TEST(test, type) {
  auto &t1 = GetDummyClassTypeIndex();
  std::type_index t2(typeid(DummyClass *));
  PADDLE_ENFORCE(std::string(t1.name()) == t2.name(),
                 "DummyClass Test: not same name");
  if (t1.name() != t2.name()) {
    LOG(WARNING) << "DummyClass Test: not same address";
  }
  PADDLE_ENFORCE(t1 == t2, "DummyClass Test: not same type %s %s", t1.name(),
                 t2.name());
}

}  // namespace framework
}  // namespace paddle
