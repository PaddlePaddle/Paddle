/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <gtest/gtest.h>
#define private public
#include "paddle/framework/op_registry.h"

USE_OP(add_two);
// USE_OP(add_two_grad);

TEST(AddOp, GetOpProto) {
  auto& protos = paddle::framework::OpRegistry::protos();
  auto it = protos.find("add_two");
  ASSERT_NE(it, protos.end());
  auto& op_creators = paddle::framework::OpRegistry::op_creators();
  auto it1 = op_creators.find("add_two_grad");
  ASSERT_NE(it1, op_creators.end());
}
