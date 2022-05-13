/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/op_proto_maker.h"

#include "gtest/gtest-message.h"
#include "gtest/gtest-test-part.h"
#include "gtest/gtest.h"

class TestAttrProtoMaker : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddAttr<float>("scale", "scale of test op");
    AddAttr<float>("scale", "scale of test op");
  }
};

TEST(ProtoMaker, DuplicatedAttr) {
  paddle::framework::proto::OpProto op_proto;
  paddle::framework::OpAttrChecker op_checker;
  TestAttrProtoMaker proto_maker;
  ASSERT_THROW(proto_maker(&op_proto, &op_checker),
               paddle::platform::EnforceNotMet);
}

class TestInOutProtoMaker : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("input", "input of test op");
    AddInput("input", "input of test op");
  }
};

TEST(ProtoMaker, DuplicatedInOut) {
  paddle::framework::proto::OpProto op_proto;
  paddle::framework::OpAttrChecker op_checker;
  TestAttrProtoMaker proto_maker;
  ASSERT_THROW(proto_maker(&op_proto, &op_checker),
               paddle::platform::EnforceNotMet);
}
