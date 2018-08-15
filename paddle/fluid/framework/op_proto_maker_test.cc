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

class TestInplaceProtoMaker : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "input of test op");
    AddOutput("XOut", "output of test op").Reuse("X");
  }
};

class TestInplaceProtoMaker2
    : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "input of test op");
    AddOutput("XOut", "output of test op").Reuse("X");
    AddOutput("NoOut", "output of test op").Reuse("NotExists");
  }
};

TEST(ProtoMaker, InplaceOutput) {
  paddle::framework::proto::OpProto op_proto, op_proto2;
  paddle::framework::OpAttrChecker op_checker;
  TestInplaceProtoMaker proto_maker;
  TestInplaceProtoMaker2 proto_maker2;

  proto_maker(&op_proto, &op_checker);

  ASSERT_THROW(proto_maker2(&op_proto2, &op_checker),
               paddle::platform::EnforceNotMet);
}

// normal reuse
class TestReuseProtoMaker : public paddle::framework::OpProtoAndCheckerMaker {
 public:
  void Make() {
    AddInput("X", "input of test op");
    AddInput("Y", "input of test op");
    AddOutput("Out", "output of test op");
    AddOutput("XOut", "output of test op");
    // avoid destructor exception.
    // Validate();
    TestReuse();
  }

  virtual void TestReuse() {}
};

// test duplicate reuse error
class TestReuseProtoMaker2 : public TestReuseProtoMaker {
 public:
  void TestReuse() {
    Reuse("Out", "X");
    Reuse("Out", "Y");
  }
};

// NotExists Input
class TestReuseProtoMaker3 : public TestReuseProtoMaker {
 public:
  void TestReuse() {
    Reuse("Out", "NotExists");
    Reuse("XOut", "X");
  }
};

// NotExists Output
class TestReuseProtoMaker4 : public TestReuseProtoMaker {
 public:
  void TestReuse() { Reuse("NotExists", "X"); }
};

TEST(ProtoMaker, Reuse) {
  paddle::framework::proto::OpProto op_proto;
  paddle::framework::OpAttrChecker op_checker;
  TestReuseProtoMaker proto_maker;
  proto_maker(&op_proto, &op_checker);
}

// NOTE(dzhwinter):
// There is a Fatal CHECK on base class destructor, which will call abort inside
// instead of
// throw an exception. If we throw an exception in Make(), we will trigger the
// CHECK and terminate the tests.
//
// I had tried to replace the default CHECK with a exception, however, it's
// still not supported by glog.
// the details:
// https://github.com/google/glog/issues/249
// https://github.com/facebookresearch/TensorComprehensions/issues/351
/*
TEST(ProtoMaker, ReuseWithException) {
  paddle::framework::proto::OpProto op_proto2, op_proto3, op_proto4;
  paddle::framework::OpAttrChecker op_checker;
  TestReuseProtoMaker2 proto_maker2;
  TestReuseProtoMaker3 proto_maker3;
  TestReuseProtoMaker4 proto_maker4;
  EXPECT_THROW(proto_maker2(&op_proto2, &op_checker),
               paddle::platform::EnforceNotMet);

  EXPECT_THROW(proto_maker3(&op_proto3, &op_checker),
               paddle::platform::EnforceNotMet);

  EXPECT_THROW(proto_maker4(&op_proto4, &op_checker),
               paddle::platform::EnforceNotMet);
}

void FailureFunction() {
  throw std::runtime_error("Check failed in destructor.");
  // return 0;
}

int main(int argc, char** argv) {
  testing::InitGoogleTest(&argc, argv);
  google::InstallFailureFunction(&FailureFunction);
  return RUN_ALL_TESTS();
}
*/
