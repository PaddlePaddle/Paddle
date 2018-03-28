//  Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "serialization.h"
#include "gtest/gtest.h"

TEST(TensorToProto, Case1) {
  paddle::optimizer::Tensor t(3), t1(3);
  for (size_t i = 0; i < t.size(); ++i) {
    t[i] = i;
    t1[i] = 10;
  }

  paddle::TensorProto proto;
  paddle::optimizer::TensorToProto(t, &proto);
  paddle::optimizer::ProtoToTensor(proto, &t1);
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], t[i]);
  }
}

TEST(TensorToProto, Case2) {
  paddle::optimizer::Tensor t(1), t1(1);
  for (size_t i = 0; i < t.size(); ++i) {
    t[i] = i;
    t1[i] = 10;
  }

  paddle::TensorProto proto;
  paddle::optimizer::TensorToProto(t, &proto);
  paddle::optimizer::ProtoToTensor(proto, &t1);
  for (size_t i = 0; i < t1.size(); ++i) {
    EXPECT_EQ(t1[i], t[i]);
  }
}
