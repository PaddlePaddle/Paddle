/*
  Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.
  Licensed under the Apache License, Version 2.0 (the "License");
  you may not use this file except in compliance with the License.
  You may obtain a copy of the License at
  http://www.apache.org/licenses/LICENSE-2.0
  Unless required by applicable law or agreed to in writing, software
  distributed under the License is distributed on an "AS IS" BASIS,
  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
  See the License for the specific language governing permissions and
  limitations under the License.
*/

#include <memory>
#include <string>

#include "gtest/gtest.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/variable.h"

TEST(Variable, GetMutable) {
  using paddle::framework::Variable;

  struct Tensor {
    int content_;
  };

  std::unique_ptr<Variable> v(new Variable());

  Tensor* t = v->GetMutable<Tensor>();
  t->content_ = 1234;

  const Tensor& tt = v->Get<Tensor>();
  EXPECT_EQ(1234, tt.content_);

  std::string* s = v->GetMutable<std::string>();
  *s = "hello";

  const std::string& ss = v->Get<std::string>();
  EXPECT_EQ("hello", ss);
}

TEST(Variable, CloneTensorType) {
  namespace f = paddle::framework;
  using f::Variable;
  using f::Tensor;
  using f::LODTensor;

  std::unique_ptr<Variable> v0(new Variable());
  std::unique_ptr<Variable> v1(new Variable());

  // create a LODTensor
  auto t0 = v0->GetMutable<LODTensor>();
  EXPECT_TRUE(v0->IsType<Tensor>());
  EXPECT_TRUE(v0->IsType<LODTensor>());
  t0->mutable_lod()->push_back(std::vector<size_t>{0, 5, 10});
  // let v1 clone it
  v1->CloneTensorType(*v0);
  EXPECT_TRUE(v1->IsType<Tensor>());
  EXPECT_TRUE(v1->IsType<LODTensor>());
}
