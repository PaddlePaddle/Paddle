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
#include <gtest/gtest.h>
#include "paddle/fluid/inference/anakin/convert/op_converter.h"
#include "paddle/fluid/inference/anakin/convert/ut_helper.h"

namespace paddle {
namespace inference {
namespace anakin {

void test_pool2d(bool global_pooling, bool ceil_mode,
                 std::string pool_type = "max") {
  auto* pool2d_converter =
      Registry<AnakinOpConverter>::Global().Lookup("pool2d");
  ASSERT_TRUE(pool2d_converter);

  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  AnakinConvertValidation validator(parameters, &scope);

  // The ITensor's Dims should not contain the batch size.
  // So, the ITensor's Dims of input and output should be C * H * W.
  validator.DeclInputVar("pool2d_x", {1, 3, 6, 7});
  if (global_pooling)
    validator.DeclOutputVar("pool2d_out", {1, 3, 1, 1});
  else if (ceil_mode)
    validator.DeclOutputVar("pool2d_out", {1, 3, 3, 4});
  else
    validator.DeclOutputVar("pool2d_out", {1, 3, 3, 3});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"pool2d_x"});
  desc.SetOutput("Out", {"pool2d_out"});

  std::vector<int> ksize({2, 2});
  std::vector<int> strides({2, 2});
  std::vector<int> paddings({0, 0});
  std::string pooling_t = pool_type;

  desc.SetAttr("pooling_type", pooling_t);
  desc.SetAttr("ksize", ksize);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("global_pooling", global_pooling);
  desc.SetAttr("ceil_mode", ceil_mode);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(1);
}

void test_pool2d2(bool global_pooling, bool ceil_mode,
                  std::string pool_type = "max") {
  auto* pool2d_converter =
      Registry<AnakinOpConverter>::Global().Lookup("pool2d");
  ASSERT_TRUE(pool2d_converter);

  framework::Scope scope;
  std::unordered_set<std::string> parameters;
  AnakinConvertValidation validator(parameters, &scope);

  // The ITensor's Dims should not contain the batch size.
  // So, the ITensor's Dims of input and output should be C * H * W.
  validator.DeclInputVar("pool2d_x", {1, 1, 17, 17});
  validator.DeclOutputVar("pool2d_out", {1, 1, 17, 17});

  // Prepare Op description
  framework::OpDesc desc;
  desc.SetType("pool2d");
  desc.SetInput("X", {"pool2d_x"});
  desc.SetOutput("Out", {"pool2d_out"});

  std::vector<int> ksize({3, 3});
  std::vector<int> strides({1, 1});
  std::vector<int> paddings({1, 1});
  std::string pooling_t = pool_type;

  desc.SetAttr("pooling_type", pooling_t);
  desc.SetAttr("ksize", ksize);
  desc.SetAttr("strides", strides);
  desc.SetAttr("paddings", paddings);
  desc.SetAttr("global_pooling", global_pooling);
  desc.SetAttr("ceil_mode", true);

  LOG(INFO) << "set OP";
  validator.SetOp(*desc.Proto());
  LOG(INFO) << "execute";

  validator.Execute(1);
}

TEST(Pool2dOpConverter, normal) { test_pool2d(false, false); }
TEST(Pool2dOpConverter, test_global_pooling) { test_pool2d(true, false); }

TEST(Pool2dOpConverter, max_ceil_test) { test_pool2d(false, true); }
TEST(Pool2dOpConverter, avg_ceil_test) { test_pool2d(false, true, "avg"); }
TEST(Pool2dOpConverter, avg_ceil_test2) { test_pool2d2(false, true, "avg"); }

}  // namespace anakin
}  // namespace inference
}  // namespace paddle

USE_OP(pool2d);
USE_ANAKIN_CONVERTER(pool2d);
