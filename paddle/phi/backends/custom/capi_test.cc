// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <gtest/gtest.h>

#include <cstring>
#include <string>

#include "paddle/phi/capi/all.h"

#ifndef UNUSED
#define UNUSED __attribute__((unused))
#endif

#include "paddle/phi/capi/capi.h"

TEST(CustomKernel, CAPI) {
  std::string str = "capi";
  EXPECT_EQ(str.data(), PD_StringAttr(&str));

  std::vector<int32_t> int32_vec({1, 2, 3});
  auto int32_list = PD_ListInt32Attr(&int32_vec);
  EXPECT_EQ(int32_list.data, int32_vec.data());
  EXPECT_EQ(int32_list.size, int32_vec.size());

  std::vector<int64_t> int64_vec({1, 2, 3});
  auto int64_list = PD_ListInt64Attr(&int64_vec);
  EXPECT_EQ(int64_list.data, int64_vec.data());
  EXPECT_EQ(int64_list.size, int64_vec.size());

  std::vector<float> float_vec({1, 2, 3});
  auto float_list = PD_ListFloatAttr(&float_vec);
  EXPECT_EQ(float_list.data, float_vec.data());
  EXPECT_EQ(float_list.size, float_vec.size());

  std::vector<double> double_vec({1, 2, 3});
  auto double_list = PD_ListDoubleAttr(&double_vec);
  EXPECT_EQ(double_list.data, double_vec.data());
  EXPECT_EQ(double_list.size, double_vec.size());

  std::vector<std::string> string_vec{"capi", "api"};
  auto string_list = PD_ListStringAttr(&string_vec);
  auto string_data = reinterpret_cast<void**>(string_list.data);
  for (size_t i = 0; i < string_vec.size(); ++i) {
    EXPECT_EQ(string_data[i], string_vec[i].data());
  }

  std::vector<bool> bool_vec{true, false, true};
  auto bool_list = PD_ListBoolAttr(&bool_vec);
  auto bool_data = reinterpret_cast<uint8_t*>(bool_list.data);
  for (size_t i = 0; i < bool_vec.size(); ++i) {
    EXPECT_EQ(bool_data[i], static_cast<uint8_t>(bool_vec[i]));
  }

  std::vector<float*> ptr_vec;
  for (size_t i = 0; i < float_vec.size(); ++i) {
    ptr_vec.push_back(&float_vec[i]);
  }
  auto ptr_list = PD_TensorVectorToList(reinterpret_cast<PD_Tensor*>(&ptr_vec));
  EXPECT_EQ(ptr_list.data, ptr_vec.data());
  EXPECT_EQ(ptr_list.size, ptr_vec.size());
}

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
