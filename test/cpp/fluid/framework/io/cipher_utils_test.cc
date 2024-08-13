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

#include "paddle/fluid/framework/io/crypto/cipher_utils.h"

#include <gtest/gtest.h>

#include <fstream>
#include <string>

namespace paddle {
namespace framework {

TEST(CipherUtils, load_config) {
  std::string filename("cryptotest_config_file.conf");

  std::ofstream fout(filename, std::ios::out);
  fout << "# annotation test line:"
          " must have two space along ':'."
       << std::endl;
  std::vector<std::string> key_value;
  key_value.emplace_back("key_str : ciphername");
  key_value.emplace_back("key_int : 1");
  key_value.emplace_back("key_bool : true");
  key_value.emplace_back("key_bool1 : false");
  key_value.emplace_back("key_bool2 : 0");
  for (auto& i : key_value) {
    fout << i << std::endl;
  }
  fout.close();

  auto config = CipherUtils::LoadConfig(filename);

  std::string out_str;
  EXPECT_TRUE(CipherUtils::GetValue<std::string>(config, "key_str", &out_str));
  EXPECT_EQ(out_str, std::string("ciphername"));

  int out_int = 0;
  EXPECT_TRUE(CipherUtils::GetValue<int>(config, "key_int", &out_int));
  EXPECT_EQ(out_int, 1);

  bool out_bool = false;
  EXPECT_TRUE(CipherUtils::GetValue<bool>(config, "key_bool", &out_bool));
  EXPECT_EQ(out_bool, true);

  bool out_bool1 = false;
  EXPECT_TRUE(CipherUtils::GetValue<bool>(config, "key_bool1", &out_bool1));
  EXPECT_EQ(out_bool1, false);

  bool out_bool2 = false;
  EXPECT_TRUE(CipherUtils::GetValue<bool>(config, "key_bool2", &out_bool2));
  EXPECT_EQ(out_bool2, false);
}

TEST(CipherUtils, gen_key) {
  std::string filename("test_keyfile");
  std::string key = CipherUtils::GenKey(256);
  std::string key1 = CipherUtils::GenKeyToFile(256, filename);
  EXPECT_NE(key, key1);
  std::string key2 = CipherUtils::ReadKeyFromFile(filename);
  EXPECT_EQ(key1, key2);
  EXPECT_EQ(static_cast<int>(key.size()), 32);
}

}  // namespace framework
}  // namespace paddle
