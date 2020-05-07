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

#include "paddle/fluid/framework/io/crypt_fstream.h"

#include <cryptopp/aes.h>
#include <cryptopp/osrng.h>
#include <gtest/gtest.h>

#include <cstdio>
#include <fstream>
#include <string>
#include <vector>

namespace paddle {
namespace framework {

TEST(test_fstream_ext, normal_function) {
  const char* input = "hello world";
  const char* filename = "test.txt";
  const char* filename1 = "test1.txt";
  size_t size = strlen(input) + 1;
  std::ofstream fout_normal(filename1, std::ios_base::binary);
  fout_normal.write(input, size);

  CryptOfstream fout(filename, std::ios_base::binary);
  fout.write(input, size);
  fout_normal.close();
  fout.close();

  std::ifstream fin_normal(filename1, std::ios_base::binary);
  char* output = new char[size];
  fin_normal.read(output, size);
  fin_normal.close();

  CryptIfstream fin(filename, std::ios_base::binary);
  char* output1 = new char[size];
  fin.read(output1, size);
  fin.close();

  remove(filename);
  remove(filename1);
  EXPECT_STREQ(output, output1);
}

TEST(test_fstream_ext, security_string) {
  std::string input("hello world");
  const char* filename = "test.txt";
  const char* filename1 = "test1.txt";

  CryptoPP::AutoSeededRandomPool prng;
  const int TAG_SIZE = 12;
  CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
  prng.GenerateBlock(key, sizeof(key));
  size_t size = input.size() + 1;

  std::ofstream fout_normal(filename1, std::ios_base::binary);
  fout_normal.write(input.c_str(), size);
  fout_normal.close();

  CryptOfstream fout(filename, std::ios::out | std::ios_base::binary, true, key,
                     sizeof(key), TAG_SIZE);
  fout.write(input.c_str(), size);
  fout.close();

  std::ifstream fin_normal(filename1, std::ios_base::binary);
  char* output = new char[size];
  fin_normal.read(output, size);
  fin_normal.close();
  CryptIfstream fin(filename, std::ios_base::binary, true, key, sizeof(key),
                    TAG_SIZE);
  char* output1 = new char[size];
  fin.read(output1, size);
  fin.close();

  remove(filename);
  remove(filename1);
  EXPECT_STREQ(output, output1);
}

TEST(test_fstream_ext, security_vector) {
  std::vector<double> input = {1, 2, 3, 4};
  const char* filename = "test.txt";
  const char* filename1 = "test1.txt";

  CryptoPP::AutoSeededRandomPool prng;
  const int TAG_SIZE = 12;
  CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
  prng.GenerateBlock(key, sizeof(key));

  std::ofstream fout_normal(filename1, std::ios_base::binary);
  CryptOfstream fout(filename, std::ios_base::binary, true, key, sizeof(key),
                     TAG_SIZE);
  for (auto& i : input) {
    fout_normal.write(reinterpret_cast<char*>(&i), sizeof(i));
    fout.write(reinterpret_cast<char*>(&i), sizeof(i));
  }
  fout_normal.close();
  fout.close();

  std::ifstream fin_normal(filename1, std::ios_base::binary);
  CryptIfstream fin(filename, std::ios_base::binary, true, key, sizeof(key),
                    TAG_SIZE);

  std::vector<double> output;
  std::vector<double> output1;
  for (size_t i = 0; i < input.size(); i++) {
    double r, r1;
    fin_normal.read(reinterpret_cast<char*>(&r), sizeof(r));
    output.emplace_back(r);
    fin.read(reinterpret_cast<char*>(&r1), sizeof(r1));
    output1.emplace_back(r1);
  }

  fin_normal.close();
  fin.close();
  remove(filename);
  remove(filename1);
  for (size_t i = 0; i < input.size(); i++) {
    EXPECT_EQ(input[i], output[i]);
    EXPECT_EQ(input[i], output1[i]);
  }
}

TEST(test_fstream_ext, mac_failed) {
  const char* filename = "test.txt";
  const char* input = "hello world";

  CryptoPP::AutoSeededRandomPool prng;
  const int TAG_SIZE = 12;
  CryptoPP::byte key[CryptoPP::AES::DEFAULT_KEYLENGTH];
  prng.GenerateBlock(key, sizeof(key));

  CryptOfstream fout(filename, std::ios_base::binary, true, key, sizeof(key),
                     TAG_SIZE);
  auto size = strlen(input) + 1;
  fout.write(input, size);
  fout.close();

  // no alter mac value
  CryptIfstream fin(filename, std::ios_base::binary, true, key, sizeof(key),
                    TAG_SIZE);
  char* output = new char[size];
  fin.read(output, size);
  fin.peek();
  EXPECT_TRUE(fin.eof());
  fin.close();

  // alter mac value
  std::ofstream fout_a(filename, std::ios::app);
  fout_a.write("1", 1);
  fout_a.close();

  CryptIfstream fin_a(filename, std::ios_base::binary, true, key, sizeof(key),
                      TAG_SIZE);
  char* output_a = new char[size + 1];
  fin_a.read(output_a, size + 1);
  fin_a.peek();

  EXPECT_FALSE(fin_a.eof());
  remove(filename);
}

}  // namespace framework
}  // namespace paddle
