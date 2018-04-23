//   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/recordio/chunk.h"

#include <sstream>

#include "gtest/gtest.h"

TEST(Chunk, SaveLoad) {
  paddle::recordio::Chunk ch;
  ch.Add(std::string("12345", 6));
  ch.Add(std::string("123", 4));
  std::stringstream ss;
  ch.Write(ss, paddle::recordio::Compressor::kNoCompress);
  ss.seekg(0);
  ch.Parse(ss);
  ASSERT_EQ(ch.NumBytes(), 10U);
}

TEST(Chunk, Compressor) {
  paddle::recordio::Chunk ch;
  ch.Add(std::string("12345", 6));
  ch.Add(std::string("123", 4));
  ch.Add(std::string("123", 4));
  ch.Add(std::string("123", 4));
  std::stringstream ss;
  ch.Write(ss, paddle::recordio::Compressor::kSnappy);
  std::stringstream ss2;
  ch.Write(ss2, paddle::recordio::Compressor::kNoCompress);
  ASSERT_LE(ss.tellp(), ss2.tellp());  // Compress should contain less data;

  ch.Clear();
  ch.Parse(ss);
  ASSERT_EQ(ch.NumBytes(), 18ul);
}
