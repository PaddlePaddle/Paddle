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

#include "paddle/fluid/recordio/header.h"

#include <sstream>

#include "gtest/gtest.h"

TEST(Recordio, ChunkHead) {
  paddle::recordio::Header hdr(0, 1, paddle::recordio::Compressor::kGzip, 3);
  std::stringstream ss;
  hdr.Write(ss);
  ss.seekg(0, std::ios::beg);
  paddle::recordio::Header hdr2;
  hdr2.Parse(ss);
  EXPECT_TRUE(hdr == hdr2);
}
