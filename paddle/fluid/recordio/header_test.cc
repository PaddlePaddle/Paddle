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

using namespace recordio;

TEST(Recordio, ChunkHead) {
  Header hdr(0, 1, Compressor::kGzip, 3);
  std::ostringstream oss;
  hdr.Write(oss);

  std::istringstream iss(oss.str());
  Header hdr2;
  hdr2.Parse(iss);

  std::ostringstream oss2;
  hdr2.Write(oss2);
  EXPECT_STREQ(oss2.str().c_str(), oss.str().c_str());
  EXPECT_EQ(hdr == hdr2);
}
