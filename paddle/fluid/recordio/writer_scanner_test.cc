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

#include <sstream>
#include <string>

#include "gtest/gtest.h"
#include "paddle/fluid/recordio/scanner.h"
#include "paddle/fluid/recordio/writer.h"

TEST(WriterScanner, Normal) {
  std::stringstream* stream = new std::stringstream();

  {
    paddle::recordio::Writer writer(stream,
                                    paddle::recordio::Compressor::kSnappy);
    writer.Write("ABC");
    writer.Write("BCD");
    writer.Write("CDE");
    writer.Flush();
  }

  {
    stream->seekg(0, std::ios::beg);
    std::unique_ptr<std::istream> stream_ptr(stream);
    paddle::recordio::Scanner scanner(std::move(stream_ptr));
    ASSERT_TRUE(scanner.HasNext());
    ASSERT_EQ(scanner.Next(), "ABC");
    ASSERT_EQ("BCD", scanner.Next());
    ASSERT_TRUE(scanner.HasNext());
    ASSERT_EQ("CDE", scanner.Next());
    ASSERT_FALSE(scanner.HasNext());
  }
}

TEST(WriterScanner, TinyChunk) {
  std::stringstream* stream = new std::stringstream();
  {
    paddle::recordio::Writer writer(
        stream, paddle::recordio::Compressor::kNoCompress, 2 /*max chunk num*/);
    writer.Write("ABC");
    writer.Write("BCD");
    writer.Write("CDE");
    writer.Write("DEFG");
    writer.Flush();
  }

  {
    stream->seekg(0, std::ios::beg);
    std::unique_ptr<std::istream> stream_ptr(stream);
    paddle::recordio::Scanner scanner(std::move(stream_ptr));
    ASSERT_TRUE(scanner.HasNext());
    ASSERT_EQ(scanner.Next(), "ABC");
    ASSERT_EQ(scanner.Next(), "BCD");
    ASSERT_EQ(scanner.Next(), "CDE");
    ASSERT_EQ(scanner.Next(), "DEFG");
    ASSERT_FALSE(scanner.HasNext());
  }
}
