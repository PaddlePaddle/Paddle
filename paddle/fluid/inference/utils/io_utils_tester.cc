// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <glog/logging.h>
#include <gtest/gtest.h>

#include "paddle/fluid/inference/utils/io_utils.h"

namespace paddle {
namespace inference {
namespace {

template <typename T>
void test_io_utils() {
  std::vector<T> input({6, 8});
  paddle::PaddleTensor in;
  in.name = "Hello";
  in.shape = {1, 2};
  in.lod = std::vector<std::vector<size_t>>{{0, 1}};
  in.data = paddle::PaddleBuf(static_cast<void*>(input.data()),
                              input.size() * sizeof(T));
  in.dtype = paddle::inference::GetPaddleDType<T>();
  std::stringstream ss;
  paddle::inference::SerializePDTensorToStream(&ss, in);
  paddle::PaddleTensor out;
  paddle::inference::DeserializePDTensorToStream(ss, &out);
  std::vector<T> output(static_cast<T*>(out.data.data()),
                        static_cast<T*>(out.data.data()) + input.size());

  ASSERT_EQ(in.name, out.name);
  ASSERT_EQ(in.lod, out.lod);
  ASSERT_EQ(in.dtype, out.dtype);
  ASSERT_EQ(input, output);
}
}  // namespace
}  // namespace inference
}  // namespace paddle

TEST(infer_io_utils, float32) { paddle::inference::test_io_utils<float>(); }
TEST(infer_io_utils, int64) { paddle::inference::test_io_utils<int64_t>(); }
