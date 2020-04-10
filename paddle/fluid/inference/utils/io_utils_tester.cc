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

#include "paddle/fluid/inference/api/helper.h"
#include "paddle/fluid/inference/utils/io_utils.h"

namespace paddle {
namespace inference {
namespace {

bool pd_tensor_equal(const paddle::PaddleTensor& ref,
                     const paddle::PaddleTensor& t) {
  bool is_equal = true;
  VLOG(3) << "ref.name: " << ref.name << ", t.name: " << t.name;
  VLOG(3) << "ref.dtype: " << ref.dtype << ", t.dtype: " << t.dtype;
  VLOG(3) << "ref.lod_level: " << ref.lod.size()
          << ", t.dtype: " << t.lod.size();
  VLOG(3) << "ref.data_len: " << ref.data.length()
          << ", t.data_len: " << t.data.length();
  return is_equal && (ref.name == t.name) && (ref.lod == t.lod) &&
         (ref.dtype == t.dtype) &&
         (std::memcmp(ref.data.data(), t.data.data(), ref.data.length()) == 0);
}

template <typename T>
void test_io_utils() {
  std::vector<T> input({6, 8});
  paddle::PaddleTensor in;
  in.name = "Hello";
  in.shape = {1, 2};
  in.lod = std::vector<std::vector<size_t>>{{0, 1}};
  in.data = paddle::PaddleBuf(static_cast<void*>(input.data()),
                              input.size() * sizeof(T));
  in.dtype = paddle::inference::PaddleTensorGetDType<T>();
  std::stringstream ss;
  paddle::inference::SerializePDTensorToStream(&ss, in);
  paddle::PaddleTensor out;
  paddle::inference::DeserializePDTensorToStream(ss, &out);
  ASSERT_TRUE(pd_tensor_equal(in, out));
}
}  // namespace
}  // namespace inference
}  // namespace paddle

TEST(infer_io_utils, float32) { paddle::inference::test_io_utils<float>(); }

TEST(infer_io_utils, tensors) {
  // Create a float32 tensor.
  std::vector<float> input_fp32({1.1f, 3.2f, 5.0f, 8.2f});
  paddle::PaddleTensor in_fp32;
  in_fp32.name = "Tensor.fp32_0";
  in_fp32.shape = {2, 2};
  in_fp32.data = paddle::PaddleBuf(static_cast<void*>(input_fp32.data()),
                                   input_fp32.size() * sizeof(float));
  in_fp32.dtype = paddle::inference::PaddleTensorGetDType<float>();

  // Create a int64 tensor.
  std::vector<float> input_int64({5, 8});
  paddle::PaddleTensor in_int64;
  in_int64.name = "Tensor.int64_0";
  in_int64.shape = {1, 2};
  in_int64.lod = std::vector<std::vector<size_t>>{{0, 1}};
  in_int64.data = paddle::PaddleBuf(static_cast<void*>(input_int64.data()),
                                    input_int64.size() * sizeof(int64_t));
  in_int64.dtype = paddle::inference::PaddleTensorGetDType<int64_t>();

  // Serialize tensors.
  std::vector<paddle::PaddleTensor> tensors_in({in_fp32});
  std::string file_path = "./io_utils_tensors";
  paddle::inference::SerializePDTensorsToFile(file_path, tensors_in);

  // Deserialize tensors.
  std::vector<paddle::PaddleTensor> tensors_out;
  paddle::inference::DeserializePDTensorsToFile(file_path, &tensors_out);

  // Check results.
  ASSERT_EQ(tensors_in.size(), tensors_out.size());
  for (size_t i = 0; i < tensors_in.size(); ++i) {
    ASSERT_TRUE(
        paddle::inference::pd_tensor_equal(tensors_in[i], tensors_out[i]));
  }
}
