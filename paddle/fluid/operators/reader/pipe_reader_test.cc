// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/pipe_reader.h"

#include "gtest/gtest.h"

using paddle::operators::reader::PipeReader;
using paddle::operators::reader::Pipe;
using paddle::operators::reader::ReadPipe;
using paddle::operators::reader::WritePipe;

void write_func(int fd, const std::vector<float>& data) {
  WritePipe writer(fd);
  uint64_t num_tensors = 1;
  writer.write(reinterpret_cast<uint8_t*>(&num_tensors), sizeof(num_tensors));
  uint64_t lod_level = 1;
  writer.write(reinterpret_cast<uint8_t*>(&lod_level), sizeof(lod_level));
  uint64_t level_size = sizeof(size_t);
  writer.write(reinterpret_cast<uint8_t*>(&level_size), sizeof(level_size));
  uint64_t level_data = 1;
  writer.write(reinterpret_cast<uint8_t*>(&level_data), sizeof(level_data));

  uint32_t dtype = paddle::framework::proto::VarType_Type_FP32;
  writer.write(reinterpret_cast<uint8_t*>(&dtype), sizeof(dtype));
  uint64_t num_dim = 1;
  writer.write(reinterpret_cast<uint8_t*>(&num_dim), sizeof(num_dim));
  uint64_t dim_size = data.size();
  writer.write(reinterpret_cast<uint8_t*>(&dim_size), sizeof(dim_size));
  writer.write(reinterpret_cast<const uint8_t*>(data.data()),
               sizeof(float) * data.size());
}

TEST(PIPE_READER, read_data) {
  auto fds = Pipe::Create();
  PipeReader reader(fds[0]);
  std::vector<float> data = {1, 2, 3, 4, 5, 6, 7};
  std::thread write_thread(write_func, fds[1], std::ref(data));

  std::vector<paddle::framework::LoDTensor> tensors;
  reader.ReadNext(&tensors);
  write_thread.join();
  auto& tensor = tensors[0];
  PADDLE_ENFORCE_EQ(tensor.dims().size(), 1, "Tensor num_dim incorrect");
  PADDLE_ENFORCE_EQ(tensor.dims()[0], data.size(), "Tensor dims incorrect");
  for (size_t i = 0; i < data.size(); i++) {
    PADDLE_ENFORCE_EQ(tensor.data<float>()[i], data[i],
                      "Tensor value [%d] incorrect", i);
  }
}
