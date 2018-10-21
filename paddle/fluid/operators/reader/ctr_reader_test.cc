// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/operators/reader/ctr_reader.h"

#include <gzstream.h>
#include <time.h>

#include <math.h>
#include <stdio.h>
#include <cstring>
#include <fstream>
#include <tuple>

#include "gtest/gtest.h"

#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/operators/reader/blocking_queue.h"

using paddle::operators::reader::LoDTensorBlockingQueue;
using paddle::operators::reader::LoDTensorBlockingQueueHolder;
using paddle::operators::reader::CTRReader;
using paddle::framework::LoDTensor;
using paddle::framework::LoD;
using paddle::platform::CPUPlace;

static void generatedata(const std::vector<std::string>& data,
                         const std::string& file_name) {
  std::ifstream in(file_name.c_str());
  if (in.good()) {
    VLOG(3) << "file " << file_name << " exist, delete it first!";
    remove(file_name.c_str());
  } else {
    in.close();
  }

  ogzstream out(file_name.c_str());
  PADDLE_ENFORCE(out.good(), "open file %s failed!", file_name);
  for (auto& c : data) {
    out << c;
  }
  out.close();
  PADDLE_ENFORCE(out.good(), "save file %s failed!", file_name);
}

TEST(CTR_READER, read_data) {
  const std::vector<std::string> ctr_data = {
      "aaaa 1 0 0:6002 1:6003 2:6004 3:6005 4:6006 -1\n",
      "bbbb 1 0 5:6003 6:6003 7:6003 8:6004 9:6004 -1\n",
      "cccc 1 1 10:6002 11:6002 12:6002 13:6002 14:6002 -2\n",
      "dddd 1 0 15:6003 16:6003 17:6003 18:6003 19:6004 -3\n",
      "1111 1 1 20:6001 21:6001 22:6001 23:6001 24:6001 12\n",
      "2222 1 1 25:6004 26:6004 27:6004 28:6005 29:6005 aa\n",
      "3333 1 0 30:6002 31:6003 32:6004 33:6004 34:6005 er\n",
      "eeee 1 1 35:6003 36:6003 37:6005 38:6005 39:6005 dd\n",
      "ffff 1 1 40:6002 41:6003 42:6004 43:6004 44:6005 66\n",
      "gggg 1 1 46:6006 45:6006 47:6003 48:6003 49:6003 ba\n",
  };
  std::string gz_file_name = "test_ctr_reader_data.gz";
  generatedata(ctr_data, gz_file_name);

  std::vector<int64_t> label_value = {0, 0, 1, 0, 1, 1, 0, 1, 1, 1};

  std::vector<std::tuple<LoD, std::vector<int64_t>>> data_slot_6002{
      {{{0, 1, 2}}, {0, 0}},
      {{{0, 5, 6}}, {10, 11, 12, 13, 14, 0}},
      {{{0, 1, 2}}, {0, 0}},
      {{{0, 1, 2}}, {30, 0}},
      {{{0, 1, 2}}, {40, 0}}};
  std::vector<std::tuple<LoD, std::vector<int64_t>>> data_slot_6003{
      {{{0, 1, 4}}, {1, 5, 6, 7}},
      {{{0, 1, 5}}, {0, 15, 16, 17, 18}},
      {{{0, 1, 2}}, {0, 0}},
      {{{0, 1, 3}}, {31, 35, 36}},
      {{{0, 1, 4}}, {41, 47, 48, 49}}};

  LoDTensorBlockingQueueHolder queue_holder;
  int capacity = 64;
  queue_holder.InitOnce(capacity, {}, false);

  std::shared_ptr<LoDTensorBlockingQueue> queue = queue_holder.GetQueue();

  int batch_size = 2;
  int thread_num = 1;
  std::vector<std::string> slots = {"6002", "6003"};
  std::vector<std::string> file_list;
  for (int i = 0; i < thread_num; ++i) {
    file_list.push_back(gz_file_name);
  }

  CTRReader reader(queue, batch_size, thread_num, slots, file_list);

  reader.Start();

  size_t batch_num = std::ceil(ctr_data.size() / batch_size) * thread_num;

  for (size_t i = 0; i < batch_num; ++i) {
    std::vector<LoDTensor> out;
    reader.ReadNext(&out);
    ASSERT_EQ(out.size(), slots.size() + 1);
    auto& label_tensor = out.back();
    ASSERT_EQ(label_tensor.dims(),
              paddle::framework::make_ddim({1, batch_size}));
    for (size_t j = 0; j < batch_size && i * batch_num + j < ctr_data.size();
         ++j) {
      auto& label = label_tensor.data<int64_t>()[j];
      ASSERT_TRUE(label == 0 || label == 1);
      ASSERT_EQ(label, label_value[i * batch_size + j]);
    }
    auto& tensor_6002 = out[0];
    ASSERT_EQ(std::get<0>(data_slot_6002[i]), tensor_6002.lod());
    ASSERT_EQ(std::memcmp(std::get<1>(data_slot_6002[i]).data(),
                          tensor_6002.data<int64_t>(),
                          tensor_6002.dims()[1] * sizeof(int64_t)),
              0);
  }
  ASSERT_EQ(queue->Size(), 0);
}
