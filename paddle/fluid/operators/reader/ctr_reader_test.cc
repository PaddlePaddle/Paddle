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

using paddle::operators::reader::LoDTensorBlockingQueues;
using paddle::operators::reader::LoDTensorBlockingQueueHolder;
using paddle::operators::reader::CTRReader;
using paddle::framework::LoDTensor;
using paddle::framework::LoD;
using paddle::framework::DDim;
using paddle::platform::CPUPlace;
using paddle::framework::make_ddim;
using paddle::operators::reader::DataDesc;

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

static inline void check_all_data(
    const std::vector<std::string>& ctr_data,
    const std::vector<std::string>& slots, const std::vector<DDim>& label_dims,
    const std::vector<int64_t>& label_value,
    const std::vector<std::tuple<LoD, std::vector<int64_t>>>& data_slot_6002,
    const std::vector<std::tuple<LoD, std::vector<int64_t>>>& data_slot_6003,
    size_t batch_num, size_t batch_size,
    std::shared_ptr<LoDTensorBlockingQueues> queue, CTRReader* reader) {
  std::vector<LoDTensor> out;
  for (size_t i = 0; i < batch_num; ++i) {
    reader->ReadNext(&out);
    ASSERT_EQ(out.size(), slots.size() + 1);
    auto& label_tensor = out.back();
    ASSERT_EQ(label_tensor.dims(), label_dims[i]);
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
  reader->ReadNext(&out);
  ASSERT_EQ(out.size(), 0);
  ASSERT_EQ(queue->Size(), 0);
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

  std::tuple<LoD, std::vector<int64_t>> a1({{0, 1, 2, 7}},
                                           {0, 0, 10, 11, 12, 13, 14});
  std::tuple<LoD, std::vector<int64_t>> a2({{0, 1, 2, 3}}, {0, 0, 0});
  std::tuple<LoD, std::vector<int64_t>> a3({{0, 1, 2, 3}}, {30, 0, 40});
  std::tuple<LoD, std::vector<int64_t>> a4({{0, 1}}, {0});
  std::vector<std::tuple<LoD, std::vector<int64_t>>> data_slot_6002{a1, a2, a3,
                                                                    a4};

  std::tuple<LoD, std::vector<int64_t>> b1({{0, 1, 4, 5}}, {1, 5, 6, 7, 0});
  std::tuple<LoD, std::vector<int64_t>> b2({{0, 4, 5, 6}},
                                           {15, 16, 17, 18, 0, 0});
  std::tuple<LoD, std::vector<int64_t>> b3({{0, 1, 3, 4}}, {31, 35, 36, 41});
  std::tuple<LoD, std::vector<int64_t>> b4({{0, 3}}, {47, 48, 49});
  std::vector<std::tuple<LoD, std::vector<int64_t>>> data_slot_6003{b1, b2, b3,
                                                                    b4};

  std::vector<DDim> label_dims = {{3, 1}, {3, 1}, {3, 1}, {1, 1}};

  LoDTensorBlockingQueueHolder queue_holder;
  int capacity = 64;
  queue_holder.InitOnce(capacity, false);

  std::shared_ptr<LoDTensorBlockingQueues> queue = queue_holder.GetQueue();

  int batch_size = 3;
  int thread_num = 1;
  std::vector<std::string> sparse_slots = {"6002", "6003"};
  std::vector<std::string> file_list;
  for (int i = 0; i < thread_num; ++i) {
    file_list.push_back(gz_file_name);
  }

  DataDesc data_desc(batch_size, file_list, "gzip", "svm", {}, {},
                     sparse_slots);

  CTRReader reader(queue, thread_num, data_desc);

  reader.Start();
  size_t batch_num =
      std::ceil(static_cast<float>(ctr_data.size()) / batch_size) * thread_num;
  check_all_data(ctr_data, sparse_slots, label_dims, label_value,
                 data_slot_6002, data_slot_6003, batch_num, batch_size, queue,
                 &reader);

  reader.Shutdown();

  reader.Start();
  check_all_data(ctr_data, sparse_slots, label_dims, label_value,
                 data_slot_6002, data_slot_6003, batch_num, batch_size, queue,
                 &reader);
  reader.Shutdown();
}

static void GenereteCsvData(const std::string& file_name,
                            const std::vector<std::string>& data) {
  std::ofstream out(file_name.c_str());
  PADDLE_ENFORCE(out.good(), "open file %s failed!", file_name);
  for (auto& c : data) {
    out << c;
  }
  out.close();
  PADDLE_ENFORCE(out.good(), "save file %s failed!", file_name);
}

static void CheckReadCsvOut(const std::vector<LoDTensor>& out) {
  ASSERT_EQ(out.size(), 3);
  ASSERT_EQ(out[0].dims()[1], 1);
  ASSERT_EQ(out[1].dims()[1], 2);
  ASSERT_EQ(out[2].dims()[1], 1);
  for (size_t i = 0; i < out[0].numel(); ++i) {
    int64_t label = out[0].data<int64_t>()[i];
    auto& dense_dim = out[1].dims();
    for (size_t j = 0; j < dense_dim[1]; ++j) {
      ASSERT_EQ(out[1].data<float>()[i * dense_dim[1] + j],
                static_cast<float>(label + 0.1));
    }
    auto& sparse_lod = out[2].lod();
    for (size_t j = sparse_lod[0][i]; j < sparse_lod[0][i + 1]; ++j) {
      ASSERT_EQ(out[2].data<int64_t>()[j], label);
    }
  }
}

TEST(CTR_READER, read_csv_data) {
  std::string file_name = "test_ctr_reader_data.csv";
  const std::vector<std::string> csv_data = {
      "0 0.1,0.1 0,0,0,0\n", "1 1.1,1.1 1,1,1,1\n", "2 2.1,2.1 2,2,2,2\n",
      "3 3.1,3.1 3,3,3,3\n",
  };
  GenereteCsvData(file_name, csv_data);

  LoDTensorBlockingQueueHolder queue_holder;
  int capacity = 64;
  queue_holder.InitOnce(capacity, false);

  std::shared_ptr<LoDTensorBlockingQueues> queue = queue_holder.GetQueue();

  int batch_size = 3;
  int thread_num = 1;
  std::vector<std::string> file_list;
  for (int i = 0; i < thread_num; ++i) {
    file_list.push_back(file_name);
  }
  DataDesc data_desc(batch_size, file_list, "plain", "csv", {1}, {2}, {});

  CTRReader reader(queue, thread_num, data_desc);

  for (size_t i = 0; i < 2; ++i) {
    reader.Start();
    std::vector<LoDTensor> out;
    while (true) {
      reader.ReadNext(&out);
      if (out.empty()) {
        break;
      }
      CheckReadCsvOut(out);
    }
    reader.Shutdown();
  }
}
