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

#include "paddle/fluid/framework/data_feed.h"

#include <fcntl.h>

#include <chrono>  // NOLINT
#include <fstream>
#include <iostream>
#include <map>
#include <mutex>  // NOLINT
#include <set>
#include <thread>  // NOLINT
#include <utility>
#include <vector>

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/text_format.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"

paddle::framework::DataFeedDesc load_datafeed_param_from_file(
    const char* filename) {
  paddle::framework::DataFeedDesc data_feed_desc;
  int file_descriptor = open(filename, O_RDONLY);
  PADDLE_ENFORCE_NE(
      file_descriptor,
      -1,
      common::errors::Unavailable(
          "Cannot open file %s c load datafeed param from file.", filename));
  google::protobuf::io::FileInputStream fileInput(file_descriptor);
  google::protobuf::TextFormat::Parse(&fileInput, &data_feed_desc);
  close(file_descriptor);
  return data_feed_desc;
}

const std::vector<std::string> load_filelist_from_file(const char* filename) {
  std::vector<std::string> filelist;
  std::ifstream fin(filename);
  PADDLE_ENFORCE_EQ(
      fin.good(),
      true,
      common::errors::Unavailable(
          "Cannot open file %s when load filelist from file.", filename));
  std::string line;
  while (getline(fin, line)) {
    filelist.push_back(line);
  }
  fin.close();
  return filelist;
}

void GenerateFileForTest(const char* protofile, const char* filelist) {
  std::ofstream w_protofile(protofile);
  w_protofile << "name: \"MultiSlotDataFeed\"\n"
                 "batch_size: 2\n"
                 "multi_slot_desc {\n"
                 "    slots {\n"
                 "        name: \"uint64_sparse_slot\"\n"
                 "        type: \"uint64\"\n"
                 "        is_dense: false\n"
                 "        is_used: true\n"
                 "    }\n"
                 "    slots {\n"
                 "        name: \"float_sparse_slot\"\n"
                 "        type: \"float\"\n"
                 "        is_dense: false\n"
                 "        is_used: true\n"
                 "    }\n"
                 "    slots {\n"
                 "        name: \"uint64_dense_slot\"\n"
                 "        type: \"uint64\"\n"
                 "        is_dense: true\n"
                 "        is_used: true\n"
                 "    }\n"
                 "    slots {\n"
                 "        name: \"float_dense_slot\"\n"
                 "        type: \"float\"\n"
                 "        is_dense: true\n"
                 "        is_used: true\n"
                 "    }\n"
                 "    slots {\n"
                 "        name: \"not_used_slot\"\n"
                 "        type: \"uint64\"\n"
                 "        is_dense: false\n"
                 "        is_used: false\n"
                 "    }\n"
                 "}";
  w_protofile.close();
  std::ofstream w_filelist(filelist);
  int total_file = 4;
  for (int i = 0; i < total_file; ++i) {
    std::string filename = "TestMultiSlotDataFeed.data." + std::to_string(i);
    w_filelist << filename;
    if (i + 1 != total_file) {
      w_filelist << std::endl;
    }
    std::ofstream w_datafile(filename.c_str());
    w_datafile << "3 3978 620 82 1 1926.08 1 1926 1 6.02 1 1996\n"
                  "2 1300 2983353 1 985.211 1 8 1 0.618 1 12\n"
                  "1 19260827 2 3.14 2.718 1 27 1 2.236 1 28\n";
    w_datafile.close();
  }
  w_filelist.close();
}

class MultiTypeSet {
 public:
  MultiTypeSet() {
    uint64_set_.clear();
    float_set_.clear();
  }
  ~MultiTypeSet() {}
  void AddValue(uint64_t v) { uint64_set_.insert(v); }
  void AddValue(float v) { float_set_.insert(v); }
  const std::set<uint64_t>& GetUint64Set() const { return uint64_set_; }
  const std::set<float>& GetFloatSet() const { return float_set_; }

 private:
  std::set<uint64_t> uint64_set_;
  std::set<float> float_set_;
};

void GetElemSetFromReader(std::vector<MultiTypeSet>* reader_elem_set,
                          const paddle::framework::DataFeedDesc& data_feed_desc,
                          const std::vector<std::string>& filelist,
                          const int thread_num) {
  int used_slot_num = 0;
  for (auto i = 0; i < data_feed_desc.multi_slot_desc().slots_size(); ++i) {
    if (data_feed_desc.multi_slot_desc().slots(i).is_used()) {
      ++used_slot_num;
    }
  }
  reader_elem_set->resize(used_slot_num);
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers;
  readers.resize(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    readers[i] = paddle::framework::DataFeedFactory::CreateDataFeed(
        data_feed_desc.name());
    readers[i]->Init(data_feed_desc);
  }
  readers[0]->SetFileList(filelist);
  std::mutex mu;
  for (int idx = 0; idx < thread_num; ++idx) {
    threads.emplace_back(std::thread([&, idx] {
      std::unique_ptr<paddle::framework::Scope> scope(
          new paddle::framework::Scope());
      const auto& multi_slot_desc = data_feed_desc.multi_slot_desc();
      std::map<std::string, const phi::DenseTensor*> lodtensor_targets;
      for (int i = 0; i < multi_slot_desc.slots_size(); ++i) {
        const auto& slot = multi_slot_desc.slots(i);
        if (slot.is_used()) {
          const auto& name = slot.name();
          readers[idx]->AddFeedVar(scope->Var(name), name);
          lodtensor_targets[name] =
              &scope->FindVar(name)->Get<phi::DenseTensor>();
        }
      }
      readers[idx]->Start();
      while (readers[idx]->Next()) {
        int index = 0;
        for (int k = 0; k < multi_slot_desc.slots_size(); ++k) {
          const auto& slot = multi_slot_desc.slots(k);
          if (!slot.is_used()) {
            continue;
          }
          const phi::DenseTensor* tens = lodtensor_targets[slot.name()];
          if (slot.is_dense()) {  // dense branch
            if (slot.type() == "uint64") {
              const int64_t* data = tens->data<int64_t>();
              int batch_size = tens->dims()[0];
              int dim = tens->dims()[1];
              for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < dim; ++j) {
                  std::lock_guard<std::mutex> lock(mu);
                  (*reader_elem_set)[index].AddValue(
                      (uint64_t)data[i * dim + j]);
                }
              }
            } else if (slot.type() == "float") {
              const float* data = tens->data<float>();
              int batch_size = tens->dims()[0];
              int dim = tens->dims()[1];
              for (int i = 0; i < batch_size; ++i) {
                for (int j = 0; j < dim; ++j) {
                  std::lock_guard<std::mutex> lock(mu);
                  (*reader_elem_set)[index].AddValue(data[i * dim + j]);
                }
              }
            } else {
              PADDLE_THROW(
                  common::errors::InvalidArgument("Error type in proto file."));
            }
          } else {  // sparse branch
            if (slot.type() == "uint64") {
              const int64_t* data = tens->data<int64_t>();
              for (size_t i = 0; i < tens->NumElements(); ++i) {
                std::pair<size_t, size_t> element = tens->lod_element(0, i);
                for (size_t j = element.first; j < element.second; ++j) {
                  std::lock_guard<std::mutex> lock(mu);
                  (*reader_elem_set)[index].AddValue((uint64_t)data[j]);
                }
              }
            } else if (slot.type() == "float") {
              const float* data = tens->data<float>();
              for (size_t i = 0; i < tens->NumElements(); ++i) {
                std::pair<size_t, size_t> element = tens->lod_element(0, i);
                for (size_t j = element.first; j < element.second; ++j) {
                  std::lock_guard<std::mutex> lock(mu);
                  (*reader_elem_set)[index].AddValue(data[j]);
                }
              }
            } else {
              PADDLE_THROW(
                  common::errors::InvalidArgument("Error type in proto file."));
            }
          }  // end sparse branch
          ++index;
        }  // end slots loop
      }    // end while Next()
    }));   // end anonymous function
  }
  for (auto& th : threads) {
    th.join();
  }
}

void CheckIsUnorderedSame(const std::vector<MultiTypeSet>& s1,
                          const std::vector<MultiTypeSet>& s2) {
  EXPECT_EQ(s1.size(), s2.size());
  for (size_t i = 0; i < s1.size(); ++i) {
    // check for uint64
    const std::set<uint64_t>& uint64_s1 = s1[i].GetUint64Set();
    const std::set<uint64_t>& uint64_s2 = s2[i].GetUint64Set();
    EXPECT_EQ(uint64_s1.size(), uint64_s2.size());
    auto uint64_it1 = uint64_s1.begin();
    auto uint64_it2 = uint64_s2.begin();
    while (uint64_it1 != uint64_s1.end()) {
      EXPECT_EQ(*uint64_it1, *uint64_it2);
      ++uint64_it1;
      ++uint64_it2;
    }
    // check for float
    const std::set<float>& float_s1 = s1[i].GetFloatSet();
    const std::set<float>& float_s2 = s2[i].GetFloatSet();
    EXPECT_EQ(float_s1.size(), float_s2.size());
    auto float_it1 = float_s1.begin();
    auto float_it2 = float_s2.begin();
    while (float_it1 != float_s1.end()) {
      EXPECT_EQ(*float_it1, *float_it2);
      ++float_it1;
      ++float_it2;
    }
  }
}

void GetElemSetFromFile(std::vector<MultiTypeSet>* file_elem_set,
                        const paddle::framework::DataFeedDesc& data_feed_desc,
                        const std::vector<std::string>& filelist) {
  int used_slot_num = 0;
  for (auto i = 0; i < data_feed_desc.multi_slot_desc().slots_size(); ++i) {
    if (data_feed_desc.multi_slot_desc().slots(i).is_used()) {
      ++used_slot_num;
    }
  }
  file_elem_set->resize(used_slot_num);
  for (const auto& file : filelist) {
    std::ifstream fin(file.c_str());
    PADDLE_ENFORCE_EQ(
        fin.good(),
        true,
        common::errors::Unavailable(
            "Can not open %s when get element set from file.", file.c_str()));
    while (1) {
      bool end_flag = false;
      int index = 0;
      for (auto i = 0; i < data_feed_desc.multi_slot_desc().slots_size(); ++i) {
        int num;
        if (fin >> num) {
          auto slot = data_feed_desc.multi_slot_desc().slots(i);
          auto type = slot.type();
          if (type == "uint64") {
            while (num--) {
              uint64_t feasign;
              fin >> feasign;
              if (slot.is_used()) {
                (*file_elem_set)[index].AddValue(feasign);
              }
            }
          } else if (type == "float") {
            while (num--) {
              float feasign;
              fin >> feasign;
              if (slot.is_used()) {
                (*file_elem_set)[index].AddValue(feasign);
              }
            }
          } else {
            PADDLE_THROW(
                common::errors::InvalidArgument("Error type in proto file."));
          }
          if (slot.is_used()) {
            ++index;
          }
        } else {
          end_flag = true;
          break;
        }
      }
      if (end_flag) {
        break;
      }
    }
    fin.close();
  }
}

TEST(DataFeed, MultiSlotUnitTest) {
  const char* protofile = "data_feed_desc.prototxt";
  const char* filelist_name = "filelist.txt";
  GenerateFileForTest(protofile, filelist_name);
  const std::vector<std::string> filelist =
      load_filelist_from_file(filelist_name);
  paddle::framework::DataFeedDesc data_feed_desc =
      load_datafeed_param_from_file(protofile);
  std::vector<MultiTypeSet> reader_elem_set;
  std::vector<MultiTypeSet> file_elem_set;
  // GetElemSetFromReader(&reader_elem_set, data_feed_desc, filelist, 4);
  // GetElemSetFromFile(&file_elem_set, data_feed_desc, filelist);
  // CheckIsUnorderedSame(reader_elem_set, file_elem_set);
}
