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

#include <chrono>  // NOLINT
#include <set>
#include <thread>  // NOLINT
#include <vector>
#include <fcntl.h>
#include <iostream>
#include <fstream>
#include <mutex>
#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/lod_tensor.h"

paddle::framework::DataFeedDesc load_datafeed_param_from_file(const char* filename) {
  paddle::framework::DataFeedDesc data_feed_desc;
  int file_descriptor = open(filename, O_RDONLY);
  if (file_descriptor == -1){
    LOG(ERROR) << "FATAL: cant open " << filename << std::endl;
    exit(-1);
  }   
  google::protobuf::io::FileInputStream fileInput(file_descriptor);
  google::protobuf::TextFormat::Parse(&fileInput, &data_feed_desc);
  close(file_descriptor);
  return data_feed_desc;
}   

const std::vector<std::string> load_filelist_from_file(const char* filename) {
  std::vector<std::string> filelist;
  std::ifstream fin(filename);
  if (!fin.good()) {
    LOG(ERROR) << "FATAL: cant open " << filename << std::endl;
    exit(-1);
  }
  std::string line;
  while (getline(fin, line)) {
    filelist.push_back(line);
  }
  fin.close();
  return filelist;
}

void GetElemSetFromReader(std::vector<std::set<int64_t>>& reader_elem_set,
        const paddle::framework::DataFeedDesc& data_feed_desc,
        const std::vector<std::string>& filelist, const int thread_num) {
  reader_elem_set.resize(data_feed_desc.multi_slot_desc().slots_size());
  std::vector<std::thread> threads;
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers;
  readers.resize(thread_num);
  for (int i = 0; i < thread_num; ++i) {
    readers[i] = paddle::framework::DataFeedFactory::CreateDataFeed(data_feed_desc.name());
    readers[i]->Init(data_feed_desc);
  }
  readers[0]->SetFileList(filelist);
  std::mutex mu;
  for (int idx = 0; idx < thread_num; ++idx) {
    threads.emplace_back(std::thread([&, idx] {
      auto* scope = new paddle::framework::Scope();
      const std::vector<std::string> & use_slot_alias =
          readers[idx]->GetUseSlotAlias();
      for (auto name: use_slot_alias){
        readers[idx]->AddFeedVar(scope->Var(name), name);
      }
      std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
      for (auto name: use_slot_alias) {
        feed_targets[name] = &scope->FindVar(name)->Get<paddle::framework::LoDTensor>();
      }
      readers[idx]->Start();
      while (readers[idx]->Next()) {
        for (size_t k = 0; k < use_slot_alias.size(); ++k){
          auto name = use_slot_alias[k];
          const paddle::framework::LoDTensor *tens = feed_targets[name];
          const int64_t* data = tens->data<int64_t>();
          for (size_t i = 0; i < tens->NumElements(); ++i){
            std::pair<size_t, size_t> element = tens->lod_element(0, i);
            for (size_t j = element.first; j < element.second; ++j){
              std::lock_guard<std::mutex> lock(mu);
              reader_elem_set[k].insert(data[j]);
            }
          }
        } // end slots loop
      } // end while Next()
    })); // end anonymous function
  }
  for (auto& th : threads) {
    th.join();
  }
}

void CheckIsUnorderedSame(const std::vector<std::set<int64_t>>& s1,
        const std::vector<std::set<int64_t>>& s2) {
  EXPECT_EQ(s1.size(), s2.size());
  for (size_t i = 0; i < s1.size(); ++i) {
    EXPECT_EQ(s1[i].size(), s2[i].size());
    auto it1 = s1[i].begin();
    auto it2 = s2[i].begin();
    while (it1 != s1[i].end()) {
      EXPECT_EQ(*it1, *it2);
      ++it1;
      ++it2;
    }
  }
}

void GetElemSetFromFile(std::vector<std::set<int64_t>>& file_elem_set,
        const paddle::framework::DataFeedDesc& data_feed_desc,
        const std::vector<std::string>& filelist) {
  file_elem_set.resize(data_feed_desc.multi_slot_desc().slots_size());
  for (const auto& file : filelist) {
    std::ifstream fin(file.c_str());
    if (!fin.good()) {
      LOG(ERROR) << "FATAL: cant open " << file << std::endl;
      exit(-1);
    }
    while (1) {
      bool end_flag = false;
      for (auto i = 0; i < data_feed_desc.multi_slot_desc().slots_size(); ++i) {
        int num;
        if (fin >> num) {
          while (num--) {
            uint64_t feasign;
            fin >> feasign;
            file_elem_set[i].insert(feasign);
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

TEST(DataFeed, ReadUintTest) {
  const char* protofile = "data_feed_desc.prototxt";
  const char* filelist_name = "filelist.txt";
  const std::vector<std::string> filelist = load_filelist_from_file(filelist_name);
  paddle::framework::DataFeedDesc data_feed_desc = load_datafeed_param_from_file(protofile);
  std::vector<std::set<int64_t>> reader_elem_set;
  std::vector<std::set<int64_t>> file_elem_set;
  GetElemSetFromReader(reader_elem_set, data_feed_desc, filelist, 12);
  GetElemSetFromFile(file_elem_set, data_feed_desc, filelist);
  CheckIsUnorderedSame(reader_elem_set, file_elem_set);
}
