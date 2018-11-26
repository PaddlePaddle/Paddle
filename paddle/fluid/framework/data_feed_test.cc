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
#include "gtest/gtest.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "paddle/fluid/framework/data_feed.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/lod_tensor.h"
using paddle::framework::DataFeed;
using paddle::framework::DataFeedFactory;

paddle::framework::DataFeedDesc load_datafeed_param_from_file(const char* filename){
  paddle::framework::DataFeedDesc data_feed_desc;
  int file_descriptor = open(filename, O_RDONLY);
  if (file_descriptor == -1){
    std::cerr << "FATAL: cant open " << filename << std::endl;
    exit(-1);
  }   
  google::protobuf::io::FileInputStream fileInput(file_descriptor);
  google::protobuf::TextFormat::Parse(&fileInput, &data_feed_desc);
  close(file_descriptor);
  return data_feed_desc;
}   

void GetElemSetFromReader(std::vector<std::set<int64_t>>& reader_elem_set,
        const paddle::framework::DataFeedDesc& data_feed_desc,
        const std::vector<std::string>& filelist) {
  std::shared_ptr<DataFeed> reader = DataFeedFactory::CreateDataFeed(data_feed_desc.name());
  reader->Init(data_feed_desc);
  reader->SetFileList(filelist);
  auto* scope = new paddle::framework::Scope();
  const std::vector<std::string> & use_slot_alias =
      reader->GetUseSlotAlias();
  for (auto name: use_slot_alias){
      reader->AddFeedVar(scope->Var(name), name);
  }
  std::map<std::string, const paddle::framework::LoDTensor*> feed_targets;
  for (auto name: use_slot_alias) {
      feed_targets[name] = &scope->FindVar(name)->Get<paddle::framework::LoDTensor>();
  }
  reader_elem_set.resize(use_slot_alias.size());
  reader->Start();
  while (reader->Next()) {
    for (auto i = 0; i < use_slot_alias.size(); ++i){
      auto name = use_slot_alias[i];
      const paddle::framework::LoDTensor *tens = feed_targets[name];
      const int64_t* data = tens->data<int64_t>();
      for (auto i = 0; i < tens->NumElements(); ++i){
        std::pair<size_t, size_t> element = tens->lod_element(0, i);
        for (auto j = element.first; j < element.second; ++j){
          reader_elem_set[i].insert(data[j]);
        }
      }
    }
  }
}

void CheckIsUnorderedSame(const std::vector<std::set<int64_t>>& s1,
        const std::vector<std::set<int64_t>>& s2) {
  EXPECT_EQ(s1.size(), s2.size());
  for (auto i = 0; i < s1.size(); ++i) {
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

void GetElemSetFromFile(std::vector<std::set<int64_t>> file_elem_set,
        const paddle::framework::DataFeedDesc& data_feed_desc,
        const std::vector<std::string>& filelist) {
  file_elem_set.resize(data_feed_desc.multi_slot_desc().slots_size());
  for (const auto& file : filelist) {
    std::ifstream fin(file.c_str());
    for (auto i = 0; i < data_feed_desc.multi_slot_desc().slots_size(); ++i) {
      int num;
      fin >> num;
      while (num--) {
        uint64_t feasign;
        fin >> feasign;
        file_elem_set[i].insert(feasign);
      }
    }
  }
}

TEST(DataFeed, ReadUintTest) {
  const char* protofile = "data_feed_desc.prototxt";
  std::vector<std::string> filelist;
  filelist.push_back("train_data_new/part-0");
  filelist.push_back("train_data_new/part-1");
  filelist.push_back("train_data_new/part-2");
  filelist.push_back("train_data_new/part-3");
  filelist.push_back("train_data_new/part-4");
  filelist.push_back("train_data_new/part-5");
  filelist.push_back("train_data_new/part-6");
  filelist.push_back("train_data_new/part-7");
  filelist.push_back("train_data_new/part-8");
  filelist.push_back("train_data_new/part-9");
  filelist.push_back("train_data_new/part-10");
  filelist.push_back("train_data_new/part-11");
  paddle::framework::DataFeedDesc data_feed_desc = load_datafeed_param_from_file(protofile);
  std::vector<std::set<int64_t>> reader_elem_set;
  std::vector<std::set<int64_t>> file_elem_set;
  GetElemSetFromReader(reader_elem_set, data_feed_desc, filelist);
  GetElemSetFromFile(file_elem_set, data_feed_desc, filelist);
  CheckIsUnorderedSame(reader_elem_set, file_elem_set);
}
