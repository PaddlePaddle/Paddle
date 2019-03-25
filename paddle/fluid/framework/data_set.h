/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 *     Unless required by applicable law or agreed to in writing, software
 *     distributed under the License is distributed on an "AS IS" BASIS,
 *     WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *     See the License for the specific language governing permissions and
 *     limitations under the License. */

#pragma once

#include <fstream>
#include <memory>
#include <mutex>  // NOLINT
#include <string>
#include <thread>  // NOLINT
#include <vector>
#include <utility>

#include "paddle/fluid/framework/data_feed.h"

namespace paddle {
namespace framework {

// Dataset is a abstract class, which defines user interfaces
// Example Usage:
//    Dataset* dataset = DatasetFactory::CreateDataset("InMemoryDataset")
//    dataset->SetFileList(std::vector<std::string>{"a.txt", "b.txt"})
//    dataset->SetThreadNum(1)
//    dataset->CreateReaders();
//    dataset->SetDataFeedDesc(your_data_feed_desc);
//    dataset->LoadIntoMemory();
//    dataset->SetTrainerNum(2);
//    dataset->GlobalShuffle();
class Dataset {
 public:
  Dataset() {}
  virtual ~Dataset() {}
  // set file list
  virtual void SetFileList(const std::vector<std::string>& filelist) = 0;
  // set readers' num
  virtual void SetThreadNum(int thread_num) = 0;
  // set workers' num
  virtual void SetTrainerNum(int trainer_num) = 0;
  // set fs name and ugi
  virtual void SetHdfsConfig(const std::string& fs_name,
                             const std::string& fs_ugi) = 0;
  // set data fedd desc, which contains:
  //   data feed name, batch size, slots
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str) = 0;
  // get file list
  virtual const std::vector<std::string>& GetFileList() = 0;
  // get thread num
  virtual int GetThreadNum() = 0;
  // get worker num
  virtual int GetTrainerNum() = 0;
  // get hdfs config
  virtual std::pair<std::string, std::string> GetHdfsConfig() = 0;
  // get data fedd desc
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() = 0;
  // get readers, the reader num depend both on thread num
  // and filelist size
  virtual std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
  GetReaders() = 0;
  // register message handler between workers
  virtual void RegisterClientToClientMsgHandler() = 0;
  // load all data into memory
  virtual void LoadIntoMemory() = 0;
  // release all memory data
  virtual void ReleaseMemory() = 0;
  // local shuffle data
  virtual void LocalShuffle() = 0;
  // global shuffle data
  virtual void GlobalShuffle() = 0;
  // create readers
  virtual void CreateReaders() = 0;
  // destroy readers
  virtual void DestroyReaders() = 0;

 protected:
  virtual int ReceiveFromClient(int msg_type, int client_id,
                                const std::string& msg) = 0;
};

// DatasetImpl is the implementation of Dataset,
// it holds memory data if user calls load_into_memory
template <typename T>
class DatasetImpl : public Dataset {
 public:
  DatasetImpl();
  virtual ~DatasetImpl() {}

  virtual void SetFileList(const std::vector<std::string>& filelist);
  virtual void SetThreadNum(int thread_num);
  virtual void SetTrainerNum(int trainer_num);
  virtual void SetHdfsConfig(const std::string& fs_name,
                             const std::string& fs_ugi);
  virtual void SetDataFeedDesc(const std::string& data_feed_desc_str);

  virtual const std::vector<std::string>& GetFileList() { return filelist_; }
  virtual int GetThreadNum() { return thread_num_; }
  virtual int GetTrainerNum() { return trainer_num_; }
  virtual std::pair<std::string, std::string> GetHdfsConfig() {
    return std::make_pair(fs_name_, fs_ugi_);
  }
  virtual const paddle::framework::DataFeedDesc& GetDataFeedDesc() {
    return data_feed_desc_;
  }
  virtual std::vector<std::shared_ptr<paddle::framework::DataFeed>>&
  GetReaders();

  virtual void RegisterClientToClientMsgHandler();
  virtual void LoadIntoMemory();
  virtual void ReleaseMemory();
  virtual void LocalShuffle();
  virtual void GlobalShuffle();
  virtual void CreateReaders();
  virtual void DestroyReaders();

 protected:
  virtual int ReceiveFromClient(int msg_type, int client_id,
                                const std::string& msg);
  std::vector<std::shared_ptr<paddle::framework::DataFeed>> readers_;
  std::vector<T> memory_data_;
  std::mutex mutex_for_update_memory_data_;
  int thread_num_;
  paddle::framework::DataFeedDesc data_feed_desc_;
  int trainer_num_;
  std::vector<std::string> filelist_;
  size_t file_idx_;
  std::mutex mutex_for_pick_file_;
  std::string fs_name_;
  std::string fs_ugi_;
};

// use std::vector<MultiSlotType> as data type
class MultiSlotDataset : public DatasetImpl<std::vector<MultiSlotType>> {
 public:
  MultiSlotDataset() {}
  virtual ~MultiSlotDataset() {}
};

}  // end namespace framework
}  // end namespace paddle
