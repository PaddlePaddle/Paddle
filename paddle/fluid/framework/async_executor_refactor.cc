/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/async_executor.h"
#include <stdio.h>
#include <string.h>
#include <fcntl.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fstream>
#include <iostream>
#include <map>
#include <algorithm>
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/pybind/pybind.h"
namespace paddle {
namespace framework {

AsyncExecutor::AsyncExecutor(const platform::Place& place) {
  place_ = place;
}

void AsyncExecutor::CreateThreads(const ExecutorThreadWorker* worker,
                                  const ProgramDesc& main_program,
                                  const DataFeed& reader,
                                  const Scope& root_scope,
                                  const int thread_index) {
  worker->SetThreadid(thread_index);
  worker->CreateThreadResource(main_program, place_);
  worker->SetDataFeed(reader);
  worker->BindingDataFeedMemory(reader);
  worker->SetRootScope(root_scope);
}

void AsyncExecutor::CheckFiles(
    const std::vector<std::string>& files) {
  // function for user to check file formats
  // should be exposed to users
}

/*
  in case there are binary files we want to train
  and in general this is the fastest way to train
  different calls allow thread_num to be different
  threads are created on the fly
  workers are created on the fly
  readers are created on the fly
  files are fed into readers on the fly
 */
/*
  class_name
  batch_size: max batch size
  use_slot: 
  queue_size: 
  buffer_size:
  padding_index: 
 */
void AsyncExecutor::RunFromFiles(
    const ProgramDesc& main_program,
    const DataFeedDesc& data_feed_desc,
    const std::vector<std::string> & files,
    const int thread_num) {
  // todo: remove fluid related interface
  root_scope_->DropKids();
  std::vector<std::thread> threads;
  threads.resize(thread_num);

  /*
    readerDesc: protobuf description for reader initlization
    argument: class_name, batch_size, use_slot, queue_size, buffer_size, padding_index
    
    reader: 
    1) each thread has a reader, reader will read input data and 
    put it into input queue
    2) each reader has a Next() iterface, that can fetch an instance
    from the input queue
   */
  // todo: should be factory method for creating datafeed
  std::vector<std::shared_ptr<DataFeed> > readers;
  readers.resize(thread_num);
  for (int i = 0; i < readers.size(); ++i) {
    readers[i] = DataFeedFactory::CreateDataFeed(data_feed_desc.name());
  }

  /*
  std::vector<std::shared_ptr<ExecutorStrategy> > workers;
  workers.resize(thread_num);
  std::string str_name = strategy_.name;
  for (auto& worker : workers) {
    worker.reset(
        ExecutorStrategyFactory::CreateExecutorStrategy(str_name));
  }
  */

  std::vector<std::shared_ptr<ExecutorThreadWorker> > workers;
  workers.resize(thread_num);
  for (auto& worker : workers) {
    worker.reset(new ExecutorThreadWorker);
  }

  // prepare thread resource here
  for (int thidx = 0; thidx < thread_num; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program,
                  readers[thidx].get(), root_scope_, thidx);
  }
  
  // start executing ops in multiple threads
  for (int thidx = 0; thidx < thread_num_; ++thidx) {
    threads.push_back(std::thread(&ExecutorThreadWorker::TrainFiles,
                                  workers[thidx].get()));
  }

  for (auto& th : threads) {
    th.join();
  }
  // fetch variables in scope 0, and return
}

}   // einit_modelnd namespace framework
}   // end namespace paddle

/* vim: set expandtab ts=2 sw=2 sts=2 tw=100: */
