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
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/data_feed_factory.h"
#include "paddle/fluid/framework/executor_thread_worker.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

namespace paddle {
namespace framework {
AsyncExecutor::AsyncExecutor(Scope* scope, const platform::Place& place)
    : root_scope_(scope), place_(place) {}

void AsyncExecutor::CreateThreads(
    ExecutorThreadWorker* worker, const ProgramDesc& main_program,
    const std::shared_ptr<DataFeed>& reader,
    const std::vector<std::string>& fetch_var_names, Scope* root_scope,
    const int thread_index, const bool debug) {
  worker->SetThreadId(thread_index);
  worker->SetDebug(debug);
  worker->SetRootScope(root_scope);
  worker->CreateThreadResource(main_program, place_);
  worker->SetDataFeed(reader);
  worker->SetFetchVarNames(fetch_var_names);
  worker->BindingDataFeedMemory();
}

void PrepareReaders(std::vector<std::shared_ptr<DataFeed>>& readers,  // NOLINT
                    const int thread_num, const DataFeedDesc& data_feed_desc,
                    const std::vector<std::string>& filelist) {
  readers.resize(thread_num);
  for (size_t i = 0; i < readers.size(); ++i) {
    readers[i] = DataFeedFactory::CreateDataFeed(data_feed_desc.name());
    readers[i]->Init(data_feed_desc);  // set batch_size and queue_size here
  }
  readers[0]->SetFileList(filelist);
}

void AsyncExecutor::RunFromFile(const ProgramDesc& main_program,
                                const std::string& data_feed_desc_str,
                                const std::vector<std::string>& filelist,
                                const int thread_num,
                                const std::vector<std::string>& fetch_var_names,
                                const bool debug) {
  std::vector<std::thread> threads;

  auto& block = main_program.Block(0);
  for (auto var_name : fetch_var_names) {
    auto var_desc = block.FindVar(var_name);
    auto shapes = var_desc->GetShape();
    PADDLE_ENFORCE(shapes[shapes.size() - 1] == 1,
                   "var %s: Fetched var has wrong shape, "
                   "only variables with the last dimension size 1 supported",
                   var_name);
  }

  DataFeedDesc data_feed_desc;
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc);

  int actual_thread_num = thread_num;
  int file_cnt = filelist.size();
  PADDLE_ENFORCE(file_cnt > 0, "File list cannot be empty");

  if (actual_thread_num > file_cnt) {
    VLOG(1) << "Thread num = " << thread_num << ", file num = " << file_cnt
            << ". Changing thread_num = " << file_cnt;
    actual_thread_num = file_cnt;
  }

  /*
    readerDesc: protobuf description for reader initlization
    argument: class_name, batch_size, use_slot, queue_size, buffer_size,
    padding_index

    reader:
    1) each thread has a reader, reader will read input data and
    put it into input queue
    2) each reader has a Next() iterface, that can fetch an instance
    from the input queue
   */
  // todo: should be factory method for creating datafeed
  std::vector<std::shared_ptr<DataFeed>> readers;
  PrepareReaders(readers, actual_thread_num, data_feed_desc, filelist);

  std::vector<std::shared_ptr<ExecutorThreadWorker>> workers;
  workers.resize(actual_thread_num);
  for (auto& worker : workers) {
    worker.reset(new ExecutorThreadWorker);
  }

  // prepare thread resource here
  for (int thidx = 0; thidx < actual_thread_num; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program, readers[thidx],
                  fetch_var_names, root_scope_, thidx, debug);
  }

  // start executing ops in multiple threads
  for (int thidx = 0; thidx < actual_thread_num; ++thidx) {
    threads.push_back(
        std::thread(&ExecutorThreadWorker::TrainFiles, workers[thidx].get()));
  }

  for (auto& th : threads) {
    th.join();
  }

  root_scope_->DropKids();

  return;
}

}  // einit_modelnd namespace framework
}  // end namespace paddle
