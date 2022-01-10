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
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/framework/trainer_factory.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"

// pten
#include "paddle/pten/kernels/declarations.h"

namespace paddle {
namespace framework {
AsyncExecutor::AsyncExecutor(Scope* scope, const platform::Place& place)
    : root_scope_(scope), place_(place) {}

void AsyncExecutor::InitServer(const std::string& dist_desc, int index) {
  fleet_ptr_ = FleetWrapper::GetInstance();
  fleet_ptr_->InitServer(dist_desc, index);
}

void AsyncExecutor::InitWorker(const std::string& dist_desc,
                               const std::vector<uint64_t>& host_sign_list,
                               int node_num, int index) {
  fleet_ptr_ = FleetWrapper::GetInstance();
  fleet_ptr_->InitWorker(dist_desc, host_sign_list, node_num, index);
}

uint64_t AsyncExecutor::StartServer() { return fleet_ptr_->RunServer(); }

void AsyncExecutor::StopServer() { fleet_ptr_->StopServer(); }

void AsyncExecutor::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                  int node_num) {
  fleet_ptr_->GatherServers(host_sign_list, node_num);
}

// todo InitModel
void AsyncExecutor::InitModel() {}

// todo SaveModel
void AsyncExecutor::SaveModel(const std::string& path) {}

void AsyncExecutor::RunFromFile(const ProgramDesc& main_program,
                                const std::string& data_feed_desc_str,
                                const std::vector<std::string>& filelist,
                                const int thread_num,
                                const std::vector<std::string>& fetch_var_names,
                                const std::string& mode, const bool debug) {
  std::vector<std::thread> threads;

  auto& block = main_program.Block(0);
  for (auto var_name : fetch_var_names) {
    auto var_desc = block.FindVar(var_name);
    PADDLE_ENFORCE_NOT_NULL(
        var_desc, platform::errors::NotFound(
                      "Variable %s is not found in main program.", var_name));
    auto shapes = var_desc->GetShape();
    PADDLE_ENFORCE_EQ(shapes[shapes.size() - 1], 1,
        platform::errors::InvalidArgument(
                   "Fetched variable %s has wrong shape, "
                   "only variables whose last dimension is 1 are supported",
                   var_name);
  }

  DataFeedDesc data_feed_desc;
  bool success = data_feed_desc.ParseFromString(data_feed_desc_str);
  PADDLE_ENFORCE_EQ(success, true,
                    platform::errors::InvalidArgument(
                        "Fail to parse DataFeedDesc from string: %s.",
                        data_feed_desc_str.c_str()));

  actual_thread_num_ = thread_num;
  int file_cnt = filelist.size();
  PADDLE_ENFORCE_GT(file_cnt, 0,
                    platform::errors::NotFound("Input file list is empty."));

  if (actual_thread_num_ > file_cnt) {
    VLOG(1) << "Thread num = " << thread_num << ", file num = " << file_cnt
            << ". Changing thread_num = " << file_cnt;
    actual_thread_num_ = file_cnt;
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
  /*
  PrepareReaders(readers, actual_thread_num_, data_feed_desc, filelist);
#ifdef PADDLE_WITH_PSLIB
  PrepareDenseThread(mode);
#endif
  */
  std::vector<std::shared_ptr<ExecutorThreadWorker>> workers;
  workers.resize(actual_thread_num_);
  for (auto& worker : workers) {
#ifdef PADDLE_WITH_PSLIB
    if (mode == "mpi") {
      worker.reset(new AsyncExecutorThreadWorker);
    } else {
      worker.reset(new ExecutorThreadWorker);
    }
#else
    worker.reset(new ExecutorThreadWorker);
#endif
  }

  // prepare thread resource here
  /*
  for (int thidx = 0; thidx < actual_thread_num_; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program, readers[thidx],
                  fetch_var_names, root_scope_, thidx, debug);
  }
  */

  // start executing ops in multiple threads
  for (int thidx = 0; thidx < actual_thread_num_; ++thidx) {
    if (debug) {
      threads.push_back(std::thread(&ExecutorThreadWorker::TrainFilesWithTimer,
                                    workers[thidx].get()));
    } else {
      threads.push_back(
          std::thread(&ExecutorThreadWorker::TrainFiles, workers[thidx].get()));
    }
  }

  for (auto& th : threads) {
    th.join();
  }
  // TODO(guru4elephant): we don't need this
  /*
#ifdef PADDLE_WITH_PSLIB
  if (mode == "mpi") {
    _pull_dense_thread->stop();
  }
#endif
  */
  VLOG(3) << "start to run from files in async_executor";
  VLOG(3) << "Drop current scope kids";
  root_scope_->DropKids();
  return;
}

}  // end namespace framework
}  // end namespace paddle
