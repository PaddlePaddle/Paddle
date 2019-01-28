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

void AsyncExecutor::InitModel() {
  for (auto table_id : _param_config.dense_table_id) {
    std::vector<paddle::ps::Region> regions;
    for (auto& t : _param_config.dense_variable_name[table_id]) {
      Variable* var = root_scope_->FindVar(t);
      CHECK(var != nullptr) << "var[" << t << "] not found";
      LoDTensor* tensor = var->GetMutable<LoDTensor>();

      float* g = tensor->data<float>();
      CHECK(g != nullptr) << "var[" << t << "] value not initialized";

      float init_range = 0.2;
      int rown = tensor->dims()[0];
      init_range /= sqrt(rown);

      std::normal_distribution<float> ndistr(0.0, 1.0);
      for (auto i = 0u; i < tensor->numel(); ++i) {
        g[i] = ndistr(local_random_engine()) * init_range;
      }

      paddle::ps::Region reg(g, tensor->numel());
      regions.emplace_back(std::move(reg));
    }

    auto push_status = _pslib_ptr->_worker_ptr->push_dense_param(
        regions.data(), regions.size(), table_id);
    push_status.wait();
    auto status = push_status.get();
    if (status != 0) {
      LOG(FATAL) << "push dense param failed, status[" << status << "]";
      exit(-1);
    }
  }
}

void AsyncExecutor::SaveModel(const std::string& path) {
  auto ret = _pslib_ptr->_worker_ptr->flush();
  ret.wait();
  ret = _pslib_ptr->_worker_ptr->save(path, 0);
  ret.wait();
  int32_t feasign_cnt = ret.get();
  if (feasign_cnt == -1) {  // (colourful-tree) TODO should be feasign_cnt < 0
    LOG(FATAL) << "save model failed";
    exit(-1);
  }
}

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
    PADDLE_ENFORCE_NOT_NULL(var_desc, "%s is not found.", var_name);
    auto shapes = var_desc->GetShape();
    PADDLE_ENFORCE(shapes[shapes.size() - 1] == 1,
                   "var %s: Fetched var has wrong shape, "
                   "only variables with the last dimension size 1 supported",
                   var_name);
  }

  DataFeedDesc data_feed_desc;
  google::protobuf::TextFormat::ParseFromString(data_feed_desc_str,
                                                &data_feed_desc);

  actual_thread_num = thread_num;
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
#ifdef PADDLE_WITH_PSLIB
  PrepareDenseThread(mode);
#endif
  std::vector<std::shared_ptr<ExecutorThreadWorker>> workers;
  workers.resize(actual_thread_num);
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
  for (int thidx = 0; thidx < actual_thread_num; ++thidx) {
    CreateThreads(workers[thidx].get(), main_program, readers[thidx],
                  fetch_var_names, root_scope_, thidx, debug);
  }

  // start executing ops in multiple threads
  for (int thidx = 0; thidx < actual_thread_num; ++thidx) {
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
#ifdef PADDLE_WITH_PSLIB
  if (mode == "mpi") {
    _pull_dense_thread->stop();
  }
#endif
  root_scope_->DropKids();

  return;
}

}  // einit_modelnd namespace framework
}  // end namespace paddle
