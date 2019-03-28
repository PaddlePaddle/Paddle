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
#ifdef PADDLE_WITH_PSLIB
#include <pslib.h>
#endif

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
#ifdef PADDLE_WITH_PSLIB
  worker->SetPSlibPtr(_pslib_ptr);
  worker->SetPullDenseThread(_pull_dense_thread);
  worker->SetParamConfig(&_param_config);
#endif
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

#ifdef PADDLE_WITH_PSLIB
void AsyncExecutor::InitServer(const std::string& dist_desc, int index) {
  _pslib_ptr = std::shared_ptr<paddle::distributed::PSlib>(
      new paddle::distributed::PSlib());
  _pslib_ptr->init_server(dist_desc, index);
  InitParamConfig();
}

void AsyncExecutor::InitWorker(const std::string& dist_desc,
                               const std::vector<uint64_t>& host_sign_list,
                               int node_num, int index) {
  _pslib_ptr = std::shared_ptr<paddle::distributed::PSlib>(
      new paddle::distributed::PSlib());
  _pslib_ptr->init_worker(
      dist_desc, const_cast<uint64_t*>(host_sign_list.data()), node_num, index);

  InitParamConfig();
}

uint64_t AsyncExecutor::StartServer() { return _pslib_ptr->run_server(); }

void AsyncExecutor::StopServer() { _pslib_ptr->stop_server(); }

void AsyncExecutor::GatherServers(const std::vector<uint64_t>& host_sign_list,
                                  int node_num) {
  _pslib_ptr->gather_servers(const_cast<uint64_t*>(host_sign_list.data()),
                             node_num);
}

void AsyncExecutor::InitParamConfig() {
  for (int i = 0; i < _pslib_ptr->get_param()
                          ->server_param()
                          .downpour_server_param()
                          .downpour_table_param_size();
       ++i) {
    if (_pslib_ptr->get_param()
            ->server_param()
            .downpour_server_param()
            .downpour_table_param(i)
            .table_class()
            .find("SparseTable") != -1) {
      _param_config.fea_dim = _pslib_ptr->get_param()
                                  ->server_param()
                                  .downpour_server_param()
                                  .downpour_table_param(i)
                                  .accessor()
                                  .fea_dim();
      break;
    }
  }
  _param_config.slot_dim = _param_config.fea_dim - 2;
  _param_config.tmp_push_dense_wait_times = static_cast<int32_t>(
      _pslib_ptr->get_param()->trainer_param().push_dense_per_batch());
  _param_config.tmp_push_sparse_wait_times = static_cast<int32_t>(
      _pslib_ptr->get_param()->trainer_param().push_sparse_per_batch());

  for (auto t = 0u; t < _pslib_ptr->get_param()->trainer_param().skip_op_size();
       ++t) {
    _param_config.skip_op.push_back(
        _pslib_ptr->get_param()->trainer_param().skip_op(t));
  }

  for (auto t = 0u;
       t < _pslib_ptr->get_param()->trainer_param().sparse_table_size(); ++t) {
    auto& table = _pslib_ptr->get_param()->trainer_param().sparse_table(t);
    std::vector<std::string> tmp_sparse_variable_name;
    for (int i = 0u; i < table.slot_value_size(); ++i) {
      tmp_sparse_variable_name.push_back(table.slot_value(i));
      _param_config.slot_alias_to_table[table.slot_key(i)] = table.table_id();
    }
    std::vector<std::string> tmp_sparse_gradient_variable_name;
    for (auto i = 0u; i < table.slot_gradient_size(); ++i) {
      tmp_sparse_gradient_variable_name.push_back(table.slot_gradient(i));
    }
    _param_config.slot_input_vec[table.table_id()] =
        std::move(tmp_sparse_variable_name);
    _param_config.gradient_var[table.table_id()] =
        std::move(tmp_sparse_gradient_variable_name);
    _param_config.sparse_table_id.push_back(table.table_id());
  }

  for (auto t = 0u;
       t < _pslib_ptr->get_param()->trainer_param().dense_table_size(); ++t) {
    auto& table = _pslib_ptr->get_param()->trainer_param().dense_table(t);
    std::vector<std::string> tmp_dense_variable_name;
    for (int i = 0u; i < table.dense_variable_name_size(); ++i) {
      tmp_dense_variable_name.push_back(table.dense_variable_name(i));
    }
    std::vector<std::string> tmp_dense_gradient_variable_name;
    for (auto i = 0u; i < table.dense_gradient_variable_name_size(); ++i) {
      tmp_dense_gradient_variable_name.push_back(
          table.dense_gradient_variable_name(i));
    }
    _param_config.dense_variable_name[table.table_id()] =
        std::move(tmp_dense_variable_name);
    _param_config.dense_gradient_variable_name[table.table_id()] =
        std::move(tmp_dense_gradient_variable_name);
    _param_config.dense_table_id.push_back(table.table_id());
    _param_config.dense_table_size.push_back(table.fea_dim());
  }
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

void AsyncExecutor::PrepareDenseThread(const std::string& mode) {
  if (mode == "mpi") {
    DensePullThreadParam param;
    param.ps_client = _pslib_ptr->_worker_ptr;
    param.threshold = 1;
    param.training_thread_num = actual_thread_num;
    param.root_scope = root_scope_;
    param.dense_params = &_param_config.dense_variable_name;

    _pull_dense_thread =
        std::shared_ptr<DensePullThread>(new DensePullThread(param));
    _pull_dense_thread->start();
  }
}
#endif

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
