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

#include "paddle/fluid/framework/executor_thread_worker.h"
#include <algorithm>
#include <utility>
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/platform/timer.h"
#include "paddle/fluid/pybind/pybind.h"

// pten
#include "paddle/pten/kernels/declarations.h"
namespace paddle {
namespace framework {

#ifdef PADDLE_WITH_PSLIB
int DensePullThread::start() {
  _running = true;
  _t = std::thread(&DensePullThread::run, this);
  return 0;
}

void DensePullThread::run() {
  while (_running) {
    _pull_dense_status.resize(0);
    for (auto& t : _dense_variable_name) {
      if (check_update_param(t.first)) {
        auto status = pull_dense(t.first);
        _pull_dense_status.emplace_back(std::move(status));
        reset_thread_version(t.first);
      }
    }
    if (_pull_dense_status.size() != 0) {
      wait_all();
    }

    usleep(_sleep_time_ms * 1000);
  }
}
bool DensePullThread::check_update_param(uint64_t table_id) {
  {
    std::lock_guard<std::mutex> lock(_mutex_for_version);
    auto& version = _training_versions[table_id];
    _current_version[table_id] =
        *(std::min_element(version.begin(), version.end()));
  }
  if (_current_version[table_id] - _last_versions[table_id] < _threshold) {
    return false;
  }
  return true;
}

void DensePullThread::reset_thread_version(uint64_t table_id) {
  std::lock_guard<std::mutex> lock(_mutex_for_version);
  _last_versions[table_id] = _current_version[table_id];
}
std::future<int32_t> DensePullThread::pull_dense(uint64_t table_id) {
  auto& regions = _regions[table_id];
  regions.clear();
  auto& variables = _dense_variable_name[table_id];
  regions.resize(variables.size());

  for (auto i = 0u; i < variables.size(); ++i) {
    auto& t = variables[i];
    Variable* var = _root_scope->FindVar(t);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();

    float* w = tensor->data<float>();
    paddle::ps::Region reg(w, tensor->numel());
    regions[i] = std::move(reg);
  }
  return _ps_client->pull_dense(regions.data(), regions.size(), table_id);
}

void DensePullThread::wait_all() {
  for (auto& t : _pull_dense_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(WARNING) << "pull dense failed times:" << ++_pull_dense_fail_times;
    }
  }

  if (_pull_dense_fail_times > 20) {
    PADDLE_THROW(
        platform::errors::Fatal("Pull dense failed more than 20 times."));
    exit(-1);
  }

  _pull_dense_status.resize(0);
}

void DensePullThread::increase_thread_version(int thread_id,
                                              uint64_t table_id) {
  std::lock_guard<std::mutex> lock(_mutex_for_version);
  _training_versions[table_id][thread_id]++;
}
#endif

void ExecutorThreadWorker::CreateThreadOperators(const ProgramDesc& program) {
  auto& block = program.Block(0);
  op_names_.clear();
  for (auto& op_desc : block.AllOps()) {
    std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
    op_names_.push_back(op_desc->Type());
    OperatorBase* local_op_ptr = local_op.release();
    ops_.push_back(local_op_ptr);
    continue;
  }
}

void ExecutorThreadWorker::CreateThreadResource(
    const framework::ProgramDesc& program,
    const paddle::platform::Place& place) {
  CreateThreadScope(program);
  CreateThreadOperators(program);
  SetMainProgram(program);
  SetPlace(place);
}

void ExecutorThreadWorker::CreateThreadScope(const ProgramDesc& program) {
  auto& block = program.Block(0);

  PADDLE_ENFORCE_NOT_NULL(
      root_scope_,
      platform::errors::PreconditionNotMet(
          "root_scope should be set before creating thread scope."));

  thread_scope_ = &root_scope_->NewScope();
  for (auto& var : block.AllVars()) {
    if (var->Persistable()) {
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    } else {
      auto* ptr = thread_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
    }
  }
}

void ExecutorThreadWorker::SetDataFeed(
    const std::shared_ptr<DataFeed>& datafeed) {
  thread_reader_ = datafeed;
}

void ExecutorThreadWorker::BindingDataFeedMemory() {
  const std::vector<std::string>& input_feed =
      thread_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    thread_reader_->AddFeedVar(thread_scope_->Var(name), name);
  }
}

void ExecutorThreadWorker::SetFetchVarNames(
    const std::vector<std::string>& fetch_var_names) {
  fetch_var_names_.clear();
  fetch_var_names_.insert(fetch_var_names_.end(), fetch_var_names.begin(),
                          fetch_var_names.end());
}

void ExecutorThreadWorker::SetDevice() {
#if defined _WIN32 || defined __APPLE__
  return;
#else
  static unsigned concurrency_cap = std::thread::hardware_concurrency();
  LOG(WARNING) << "concurrency capacity " << concurrency_cap;
  int thread_id = this->thread_id_;

  if (static_cast<unsigned>(thread_id) < concurrency_cap) {
    unsigned proc = thread_id;

    cpu_set_t mask;
    CPU_ZERO(&mask);
    CPU_SET(proc, &mask);

    if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
      VLOG(1) << "WARNING: Failed to set thread affinity for thread "
              << thread_id;
    } else {
      CPU_ZERO(&mask);
      if ((0 != sched_getaffinity(0, sizeof(mask), &mask)) ||
          (CPU_ISSET(proc, &mask) == 0)) {
        VLOG(3) << "WARNING: Failed to set thread affinity for thread "
                << thread_id;
      }
    }
  } else {
    VLOG(1) << "WARNING: Failed to set thread affinity for thread "
            << thread_id;
  }
#endif
}

template <typename T>
void print_lod_tensor(std::string var_name, const LoDTensor& lod_tensor) {
  auto inspect = lod_tensor.data<T>();
  auto element_num = lod_tensor.numel();

  std::ostringstream sstream;
  sstream << var_name << " (element num " << element_num << "): [";
  sstream << inspect[0];
  for (int j = 1; j < element_num; ++j) {
    sstream << " " << inspect[j];
  }
  sstream << "]";

  std::cout << sstream.str() << std::endl;
}

static void print_fetch_var(Scope* scope, const std::string& var_name) {
  auto& tensor = scope->FindVar(var_name)->Get<LoDTensor>();

#define PrintLoDTensorCallback(cpp_type, proto_type)                    \
  do {                                                                  \
    if (framework::TransToProtoVarType(tensor.dtype()) == proto_type) { \
      print_lod_tensor<cpp_type>(var_name, tensor);                     \
      return;                                                           \
    }                                                                   \
  } while (0)

  _ForEachDataType_(PrintLoDTensorCallback);
  VLOG(1) << "print_fetch_var: unrecognized data type:" << tensor.dtype();
}

void ExecutorThreadWorker::TrainFilesWithTimer() {
  platform::SetNumThreads(1);
  SetDevice();
  thread_reader_->Start();

  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto& op : ops_) {
    op_name.push_back(op->Type());
  }
  op_total_time.resize(ops_.size());
  for (size_t i = 0; i < op_total_time.size(); ++i) {
    op_total_time[i] = 0.0;
  }
  platform::Timer timeline;
  double total_time = 0.0;
  double read_time = 0.0;
  int cur_batch;
  int batch_cnt = 0;
  timeline.Start();
  while ((cur_batch = thread_reader_->Next()) > 0) {
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    for (size_t i = 0; i < ops_.size(); ++i) {
      timeline.Start();
      ops_[i]->Run(*thread_scope_, place_);
      timeline.Pause();
      op_total_time[i] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }
    ++batch_cnt;
    thread_scope_->DropKids();
    if (thread_id_ == 0) {
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < ops_.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        int fetch_var_num = fetch_var_names_.size();
        for (int i = 0; i < fetch_var_num; ++i) {
          print_fetch_var(thread_scope_, fetch_var_names_[i]);
        }
        fprintf(stderr, "IO percent: %f\n", read_time / total_time);
      }
    }
    timeline.Start();
  }
}

void ExecutorThreadWorker::TrainFiles() {
  platform::SetNumThreads(1);

  // todo: configurable
  // SetDevice();

  int fetch_var_num = fetch_var_names_.size();
  fetch_values_.clear();
  fetch_values_.resize(fetch_var_num);

  thread_reader_->Start();

  int cur_batch;
  int batch_cnt = 0;
  while ((cur_batch = thread_reader_->Next()) > 0) {
    // executor run here
    for (auto& op : ops_) {
      op->Run(*thread_scope_, place_);
    }

    ++batch_cnt;
    thread_scope_->DropKids();

    if (debug_ == false || thread_id_ != 0) {
      continue;
    }

    for (int i = 0; i < fetch_var_num; ++i) {
      print_fetch_var(thread_scope_, fetch_var_names_[i]);
    }  // end for (int i = 0...)
  }    // end while ()
}

void ExecutorThreadWorker::SetThreadId(int tid) { thread_id_ = tid; }

void ExecutorThreadWorker::SetPlace(const platform::Place& place) {
  place_ = place;
}

void ExecutorThreadWorker::SetMainProgram(
    const ProgramDesc& main_program_desc) {
  main_program_.reset(new ProgramDesc(main_program_desc));
}

void ExecutorThreadWorker::SetRootScope(Scope* g_scope) {
  root_scope_ = g_scope;
}

#ifdef PADDLE_WITH_PSLIB
//  AsyncExecutor
void AsyncExecutorThreadWorker::TrainFiles() {
  SetDevice();

  int fetch_var_num = fetch_var_names_.size();
  fetch_values_.clear();
  fetch_values_.resize(fetch_var_num);

  thread_reader_->Start();

  int cur_batch;
  int batch_cnt = 0;
  while ((cur_batch = thread_reader_->Next()) > 0) {
    // executor run here
    TrainOneNetwork();

    ++batch_cnt;
    thread_scope_->DropKids();

    if (debug_ == false || thread_id_ != 0) {
      continue;
    }

    for (int i = 0; i < fetch_var_num; ++i) {
      print_fetch_var(thread_scope_, fetch_var_names_[i]);
    }  // end for (int i = 0...)
  }    // end while ()
}

void AsyncExecutorThreadWorker::SetPSlibPtr(
    std::shared_ptr<paddle::distributed::PSlib> pslib_ptr) {
  _pslib_ptr = pslib_ptr;
}

void AsyncExecutorThreadWorker::SetPullDenseThread(
    std::shared_ptr<DensePullThread> dpt) {
  _pull_dense_thread = dpt;
}

void AsyncExecutorThreadWorker::TrainOneNetwork() {
  PrepareParams();

  for (auto& op : ops_) {
    if (op->Type().find("sgd") != std::string::npos) {
      continue;
    }
    bool need_skip = false;
    for (auto t = 0u; t < _param_config->skip_op.size(); ++t) {
      if (op->Type().find(_param_config->skip_op[t]) != std::string::npos) {
        need_skip = true;
        break;
      }
    }
    if (!need_skip) {
      op->Run(*thread_scope_, place_);
    }
  }
  UpdateParams();
}

void AsyncExecutorThreadWorker::SetParamConfig(
    AsyncWorkerParamConfig* param_config) {
  _param_config = param_config;
}

void AsyncExecutorThreadWorker::PrepareParams() {
  for (auto table_id : _param_config->sparse_table_id) {
    PullSparse(table_id);
    for (auto& t : _pull_sparse_status) {
      t.wait();
      auto status = t.get();
      if (status != 0) {
        LOG(ERROR) << "pull sparse failed, status[" << status << "]";
        exit(-1);
      }
    }
  }
  _pull_sparse_status.resize(0);

  for (auto table_id : _param_config->sparse_table_id) {
    FillSparse(table_id);
  }
}

void AsyncExecutorThreadWorker::UpdateParams() {
  for (auto i : _param_config->sparse_table_id) {
    PushSparse(i);
  }
  for (auto i : _param_config->dense_table_id) {
    PushDense(i);
  }
  int32_t tmp_push_dense_wait_times = -1;
  int32_t tmp_push_sparse_wait_times = -1;
  static uint32_t push_dense_wait_times =
      static_cast<uint32_t>(tmp_push_dense_wait_times);
  static uint32_t push_sparse_wait_times =
      static_cast<uint32_t>(tmp_push_sparse_wait_times);

  if (_push_dense_status.size() >= push_dense_wait_times) {
    for (auto& t : _push_dense_status) {
      t.wait();
    }
    _push_dense_status.resize(0);
  }
  if (tmp_push_dense_wait_times == -1) {
    _push_dense_status.resize(0);
  }
  if (_push_sparse_status.size() >= push_sparse_wait_times) {
    for (auto& t : _push_sparse_status) {
      t.wait();
    }
    _push_sparse_status.resize(0);
  }
  if (tmp_push_sparse_wait_times == -1) {
    _push_sparse_status.resize(0);
  }
  for (auto dense_table_id : _param_config->dense_table_id) {
    _pull_dense_thread->increase_thread_version(thread_id_, dense_table_id);
  }
}

void AsyncExecutorThreadWorker::PushDense(int table_id) {
  std::vector<paddle::ps::Region> regions;
  for (auto& t : _param_config->dense_gradient_variable_name[table_id]) {
    Variable* var = thread_scope_->FindVar(t);
    CHECK(var != nullptr) << "var[" << t << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int count = tensor->numel();
    float* g = tensor->data<float>();
    paddle::ps::Region reg(g, count);
    regions.emplace_back(std::move(reg));
  }

  auto status = _pslib_ptr->_worker_ptr->push_dense(regions.data(),
                                                    regions.size(), table_id);
  _push_dense_status.push_back(std::move(status));
}

void AsyncExecutorThreadWorker::PullSparse(int table_id) {
  auto& features = _features[table_id];
  auto& feature_value = _feature_value[table_id];
  auto fea_dim = _param_config->fea_dim;
  // slot id starts from 1
  features.clear();
  features.resize(0);
  features.reserve(MAX_FEASIGN_NUM);
  const std::vector<std::string>& feed_vec = thread_reader_->GetUseSlotAlias();
  // slot_idx = 0 is label TODO
  for (auto slot_idx = 1u; slot_idx < feed_vec.size(); ++slot_idx) {
    Variable* var = thread_scope_->FindVar(feed_vec[slot_idx]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    for (auto i = 0u; i < len; ++i) {
      // todo(colourful-tree): current trick - filter feasign=use_slot_mod(
      // bug: datafeed fill use_slot_mod for empty slot)
      if (ids[i] == 0u) {
        continue;
      }
      features.push_back(static_cast<uint64_t>(ids[i]));
    }
  }
  check_pull_push_memory(features, &feature_value, fea_dim);

  std::vector<float*> pull_feature_value;
  for (auto i = 0u; i < features.size(); ++i) {
    pull_feature_value.push_back(feature_value[i].data());
  }

  auto status = _pslib_ptr->_worker_ptr->pull_sparse(
      pull_feature_value.data(), table_id, features.data(), features.size());
  _pull_sparse_status.push_back(std::move(status));

  auto& push_g = _feature_push_value[table_id];
  check_pull_push_memory(features, &push_g, fea_dim);
  collect_feasign_info(table_id);
}

void AsyncExecutorThreadWorker::FillSparse(int table_id) {
  auto slot_dim = _param_config->slot_dim;
  auto fea_dim = _param_config->fea_dim;
  auto& features = _features[table_id];
  auto& fea_value = _feature_value[table_id];

  CHECK(features.size() > 0) << "feature size check failed";

  auto fea_idx = 0u;

  std::vector<float> init_value(fea_dim);

  const std::vector<std::string>& feed_vec = thread_reader_->GetUseSlotAlias();
  // slot_idx = 0 is label TODO
  for (auto slot_idx = 1u; slot_idx < feed_vec.size(); ++slot_idx) {
    Variable* var = thread_scope_->FindVar(feed_vec[slot_idx]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();
    int len = tensor->numel();
    Variable* var_emb = thread_scope_->FindVar(
        _param_config->slot_input_vec[table_id][slot_idx - 1]);
    LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
    float* ptr =
        tensor_emb->mutable_data<float>({len, slot_dim}, platform::CPUPlace());
    memset(ptr, 0, sizeof(float) * len * slot_dim);
    auto& tensor_lod = tensor->lod()[0];

    LoD data_lod{tensor_lod};
    tensor_emb->set_lod(data_lod);

    for (auto index = 0u; index < len; ++index) {
      if (ids[index] == 0u) {
        memcpy(ptr + slot_dim * index, init_value.data() + 2,
               sizeof(float) * slot_dim);
        continue;
      }
      memcpy(ptr + slot_dim * index, fea_value[fea_idx].data() + 2,
             sizeof(float) * slot_dim);
      fea_idx++;
    }
  }
}

void AsyncExecutorThreadWorker::PushSparse(int table_id) {
  auto slot_dim = _param_config->slot_dim;
  auto fea_dim = _param_config->fea_dim;
  auto& features = _features[table_id];
  auto& push_g = _feature_push_value[table_id];
  check_pull_push_memory(features, &push_g, fea_dim);
  CHECK(push_g.size() == features.size() + 1)
      << "push_g size:" << push_g.size()
      << " features size:" << features.size();
  uint64_t fea_idx = 0u;
  auto& fea_info = _fea_info[table_id];
  int offset = 2;
  const std::vector<std::string>& feed_vec = thread_reader_->GetUseSlotAlias();
  // slot_idx = 0 is label
  for (auto slot_idx = 1u; slot_idx < feed_vec.size(); ++slot_idx) {
    if (_param_config->slot_alias_to_table.find(feed_vec[slot_idx]) ==
        _param_config->slot_alias_to_table.end()) {
      LOG(ERROR) << "ERROR slot_idx:" << slot_idx
                 << " name:" << feed_vec[slot_idx];
    } else if (_param_config->slot_alias_to_table[feed_vec[slot_idx]] !=
               table_id) {
      continue;
    }
    Variable* g_var = thread_scope_->FindVar(
        _param_config->gradient_var[table_id][slot_idx - 1]);
    CHECK(g_var != nullptr)
        << "var[" << _param_config->gradient_var[table_id][slot_idx - 1]
        << "] not found";
    LoDTensor* g_tensor = g_var->GetMutable<LoDTensor>();
    if (g_tensor == NULL) {
      LOG(ERROR) << "var["
                 << _param_config->gradient_var[table_id][slot_idx - 1]
                 << "] not found";
      exit(-1);
    }
    float* g = g_tensor->data<float>();

    Variable* var = thread_scope_->FindVar(feed_vec[slot_idx]);
    CHECK(var != nullptr) << "var[" << feed_vec[slot_idx] << "] not found";
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    if (tensor == NULL) {
      LOG(ERROR) << "var[" << feed_vec[slot_idx] << "] not found";
      exit(-1);
    }
    int len = tensor->numel();
    CHECK(slot_dim * len == g_tensor->numel())
        << "len:" << len << " g_numel:" << g_tensor->numel();
    CHECK(len == tensor->numel()) << "len:" << len
                                  << "t_numel:" << tensor->numel();
    int64_t* ids = tensor->data<int64_t>();
    for (auto id_idx = 0u; id_idx < len; ++id_idx) {
      if (ids[id_idx] == 0) {
        g += slot_dim;
        continue;
      }
      memcpy(push_g[fea_idx].data() + offset, g, sizeof(float) * slot_dim);
      push_g[fea_idx][0] = 1.0f;
      CHECK(fea_idx < fea_info.size()) << "fea_idx:" << fea_idx
                                       << " size:" << fea_info.size();
      push_g[fea_idx][1] = static_cast<float>(fea_info[fea_idx].label);
      g += slot_dim;
      fea_idx++;
    }
  }
  CHECK(fea_idx == features.size()) << "fea_idx:" << fea_idx
                                    << " features size:" << features.size();
  CHECK_GT(features.size(), 0);

  std::vector<float*> push_g_vec;
  for (auto i = 0u; i < features.size(); ++i) {
    push_g_vec.push_back(push_g[i].data());
  }
  auto status = _pslib_ptr->_worker_ptr->push_sparse(
      table_id, features.data(), (const float**)push_g_vec.data(),
      features.size());
  _push_sparse_status.push_back(std::move(status));
}

void AsyncExecutorThreadWorker::collect_feasign_info(int table_id) {
  auto& fea_info = _fea_info[table_id];
  auto& feature = _features[table_id];
  fea_info.resize(feature.size());
  const std::vector<std::string>& feed_vec = thread_reader_->GetUseSlotAlias();
  Variable* var = thread_scope_->FindVar(feed_vec[0]);
  LoDTensor* tensor = var->GetMutable<LoDTensor>();
  int64_t* label = tensor->data<int64_t>();

  int global_index = 0;
  for (auto slot_idx = 1u; slot_idx < feed_vec.size(); ++slot_idx) {
    Variable* var = thread_scope_->FindVar(feed_vec[slot_idx]);
    LoDTensor* tensor = var->GetMutable<LoDTensor>();
    int64_t* ids = tensor->data<int64_t>();

    int fea_idx = 0;
    for (auto ins_idx = 1u; ins_idx < tensor->lod()[0].size(); ++ins_idx) {
      for (; fea_idx < tensor->lod()[0][ins_idx]; ++fea_idx) {
        if (ids[fea_idx] == 0u) {
          continue;
        }
        FeasignInfo info{slot_idx, ins_idx, label[ins_idx - 1]};

        fea_info[global_index++] = std::move(info);
      }
    }
  }
  CHECK(global_index == feature.size())
      << "expect fea info size:" << feature.size() << " real:" << global_index;
}

void AsyncExecutorThreadWorker::check_pull_push_memory(
    const std::vector<uint64_t>& features,
    std::vector<std::vector<float>>* push_g, int dim) {
  push_g->resize(features.size() + 1);
  for (auto& t : *push_g) {
    t.resize(dim);
  }
}

void AsyncExecutorThreadWorker::check_pull_push_memory(
    const std::vector<uint64_t>& features, std::vector<float*>* push_g,
    int dim) {
  if (features.size() > push_g->size()) {
    push_g->reserve(features.size() + 1);
    auto size = features.size() - push_g->size() + 1;
    for (auto i = 0u; i < size; ++i) {
      float* ptr = new float[dim];
      push_g->push_back(ptr);
    }
  }
}
#endif

}  // einit_modelnd namespace framework
}  // end namespace paddle
