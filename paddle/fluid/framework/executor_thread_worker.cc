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
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "gflags/gflags.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/inference/io.h"
#include "paddle/fluid/platform/place.h"
#include "paddle/fluid/pybind/pybind.h"
namespace paddle {
namespace framework {

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
        _current_version[table_id] = *(std::min_element(version.begin(), version.end()));
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
        LOG(FATAL) << "pull dense failed times more than 20 times";
        exit(-1);
    }

    _pull_dense_status.resize(0);
}

void DensePullThread::increase_thread_version(int thread_id, uint64_t table_id) {
    std::lock_guard<std::mutex> lock(_mutex_for_version);
    _training_versions[table_id][thread_id]++;
}
    
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
      root_scope_, "root_scope should be set before creating thread scope");

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

void ExecutorThreadWorker::SetPSlibPtr(std::shared_ptr<paddle::distributed::PSlib> pslib_ptr) {

}


void ExecutorThreadWorker::SetDevice() {
#if defined _WIN32 || defined __APPLE__
  return;
#else
  static unsigned concurrency_cap = std::thread::hardware_concurrency();
  int thread_id = this->thread_id_;

  if (thread_id < concurrency_cap) {
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

void print_fetch_var(Scope* scope, std::string var_name) {
  const LoDTensor& tensor = scope->FindVar(var_name)->Get<LoDTensor>();

  if (std::type_index(tensor.type()) ==
      std::type_index(typeid(platform::float16))) {
    print_lod_tensor<platform::float16>(var_name, tensor);
  } else if (std::type_index(tensor.type()) == std::type_index(typeid(float))) {
    print_lod_tensor<float>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(double))) {
    print_lod_tensor<double>(var_name, tensor);
  } else if (std::type_index(tensor.type()) == std::type_index(typeid(int))) {
    print_lod_tensor<int>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(int64_t))) {
    print_lod_tensor<int64_t>(var_name, tensor);
  } else if (std::type_index(tensor.type()) == std::type_index(typeid(bool))) {
    print_lod_tensor<bool>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(uint8_t))) {
    print_lod_tensor<uint8_t>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(int16_t))) {
    print_lod_tensor<int16_t>(var_name, tensor);
  } else if (std::type_index(tensor.type()) ==
             std::type_index(typeid(int8_t))) {
    print_lod_tensor<int8_t>(var_name, tensor);
  } else {
    VLOG(1) << "print_fetch_var: unrecognized data type:"
            << tensor.type().name();
  }

  return;
}

void ExecutorThreadWorker::TrainFiles() {
  // todo: configurable
  SetDevice();

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

//AsyncExecutor
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

void AsyncExecutorThreadWorker::SetPSlibPtr(std::shared_ptr<paddle::distributed::PSlib> pslib_ptr) {
    _pslib_ptr = pslib_ptr;
}
void AsyncExecutorThreadWorker::SetPullDenseThread(std::shared_ptr<DensePullThread> dpt) {
    _pull_dense_thread = dpt;
}
void AsyncExecutorThreadWorker::TrainOneNetwork() {
    PrepareParams();

    for (auto& op : ops_) {
        if (op->Type().find("sgd") != std::string::npos) {
            continue;
        }
        op->Run(*thread_scope_, place_);
    }

    UpdateParams();
}

void AsyncExecutorThreadWorker::BindingSlotVariableMemory() {
    /*
    std::vector<int> ins_slot_offset(batch_size + 1, 0);
    for (auto i = 1u; i <= batch_size; ++i) {
        ins_slot_offset[i] += ins_slot_offset[i - 1] + slot_dim;
    }

    std::vector<int> tensor_lod(batch_size + 1, 0);
    for (auto i = 1u; i <= batch_size; ++i) {
        tensor_lod[i] += tensor_lod[i - 1] + 1;
    }

    auto& used_slots = reader->get_use_slot_alias();
    slot_input_vec.resize(used_slots.size() - 1);
    for (auto slot_idx = 1u; slot_idx < used_slots.size(); ++slot_idx) {
        auto var = slot_input_variable_name[slot_idx];

        auto v = thread_scope->FindVar(var);
        CHECK(v != nullptr) << "var[" << var << "] not found";

        LoDTensor* tensor = v->GetMutable<LoDTensor>();
        float* tensor_ptr = tensor->mutable_data<float>({batch_size, slot_dim}, platform::CPUPlace());
        memset(tensor_ptr, 0, sizeof(float) * ins_slot_offset.back());

        LoD data_lod{tensor_lod};
        tensor->set_lod(data_lod);

        slot_input_vec[slot_idx - 1].reset(tensor);
    }
    */
}
void AsyncExecutorThreadWorker::SetParamConfig(AsyncWorkerParamConfig* pc) {
    _param_config = pc;
}

void AsyncExecutorThreadWorker::PrepareParams() {
    int table_id = 0; //TODO
    PullSparse(table_id);
    for (auto& t : _pull_sparse_status) {
        t.wait();
        auto status = t.get();
        if (status != 0) {
            LOG(ERROR) << "pull sparse failed, status[" << status << "]";
            exit(-1);
        }
    }
    _pull_sparse_status.resize(0);

    FillSparse(table_id);
}

void AsyncExecutorThreadWorker::UpdateParams() {
    //for (auto i = 0u; i < GlobalConfig::instance().dense_table_id.size(); ++i) {//TODO
    for (int i = 0; i < 1; ++i) {
        PushSparse(i); 
    }
    //for (auto i = 0u; i < GlobalConfig::instance().dense_table_id.size(); ++i) {//TODO
    for (int i = 1; i < 2; ++i) {
        PushDense(i);
    }
    int32_t tmp_push_dense_wait_times = _param_config->tmp_push_dense_wait_times; //TODO
    int32_t tmp_push_sparse_wait_times = _param_config->tmp_push_sparse_wait_times; //TODO
    static uint32_t push_dense_wait_times = static_cast<uint32_t>(tmp_push_dense_wait_times);
    static uint32_t push_sparse_wait_times = static_cast<uint32_t>(tmp_push_sparse_wait_times);

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

    //for (auto dense_table_id : GlobalConfig::instance().dense_table_id) {//TODO
        int dense_table_id = 1;
        _pull_dense_thread->increase_thread_version(thread_id_, dense_table_id);
    //}
}

void AsyncExecutorThreadWorker::PushDense(int table_id) {
    //auto table_id = GlobalConfig::instance().dense_table_id[table_id_index]; TODO

    std::vector<paddle::ps::Region> regions;
    //auto& variables = GlobalConfig::instance().dense_gradient_variable_name[table_id];
    std::vector<std::string> variables;
    for (auto& t : variables) {
        Variable* var = thread_scope_->FindVar(t);
        CHECK(var != nullptr) << "var[" << t << "] not found";
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        int count = tensor->numel();
        float* g = tensor->data<float>();
        paddle::ps::Region reg(g, count);
        regions.emplace_back(std::move(reg));
    }

    auto status = _pslib_ptr->_worker_ptr->push_dense(regions.data(), regions.size(), table_id);
    _push_dense_status.push_back(std::move(status));

}

void AsyncExecutorThreadWorker::PullSparse(int table_id) {


    auto& features = _features[table_id];
    auto& feature_value = _feature_value[table_id];
    auto fea_dim = _param_config->fea_dim; //TODO
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
            //todo: current trick - filter feasign=use_slot_mod(bug: datafeed fill use_slot_mod for empty slot)
            if (ids[i] == 0u) {
                continue;
            }
            features.push_back(static_cast<uint64_t>(ids[i]));
        }
    }

    check_pull_push_memory(features, feature_value, fea_dim);

    std::vector<float*> pull_feature_value;
    for (auto i = 0u; i < features.size(); ++i) {
        pull_feature_value.push_back(feature_value[i].data());
    }

    auto status = _pslib_ptr->_worker_ptr->pull_sparse(
            pull_feature_value.data(), table_id, features.data(), features.size());
    _pull_sparse_status.push_back(std::move(status));

    //to save time
    auto& push_g = _feature_push_value[table_id];
    check_pull_push_memory(features, push_g, fea_dim);

    //binding_slot_embed_with_concat(); TODO
    collect_feasign_info(table_id); //TODO
}

void AsyncExecutorThreadWorker::FillSparse(int table_id) {
    auto slot_dim = _param_config->slot_dim; // TODO
    auto fea_dim = _param_config->fea_dim; //TODO
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
        
        Variable* var_emb = thread_scope_->FindVar(_param_config->slot_input_vec[slot_idx - 1]);
        LoDTensor* tensor_emb = var_emb->GetMutable<LoDTensor>();
        float* ptr = tensor_emb->data<float>();

        for (auto index = 0u; index < len; ++index){
            //if (_current_train_job.use_cvm_feature()) {
            //    if (ids[index] == 0u) {
            //        memcpy(ptr + slot_dim * index, init_value.data(), sizeof(float) * slot_dim);
            //        continue;
            //    }
            //    memcpy(ptr + slot_dim * index, fea_value[fea_idx].data(), sizeof(float) * slot_dim);
            //    (ptr + slot_dim * index)[0] = log((ptr + slot_dim * index)[0] + 1);
            //    (ptr + slot_dim * index)[1] = log((ptr + slot_dim * index)[1] + 1) - (ptr + slot_dim * index)[0];
            //    fea_idx++;
            //} else {
                if (ids[index] == 0u) {
                    memcpy(ptr + slot_dim * index, init_value.data() + 2, sizeof(float) * slot_dim);
                    continue;
                }
                memcpy(ptr + slot_dim * index, fea_value[fea_idx].data() + 2, sizeof(float) * slot_dim);
                fea_idx++;
            //}
        }
    }
}

void AsyncExecutorThreadWorker::PushSparse(int table_id) {

    auto slot_dim = _param_config->slot_dim; //TODO
    auto fea_dim = _param_config->fea_dim;//_current_train_job.fea_dim();TODO
    auto& features = _features[table_id];
    //std::vector<std::string> gradient_var;
    //auto& gradient_var = GlobalConfig::instance().input_gradient_variable_name; //TODO
    auto& push_g = _feature_push_value[table_id];
    check_pull_push_memory(features, push_g, fea_dim);
    uint64_t fea_idx = 0u;
    auto& fea_info = _fea_info[table_id]; //TODO
    int offset = 0;
    //if (!_current_train_job.use_cvm_feature()) { //TODO
        offset = 2;
    //}

    const std::vector<std::string>& feed_vec = thread_reader_->GetUseSlotAlias();

    // slot_idx = 0 is label TODO
    for (auto slot_idx = 1u; slot_idx < feed_vec.size(); ++slot_idx) {
        if (_slot_alias_to_table[feed_vec[slot_idx]] != table_id) {
            continue;
        }
        Variable* g_var = thread_scope_->FindVar(_param_config->gradient_var[slot_idx - 1]);
        LoDTensor* g_tensor = g_var->GetMutable<LoDTensor>();
        //int count = g_tensor->numel();
        float* g = g_tensor->data<float>();
        /*
        if (FLAGS_scale_sparse_gradient_with_batch_size) {
            Eigen::Map<Eigen::MatrixXf> g_mat(g, 1, tensor->numel());
            g_mat *= _batch_size;
        }
        */

        Variable* var = thread_scope_->FindVar(feed_vec[slot_idx]);
        LoDTensor* tensor = var->GetMutable<LoDTensor>();
        int len = tensor->lod()[0].back();
        //assert(slot_dim * len == count);
        int64_t* ids = tensor->data<int64_t>();
        for (auto id_idx = 0u; id_idx < len; ++id_idx){
            if (ids[id_idx] == 0) {
                g += slot_dim;
                continue;
            }
            memcpy(push_g[fea_idx].data() + offset, g, sizeof(float) * slot_dim);
            push_g[fea_idx][0] = 1.0f;
            push_g[fea_idx][1] = static_cast<float>(fea_info[fea_idx].label);
            g += slot_dim;
            fea_idx++;
        }
    }
    assert(fea_idx == features.size());
    CHECK(features.size() > 0);

    std::vector<float*> push_g_vec;
    for (auto i = 0u; i < features.size(); ++i) {
        push_g_vec.push_back(push_g[i].data());
    }
    auto status = _pslib_ptr->_worker_ptr->push_sparse(
            table_id, features.data(), (const float**)push_g_vec.data(), features.size());
    _push_sparse_status.push_back(std::move(status));
}

void AsyncExecutorThreadWorker::collect_feasign_info(
        int table_id) {
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
    CHECK(global_index == feature.size()) << "expect fea info size:" << feature.size()
        << " real:" << global_index;
}

void AsyncExecutorThreadWorker::check_pull_push_memory(
        std::vector<uint64_t>& features,
        std::vector<std::vector<float>>& push_g,
        int dim) {
    push_g.resize(features.size() + 1);
    for (auto& t : push_g) {
        t.resize(dim);
    }
}

void AsyncExecutorThreadWorker::check_pull_push_memory(
        std::vector<uint64_t>& features,
        std::vector<float*>& push_g,
        int dim) {
    if (features.size() > push_g.size()) {
        push_g.reserve(features.size() + 1);
        auto size = features.size() - push_g.size() + 1;
        for (auto i = 0u; i < size; ++i) {
            float* ptr = new float[dim];
            push_g.push_back(ptr);
        }
    }
}

}  // einit_modelnd namespace framework
}  // end namespace paddle
