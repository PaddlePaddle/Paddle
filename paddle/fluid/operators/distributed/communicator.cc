/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/operators/distributed/communicator.h"

#include <gflags/gflags.h>
#include <paddle/fluid/framework/program_desc.h>
#include <chrono>  // NOLINT
#include <map>
#include <thread>  // NOLINT
#include <unordered_set>
#include "paddle/fluid/framework/threadpool.h"

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/parameter_recv.h"
#include "paddle/fluid/operators/distributed/parameter_send.h"

DECLARE_int32(communicator_max_merge_var_num);
DECLARE_int32(communicator_send_queue_size);

DEFINE_bool(communicator_independent_recv_thread, true,
            "use an independent to recv vars from parameter server");
DEFINE_int32(communicator_min_send_grad_num_before_recv, 20,
             "max grad num to send before recv parameters");
DEFINE_int32(communicator_thread_pool_size, 5, "thread num to do send or recv");
DEFINE_int32(communicator_send_wait_times, 5,
             "times that send thread will wait if merge num does not reach "
             "max_merge_var_num");
DEFINE_bool(communicator_fake_rpc, false,
            "fake mode does not really send any thing");
DEFINE_bool(communicator_merge_sparse_grad, true,
            "merge sparse gradient before sending");
DEFINE_int32(communicator_merge_sparse_bucket, 2000,
             "number of threads for sparse var");

namespace paddle {
namespace operators {
namespace distributed {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
}

template <typename T>
inline void VSUB(int n, const T *x, const T *y, T *z) const {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] - y[i];
  }
}

inline std::vector<int> bucket(const int v_size, const int b_size) {
  int remainder = v_size % b_size;
  int bucket = v_size / b_size;
  std::vector<int> ret_vec(b_size, bucket);
  for (int i = 0; i < remainder; ++i) {
    ret_vec[i] = ret_vec[i] + 1;
  }
  int cur_bucket = 0;
  for (int j = 0; j < ret_vec.size(); ++j) {
    int tmp = ret_vec[j];
    ret_vec[j] = cur_bucket;
    cur_bucket += tmp;
  }
  ret_vec.push_back(cur_bucket);
  return ret_vec;
}

std::shared_ptr<Communicator> Communicator::communicator_(nullptr);

Communicator::Communicator(const RpcCtxMap &send_varname_to_ctx,
                           const RpcCtxMap &recv_varname_to_ctx,
                           Scope *recv_scope)
    : send_varname_to_ctx_(send_varname_to_ctx),
      recv_varname_to_ctx_(recv_varname_to_ctx),
      recv_scope_(recv_scope) {
  // get all send information from graph, build vars_to_send
  VLOG(0) << "communicator_independent_recv_thread: "
          << FLAGS_communicator_independent_recv_thread;
  VLOG(0) << "communicator_send_queue_size: "
          << FLAGS_communicator_send_queue_size;
  VLOG(0) << "communicator_min_send_grad_num_before_recv: "
          << FLAGS_communicator_min_send_grad_num_before_recv;
  VLOG(0) << "communicator_thread_pool_size: "
          << FLAGS_communicator_thread_pool_size;
  VLOG(0) << "communicator_send_wait_times: "
          << FLAGS_communicator_send_wait_times;
  VLOG(0) << "communicator_max_merge_var_num: "
          << FLAGS_communicator_max_merge_var_num;
  VLOG(0) << "communicator_fake_rpc: " << FLAGS_communicator_fake_rpc;
  VLOG(0) << "communicator_merge_sparse_grad: "
          << FLAGS_communicator_merge_sparse_grad;
  VLOG(0) << "communicator_merge_sparse_bucket: "
          << FLAGS_communicator_merge_sparse_bucket;

  if (send_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be send, will not start send_thread";
  } else {
    send_scope_.reset(new Scope());
    for (auto &iter : send_varname_to_ctx_) {
      send_varname_to_queue_[iter.first] =
          std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(
              FLAGS_communicator_send_queue_size);
    }
    send_threadpool_.reset(
        new ::ThreadPool(FLAGS_communicator_thread_pool_size));
  }

  if (recv_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be received, will not start recv_thread";
  } else {
    recv_threadpool_.reset(
        new ::ThreadPool(FLAGS_communicator_thread_pool_size));
  }
}

Communicator::~Communicator() {
  if (FLAGS_v >= 3) {
    std::string msg("~Communicator");
    fwrite(msg.c_str(), msg.length(), 1, stdout);
  }
  running_ = false;
  if (send_thread_) send_thread_->join();
  if (recv_thread_) recv_thread_->join();
  if (FLAGS_v >= 3) {
    std::string msg("~Communicator done");
    fwrite(msg.c_str(), msg.length(), 1, stdout);
  }
}

void Communicator::SendThread() {
  VLOG(1) << "SendThread start!";
  auto before_run_training = GetCurrentUS();
  while (running_) {
    std::vector<std::future<void>> task_futures;
    task_futures.reserve(send_varname_to_ctx_.size());
    VLOG(3) << "run send graph";
    auto before_run_send_graph = GetCurrentUS();

    if (is_geo_sgd_) {
      if (ids_send_vec_.size() < geo_need_push_nums_) {
        VLOG(3) << "ids_send_queue_ Size: " << ids_send_vec_.size();
        if (need_push_queue_->Size() > 0) {
          ids_send_vec_.push_back(*(need_push_queue_->Pop()));
          VLOG(3) << "ids_send_queue pushed";
        }
      }
      if (ids_send_vec_.size() >= geo_need_push_nums_) {
        auto after_run_training = GetCurrentUS();
        VLOG(1) << "run Training use time "
                << after_run_training - before_run_training;
        before_run_training = GetCurrentUS();
        VLOG(1) << "Start send after get need_push_num";

        for (auto &iter : send_varname_to_ctx_) {
          auto &var_name = iter.first;
          auto send_task = [this, &var_name] {
            auto origin_var_name = DeltaVarToVar(var_name);

            auto before_send = GetCurrentUS();
            if (var_list_[origin_var_name] == true) {
              auto ids_set = SparseIdsMerge(ids_send_vec_, origin_var_name);
              VLOG(1) << "Before send update var name: " << origin_var_name;
              SendUpdateSparseVars(origin_var_name, ids_set);
            } else {
              VLOG(1) << "Before send update var name: " << origin_var_name;
              SendUpdateDenseVars(origin_var_name);
            }
            auto send_functor = distributed::ParameterSend<float>();

            auto &ctx = send_varname_to_ctx_.at(var_name);
            // delta parameter is in delta scope
            if (!FLAGS_communicator_fake_rpc) {
              send_functor(ctx, *delta_scope_.get(), true);
            }
            auto after_send = GetCurrentUS();
            VLOG(1) << "send " << var_name << " use time "
                    << after_send - before_send;
          };
          task_futures.emplace_back(
              send_threadpool_->enqueue(std::move(send_task)));
        }
      }
    } else {
      for (auto &iter : send_varname_to_queue_) {
        auto &var_name = iter.first;
        auto &var_queue = iter.second;
        if (var_queue->Size() > 0) {
          auto send_task = [this, &var_name, &var_queue] {
            VLOG(3) << var_name << " merge and send";
            std::vector<std::shared_ptr<Variable>> vars;
            size_t merged_var_num = 0;
            size_t wait_times = 0;
            while (merged_var_num < FLAGS_communicator_max_merge_var_num) {
              if (var_queue->Size() == 0) {
                VLOG(3) << "wait_times -> " << wait_times;
                if (wait_times >= FLAGS_communicator_send_wait_times) {
                  break;
                }
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
                wait_times++;
                continue;
              } else {
                wait_times = 0;
                vars.push_back(var_queue->Pop());
                // only count the send number of the first var
                if (var_name == send_varname_to_queue_.begin()->first) {
                  grad_num_.fetch_add(1, std::memory_order_relaxed);
                }
                merged_var_num++;
              }
            }
            auto before_merge = GetCurrentUS();
            MergeVars(var_name, vars, send_scope_.get());
            auto after_merge = GetCurrentUS();
            VLOG(3) << "merge " << merged_var_num << " " << var_name
                    << " use time " << after_merge - before_merge;
            auto send_functor = distributed::ParameterSend<float>();
            auto &ctx = send_varname_to_ctx_.at(var_name);
            if (!FLAGS_communicator_fake_rpc) {
              send_functor(ctx, *send_scope_, true);
            }
            auto after_send = GetCurrentUS();
            VLOG(3) << "send " << var_name << " use time "
                    << after_send - after_merge;
          };
          task_futures.emplace_back(
              send_threadpool_->enqueue(std::move(send_task)));
        } else {
          VLOG(4) << var_name << " queue empty";
        }
      }
    }

    for (auto &task_f : task_futures) {
      task_f.wait();
      have_push_.fetch_add(1, std::memory_order_relaxed);
    }
    auto after_run_send_graph = GetCurrentUS();
    if (after_run_send_graph - before_run_send_graph > 100) {
      VLOG(1) << "run send graph use time "
              << after_run_send_graph - before_run_send_graph;
    }

    RecvNonIndependent();
  }
  VLOG(0) << "communicator stopped, send thread exit";
}

void Communicator::RecvNonIndependent() {
  // Todo: Review this option
  if (FLAGS_communicator_independent_recv_thread && !is_geo_sgd_) {
    return;
  }

  if (is_geo_sgd_) {
    auto push_nums = have_push_.load();
    if (push_nums >= send_varname_to_ctx_.size()) {
      ids_send_vec_.clear();
      RecvAll();
      have_push_.store(0);
    }
  } else {
    auto grad_num = grad_num_.load();
    if (grad_num > 0) {
      RecvAll();
      grad_num_.store(0);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
}

void Communicator::RecvAll() {
  VLOG(2) << "parallel run recv graph";
  if (!running_) return;
  auto before_recv = GetCurrentUS();
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(recv_varname_to_ctx_.size());
  for (auto &iter : recv_varname_to_ctx_) {
    auto recv_task = [this, &iter] {
      auto &var_name = iter.first;
      VLOG(2) << "recv var " << var_name;
      auto recv_functor = distributed::ParameterRecv<float>();
      if (!FLAGS_communicator_fake_rpc && !is_geo_sgd_) {
        recv_functor(iter.second, *recv_scope_);
      }
      // for geo-sgd

      if (!FLAGS_communicator_fake_rpc && is_geo_sgd_) {
        auto before_parameter_recv = GetCurrentUS();
        recv_functor(iter.second, *pserver_scope_.get());
        auto after_parameter_recv = GetCurrentUS();
        VLOG(1) << "run parameter recv var " << var_name << " use time "
                << after_parameter_recv - before_parameter_recv;
        RecvUpdateVars(var_name);
      }
    };
    task_futures.emplace_back(recv_threadpool_->enqueue(std::move(recv_task)));
  }
  for (auto &task : task_futures) {
    task.wait();
  }
  auto after_recv = GetCurrentUS();
  VLOG(1) << "run recv graph use time " << after_recv - before_recv;
}

void Communicator::RecvThread() {
  VLOG(1) << "RecvThread start!";
  while (running_) {
    auto grad_num = grad_num_.load();
    if (grad_num > FLAGS_communicator_min_send_grad_num_before_recv) {
      VLOG(1) << "current grad num " << grad_num;
      RecvAll();
      grad_num_.store(0);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  VLOG(0) << "communicator stopped, recv thread exit";
}

void Communicator::Send(const std::string &var_name,
                        const framework::Scope &scope) {
  VLOG(3) << "communicator send " << var_name;
  // for geo sgd
  if (is_geo_sgd_) {
    VLOG(3) << "run into geo sgd communicator Start()";
    GeoSgdStart(var_name, scope);
    return;
  }
  // push var into send queue by var_name
  auto *grad_var = scope.FindVar(var_name);
  PADDLE_ENFORCE(grad_var->IsInitialized(), "grad var should be inited");
  if (grad_var->IsType<framework::SelectedRows>() &&
      !FLAGS_communicator_merge_sparse_grad) {
    auto send_functor = distributed::ParameterSend<float>();
    auto &ctx = send_varname_to_ctx_.at(var_name);
    if (!FLAGS_communicator_fake_rpc) {
      send_functor(ctx, scope, true);
    }
  } else {
    auto tmp_grad_var = std::make_shared<Variable>();
    framework::CopyVariable(*grad_var, tmp_grad_var.get());
    auto &queue = send_varname_to_queue_.at(var_name);
    VLOG(3) << "send " << var_name << " queue size " << queue->Size();
    queue->Push(tmp_grad_var);
  }
}

void Communicator::Init(const paddle::framework::ProgramDesc &program,
                        Scope *param_scope) {
  using RpcCtxMap = operators::distributed::RpcCtxMap;
  VLOG(3) << "ProcessGraph";
  RpcCtxMap send_varname_to_ctx;
  RpcCtxMap recv_varname_to_ctx;
  for (auto *op : program.Block(0).AllOps()) {
    VLOG(3) << "node name " << op->Type();
    if (op->Type() == "send") {
      auto send_var_name = op->Input("X")[0];
      auto send_varnames = boost::get<std::vector<std::string>>(
          op->GetNullableAttr("send_varnames"));
      auto epmap =
          boost::get<std::vector<std::string>>(op->GetNullableAttr("epmap"));
      auto height_section =
          boost::get<std::vector<int64_t>>(op->GetNullableAttr("sections"));
      auto trainer_id = boost::get<int>(op->GetNullableAttr("trainer_id"));
      send_varname_to_ctx[send_var_name] = operators::distributed::RpcContext(
          send_var_name, send_varnames, epmap, height_section, trainer_id);
      VLOG(3) << "find and init an send op: "
              << send_varname_to_ctx[send_var_name];
    } else if (op->Type() == "recv") {
      auto do_not_run = boost::get<int>(op->GetNullableAttr("do_not_run"));
      PADDLE_ENFORCE_GT(do_not_run, 0, "recv should not run!");
      auto recv_var_name = op->Output("Out")[0];
      auto recv_varnames = boost::get<std::vector<std::string>>(
          op->GetNullableAttr("recv_varnames"));
      auto epmap =
          boost::get<std::vector<std::string>>(op->GetNullableAttr("epmap"));
      auto trainer_id = boost::get<int>(op->GetNullableAttr("trainer_id"));
      recv_varname_to_ctx[recv_var_name] = operators::distributed::RpcContext(
          recv_var_name, recv_varnames, epmap, {}, trainer_id);
      VLOG(3) << "find and init an recv op: "
              << recv_varname_to_ctx[recv_var_name];
    }
  }
  // init communicator here
  if (send_varname_to_ctx.size() == 0 && recv_varname_to_ctx.size() == 0) {
    LOG(WARNING) << "no var need to send and recv!!";
  }
  operators::distributed::Communicator::Init(send_varname_to_ctx,
                                             recv_varname_to_ctx, param_scope);
}

Communicator *Communicator::GetInstance() { return communicator_.get(); }

std::shared_ptr<Communicator> Communicator::GetInstantcePtr() {
  return communicator_;
}

void Communicator::Start() {
  VLOG(0) << "Communicator start";
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    VLOG(1) << "start send thread and recv thread";
    running_ = true;
    // start send and recv thread
    send_thread_.reset(
        new std::thread(std::bind(&Communicator::SendThread, this)));
    if (FLAGS_communicator_independent_recv_thread && !is_geo_sgd_) {
      recv_thread_.reset(
          new std::thread(std::bind(&Communicator::RecvThread, this)));
    }
  }
}

void Communicator::Stop() {
  VLOG(0) << "Communicator stop";
  running_ = false;
  if (!communicator_) {
    VLOG(0) << "Communicator is not inited, do nothing";
  } else {
    if (send_thread_) {
      VLOG(1) << "stop send thread";
      send_thread_->join();
      send_thread_.reset(nullptr);
    }
    if (recv_thread_) {
      VLOG(1) << "stop recv thread";
      recv_thread_->join();
      recv_thread_.reset(nullptr);
    }
  }
  VLOG(0) << "Communicator stop done";
}

// follow functions are created for geo-sgd mode
Communicator::Communicator(const RpcCtxMap &send_varname_to_ctx,
                           const RpcCtxMap &recv_varname_to_ctx,
                           Scope *recv_scope, int &trainers,
                           int &geo_need_push_nums,
                           std::unordered_map<std::string, bool> &var_list)
    : send_varname_to_ctx_(send_varname_to_ctx),
      recv_varname_to_ctx_(recv_varname_to_ctx),
      recv_scope_(recv_scope),
      trainer_nums_(trainers),
      geo_need_push_nums_(geo_need_push_nums),
      var_list_(var_list) {
  // get all send information from graph, build vars_to_send
  VLOG(0) << "communicator_independent_recv_thread: "
          << FLAGS_communicator_independent_recv_thread;
  VLOG(0) << "communicator_send_queue_size: "
          << FLAGS_communicator_send_queue_size;
  VLOG(0) << "communicator_min_send_grad_num_before_recv: "
          << FLAGS_communicator_min_send_grad_num_before_recv;
  VLOG(0) << "communicator_thread_pool_size: "
          << FLAGS_communicator_thread_pool_size;
  VLOG(0) << "communicator_send_wait_times: "
          << FLAGS_communicator_send_wait_times;
  VLOG(0) << "communicator_max_merge_var_num: "
          << FLAGS_communicator_max_merge_var_num;
  VLOG(0) << "communicator_fake_rpc: " << FLAGS_communicator_fake_rpc;
  VLOG(0) << "communicator_merge_sparse_grad: "
          << FLAGS_communicator_merge_sparse_grad;
  VLOG(0) << "Trainer nums: " << trainer_nums_;
  VLOG(0) << "geo_sgd_push_before_local_train_nums: " << geo_need_push_nums_;
  VLOG(0) << "communicator_merge_sparse_bucket"
          << FLAGS_communicator_merge_sparse_bucket;

  if (send_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be send, will not start send_thread";
  } else {
    for (auto &iter : send_varname_to_ctx_) {
      send_varname_to_queue_[iter.first] =
          std::make_shared<BlockingQueue<std::shared_ptr<Variable>>>(
              FLAGS_communicator_send_queue_size);
      VLOG(1) << "send_varname_to_queue " << iter.first << " done";
    }
    send_threadpool_.reset(
        new ::ThreadPool(FLAGS_communicator_thread_pool_size));
  }
  if (recv_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be received, will not start recv_thread";
  } else {
    recv_threadpool_.reset(
        new ::ThreadPool(FLAGS_communicator_thread_pool_size));
  }
  is_geo_sgd_ = true;
  FLAGS_communicator_independent_recv_thread = false;
  VLOG(1) << "geo sgd push nums: " << geo_need_push_nums;
  need_push_queue_ =
      std::make_shared<BlockingQueue<std::shared_ptr<SparseIdsMap>>>(
          geo_need_push_nums);

  delta_scope_.reset(new Scope());

  old_scope_.reset(
      new Scope());  // parameter local, storage the param after last recv
  VLOG(1) << "Init old scope";
  GeoSgdParamInit(old_scope_.get());

  pserver_scope_.reset(new Scope());  // parameter on pserver, global scope
  VLOG(1) << "Init pserver(global) scope";
  GeoSgdParamInit(pserver_scope_.get());
}

void Communicator::GeoSgdInit(
    const paddle::framework::ProgramDesc &program, Scope *param_scope,
    std::map<std::string, std::map<std::string, std::vector<std::string>>>
        &vars_info,
    int &trainers, int &geo_need_push_nums) {
  VLOG(0) << "ProcessGraph Geo_Sgd_Communicator";
  RpcCtxMap send_varname_to_ctx;
  RpcCtxMap recv_varname_to_ctx;
  std::unordered_map<std::string, bool> var_list;
  for (auto &iter : vars_info) {
    std::string var_name = iter.first;
    std::string send_var_name = var_name;
    std::string delta_var_name = send_var_name.append(".delta");
    std::vector<std::string> vars_names = iter.second["var_names"];
    std::vector<std::string> delta_var_names;
    for (auto origin_var_name : vars_names) {
      std::string delta_name = origin_var_name.append(".delta");
      delta_var_names.push_back(delta_name);
    }
    std::vector<std::string> vars_sections_str = iter.second["sections"];
    std::vector<int64_t> vars_sections_int = {};
    for (std::string str : vars_sections_str) {
      int64_t str2i = std::stol(str.c_str());
      vars_sections_int.push_back(str2i);
    }
    std::vector<std::string> vars_epmap = iter.second["epmap"];
    bool is_sparse = iter.second["is_sparse"].front() == std::string("True");
    var_list[var_name] = is_sparse;
    int trainer_id = 0;
    send_varname_to_ctx[delta_var_name] = operators::distributed::RpcContext(
        delta_var_name, delta_var_names, vars_epmap, vars_sections_int,
        trainer_id);
    recv_varname_to_ctx[var_name] = operators::distributed::RpcContext(
        var_name, vars_names, vars_epmap, vars_sections_int, trainer_id);
    VLOG(1) << "find and init an send&recv param: "
            << send_varname_to_ctx[delta_var_name]
            << "is sparse: " << is_sparse;
  }

  // init communicator here
  if (send_varname_to_ctx.size() == 0 && recv_varname_to_ctx.size() == 0) {
    LOG(WARNING) << "no var need to send and recv!!";
  }
  Communicator::Init(send_varname_to_ctx, recv_varname_to_ctx, param_scope,
                     trainers, geo_need_push_nums, var_list);
}

void Communicator::GeoSgdStart(const std::string &var_name,
                               const framework::Scope &scope) {
  if (var_name == "param_init") {
    // when execute trainer startup program, recv init parameter from pserver
    // old_scope param will copy it for storage
    VLOG(1) << "Parameter init from recv_scope";
    for (auto &iter : recv_varname_to_ctx_) {
      auto local_var_name = iter.first;
      GeoSgdParamCopy(*recv_scope_, *old_scope_.get(), local_var_name);
      if (var_list_[local_var_name] == true) {
        // sparse param
        GeoSgdSparseParamInit(*recv_scope_, *pserver_scope_.get(),
                              local_var_name);
      } else {
        GeoSgdParamCopy(*recv_scope_, *pserver_scope_.get(), local_var_name);
      }
    }
    return;
  }
}

void Communicator::GeoSgdSend(const std::vector<std::string> &sparse_var_names,
                              const std::vector<std::string> &sparse_var_tables,
                              const framework::Scope &scope) {
  // trainer thread send sparse ids after batch training

  VLOG(4) << "Geo Sgd Send Sparse ids, shape: " << sparse_var_names.size()
          << " using scope: " << &scope;

  // SparseIdsMap = std::unordered_map<std::string,std::unordered_set<int64_t>>
  std::shared_ptr<SparseIdsMap> ids_table = std::make_shared<SparseIdsMap>();
  for (size_t i = 1; i < sparse_var_tables.size(); i++) {
    // sparse_var_tables first(i=0) element is "FLAG_GEO_SGD_SPARSE_PARAMETER",
    // skip it
    if (ids_table->find(sparse_var_tables[i]) == ids_table->end()) {
      // create empty set for new sparse var
      ids_table->insert(std::pair<std::string, std::unordered_set<int64_t>>(
          sparse_var_tables[i], std::unordered_set<int64_t>{}));
    }
    auto *var = scope.FindVar(sparse_var_names[i - 1]);
    auto var_tensor = var->Get<framework::LoDTensor>();
    int element_number = var_tensor.numel();
    int *var_mutable_data = var_tensor.mutable_data<int>(var_tensor.place());
    // insert ids which has not been record
    for (size_t j = 0; j < element_number; j++) {
      if (ids_table->at(sparse_var_tables[i]).find(var_mutable_data[j]) ==
          ids_table->at(sparse_var_tables[i]).end()) {
        ids_table->at(sparse_var_tables[i]).insert(var_mutable_data[j]);
        VLOG(4) << "Sparse var " << sparse_var_tables[i] << " insert "
                << var_mutable_data[j];
      }
    }
  }
  need_push_queue_->Push(ids_table);

  VLOG(4) << "GeoSgd send complete";
}

std::unordered_set<int64_t> Communicator::SparseIdsMerge(
    std::vector<SparseIdsMap> &ids_send_vec, const std::string &var_name) {
  auto before_run_ids_merge_ = GetCurrentUS();
  std::unordered_set<int64_t> ids_set;
  VLOG(2) << "Sparse ids merge name: " << var_name;
  VLOG(2) << "ids_send_vec Size: " << ids_send_vec.size();
  for (auto table : ids_send_vec) {
    for (auto ids : table[var_name]) {
      if (ids_set.find(ids) == ids_set.end()) {
        ids_set.insert(ids);
      }
    }
  }
  auto after_run_ids_merge_ = GetCurrentUS();
  VLOG(1) << "run SparseIdsMerge use time "
          << after_run_ids_merge_ - before_run_ids_merge_;
  return ids_set;
}

void Communicator::SendUpdateDenseVars(const std::string &var_name) {
  // calc var_delata = (var_recv - var_old)/trainer_nums
  // calc var_old += var_delta

  VLOG(2) << "Geo-Sgd Communicator Send update Dense Vars: " << var_name;
  auto before_run_send_dense = GetCurrentUS();

  auto *var_x = recv_scope_->FindVar(var_name);
  auto *var_y = old_scope_->FindVar(var_name);

  auto var_x_tensor = var_x->Get<framework::LoDTensor>();
  auto var_y_tensor = var_y->Get<framework::LoDTensor>();

  auto cpu_ctx = paddle::platform::CPUDeviceContext();
  auto dims = var_x_tensor.dims();

  VLOG(2) << "Var " << var_name << " Dim[0]: " << dims[0] << " Dim[1] "
          << dims[1];

  // create temp var for sub
  auto *var_y_sub = old_scope_->Var(VarToDeltaVar(var_name));
  framework::CopyVariable(*var_y, var_y_sub);
  auto var_y_sub_tensor = var_y_sub->Get<framework::LoDTensor>();

  // create delta var in delta scope
  auto *var_z = delta_scope_->Var(VarToDeltaVar(var_name));
  auto *var_z_tensor = var_z->GetMutable<framework::LoDTensor>();
  var_z_tensor->mutable_data<float>(dims, var_x_tensor.place());
  var_z_tensor->set_lod(var_x_tensor.lod());

  math::SetConstant<paddle::platform::CPUDeviceContext, float> constant_functor;
  constant_functor(cpu_ctx, var_z_tensor, static_cast<float>(0));

  // calc sub = var_recv - var_old
  auto blas = math::GetBlas<paddle::platform::CPUDeviceContext, float>(cpu_ctx);
  blas.SCAL(var_y_sub_tensor.numel(), -1,
            var_y_sub_tensor.mutable_data<float>(var_y_sub_tensor.place()));
  blas.VADD(var_x_tensor.numel(),
            var_x_tensor.mutable_data<float>(var_x_tensor.place()),
            var_y_sub_tensor.mutable_data<float>(var_y_sub_tensor.place()),
            var_z_tensor->mutable_data<float>(var_z_tensor->place()));
  // calc var_delta = sub / trainer_nums
  float trainer_param = 1.0 / static_cast<float>(trainer_nums_);
  blas.SCAL(var_z_tensor->numel(), trainer_param,
            var_z_tensor->mutable_data<float>(var_z_tensor->place()));
  // calc var_old += var_delta
  blas.VADD(var_y_tensor.numel(),
            var_y_tensor.mutable_data<float>(var_y_tensor.place()),
            var_z_tensor->mutable_data<float>(var_z_tensor->place()),
            var_y_tensor.mutable_data<float>(var_y_tensor.place()));

  auto after_run_send_dense = GetCurrentUS();
  VLOG(1) << "run send update dense var " << var_name << " use time "
          << after_run_send_dense - before_run_send_dense;
}

void Communicator::SendUpdateSparseVars(
    const std::string &var_name, std::unordered_set<int64_t> &ids_table) {
  VLOG(2) << "Geo-Sgd Communicator Send update Sparse Vars: " << var_name;
  auto before_run_send_sparse = GetCurrentUS();

  auto ids_num = ids_table.size();
  VLOG(2) << "Ids nums is : " << ids_num;
  auto *var_x = recv_scope_->FindVar(var_name);
  auto *var_y = old_scope_.get()->FindVar(var_name);
  auto var_x_tensor = var_x->Get<framework::LoDTensor>();
  auto var_y_tensor = var_y->Get<framework::LoDTensor>();

  auto dims = var_x_tensor.dims();
  auto rows = dims[0];
  auto row_numel = dims[1];
  VLOG(2) << "Sparse var dims[0]: " << rows << " dims[1]: " << row_numel;
  float *x_mutable_data =
      var_x_tensor.mutable_data<float>(var_x_tensor.place());
  float *y_mutable_data =
      var_y_tensor.mutable_data<float>(var_y_tensor.place());

  auto *var_z = delta_scope_->Var(VarToDeltaVar(var_name));
  auto *var_z_select_rows = var_z->GetMutable<framework::SelectedRows>();

  var_z_select_rows->set_height(rows);

  // copy value
  auto *var_z_value = var_z_select_rows->mutable_value();
  var_z_value->Resize({ids_num, row_numel});
  auto *z_mutable_data = var_z_value->mutable_data<float>(var_x_tensor.place());

  std::vector<int64_t> new_rows;
  new_rows.insert(new_rows.begin(), ids_table.begin(), ids_table.end());
  var_z_select_rows->set_rows(new_rows);

  std::vector<int> buts =
      bucket(new_rows.size(), FLAGS_communicator_merge_sparse_bucket);

  std::vector<std::future<void>> fs;

  for (int x = 0; x < buts.size() - 1; x++) {
    int start = buts[x];
    int end = buts[x + 1];
    float avg = 1 / static_cast<float>(trainer_nums_);

    fs.push_back(
        framework::Async([&x_mutable_data, &y_mutable_data, &z_mutable_data,
                          &new_rows, row_numel, start, end, avg]() {
          auto x_value = x_mutable_data.data<float>();
          auto y_value = y_mutable_data.data<float>();
          auto z_value = z_mutable_data.data<float>();

          auto cpu_ctx = paddle::platform::CPUDeviceContext();
          auto blas =
              math::GetBlas<paddle::platform::CPUDeviceContext, float>(cpu_ctx);

          for (int y = start; y < end; y++) {
            auto ids = new_rows[y];
            std::vector<float> row_diff(row_numel, 0);
            VSUB(x_value, y_value, row_diff.data());
            blas.SCAL(row_numel, avg, row_diff.data());

            float *x_val = x_value + ids * row_numel;
            float *y_val = y_value + ids * row_numel;
            float *z_val = z_value + y * row_numel;

            blas.VADD(row_numel, row_diff.data(), y_val, y_val);
            blas.VCOPY(row_numel, row_diff.data(), z_val);
          }
        }));
  }
  for (size_t i = 0; i < fs.size(); ++i) fs[i].wait();

  auto after_run_send_sparse = GetCurrentUS();
  VLOG(1) << "run send update sparse var " << var_name << " use time "
          << after_run_send_sparse - before_run_send_sparse;
}

void Communicator::RecvUpdateVars(const std::string &var_name) {
  // calc var_recv = var_pserver - var_old
  // calc var_old = var_pserver
  VLOG(1) << "Geo-Sgd Communicator Recv update Vars: " << var_name;
  auto before_run_recv = GetCurrentUS();

  auto *var_x = recv_scope_->FindVar(var_name);
  auto *var_y = old_scope_->FindVar(var_name);

  auto var_x_tensor = var_x->Get<framework::LoDTensor>();
  auto var_y_tensor = var_y->Get<framework::LoDTensor>();

  float *x_mutable_data =
      var_x_tensor.mutable_data<float>(var_x_tensor.place());
  float *y_mutable_data =
      var_y_tensor.mutable_data<float>(var_y_tensor.place());

  if (var_list_[var_name] == true) {
    // sparse param
    auto *var_z = pserver_scope_.get()->FindVar(var_name);
    auto var_z_slr = var_z->GetMutable<framework::SelectedRows>();
    auto &new_rows = var_z_slr->rows();
    auto &new_value = var_z_slr->value();
    int64_t row_numel = new_value.numel() / new_rows.size();
    auto *z_mutable_data = new_value.data<float>();
    VLOG(1) << "Geo-Sgd Recv Sparse var " << var_name << " row size "
            << new_rows.size();
    for (size_t i = 0; i < new_rows.size(); i++) {
      float diff = 0;
      VLOG(2) << "Geo-Sgd Recv " << new_rows[i]
              << " before update Vars recv_scope: "
              << x_mutable_data[new_rows[i] * row_numel]
              << " ;old_scope: " << y_mutable_data[new_rows[i] * row_numel]
              << " ;pserver_scope: " << z_mutable_data[i * row_numel];
      for (int64_t j = 0; j < row_numel; j++) {
        if (j == 0) {
          diff = z_mutable_data[i * row_numel + j] -
                 y_mutable_data[new_rows[i] * row_numel + j];
        }
        x_mutable_data[new_rows[i] * row_numel + j] +=
            (z_mutable_data[i * row_numel + j] -
             y_mutable_data[new_rows[i] * row_numel + j]);
        y_mutable_data[new_rows[i] * row_numel + j] =
            z_mutable_data[i * row_numel + j];
      }
      VLOG(2) << "Geo-Sgd Recv " << new_rows[i]
              << " after update Vars recv_scope: "
              << x_mutable_data[new_rows[i] * row_numel]
              << " ;old_scope: " << y_mutable_data[new_rows[i] * row_numel]
              << " ;pserver_scope: " << z_mutable_data[i * row_numel]
              << " ;diff: " << diff;
    }
  } else {
    // dense param
    auto *var_y_sub = old_scope_->Var(VarToDeltaVar(var_name));
    framework::CopyVariable(*var_y, var_y_sub);
    auto var_y_sub_tensor = var_y_sub->Get<framework::LoDTensor>();

    auto *var_z = pserver_scope_.get()->FindVar(var_name);
    auto var_z_tensor = var_z->Get<framework::LoDTensor>();

    auto cpu_ctx = paddle::platform::CPUDeviceContext();
    auto blas =
        math::GetBlas<paddle::platform::CPUDeviceContext, float>(cpu_ctx);
    // calc sub = pserver - old
    blas.SCAL(var_y_sub_tensor.numel(), -1,
              var_y_sub_tensor.mutable_data<float>(var_y_sub_tensor.place()));
    blas.VADD(var_y_tensor.numel(),
              var_y_sub_tensor.mutable_data<float>(var_y_sub_tensor.place()),
              var_z_tensor.mutable_data<float>(var_z_tensor.place()),
              var_y_sub_tensor.mutable_data<float>(var_y_sub_tensor.place()));
    // calc recv += sub
    blas.VADD(var_x_tensor.numel(),
              var_x_tensor.mutable_data<float>(var_x_tensor.place()),
              var_y_sub_tensor.mutable_data<float>(var_y_sub_tensor.place()),
              var_x_tensor.mutable_data<float>(var_x_tensor.place()));
    // calc old = pserver
    framework::CopyVariable(*var_z, var_y);
  }

  auto after_run_recv = GetCurrentUS();
  VLOG(1) << "run recv update var " << var_name << " use time "
          << after_run_recv - before_run_recv;
}

void Communicator::GeoSgdParamCopy(const framework::Scope &scope_x,
                                   const framework::Scope &scope_y,
                                   const std::string var_name) {
  auto *var_x = scope_x.FindVar(var_name);
  auto *var_y = scope_y.FindVar(var_name);
  framework::CopyVariable(*var_x, var_y);
}

void Communicator::GeoSgdSparseParamInit(const framework::Scope &scope_x,
                                         const framework::Scope &scope_y,
                                         const std::string var_name) {
  // create selectedrows var from lodtensor var info
  auto *var_x = scope_x.FindVar(var_name);
  auto *var_y = scope_y.FindVar(var_name);

  auto var_x_tensor = var_x->Get<framework::LoDTensor>();
  auto *var_y_select_rows = var_y->GetMutable<framework::SelectedRows>();

  auto dims = var_x_tensor.dims();
  auto rows = dims[0];
  auto row_numel = dims[1];
  VLOG(1) << "Sparse var dims[0]: " << rows << " dims[1]: " << row_numel;

  var_y_select_rows->set_height(rows);
  std::vector<int64_t> new_rows{};
  var_y_select_rows->set_rows(new_rows);
  auto *var_y_value = var_y_select_rows->mutable_value();
  var_y_value->Resize({rows, row_numel});
  var_y_value->mutable_data<float>(var_x_tensor.place());
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
