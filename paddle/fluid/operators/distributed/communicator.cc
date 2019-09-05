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
#include <thread>  // NOLINT

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


namespace paddle {
namespace operators {
namespace distributed {

inline double GetCurrentUS() {
  struct timeval time;
  gettimeofday(&time, NULL);
  return 1e+6 * time.tv_sec + time.tv_usec;
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
  while (running_) {
    std::vector<std::future<void>> task_futures;
    task_futures.reserve(send_varname_to_ctx_.size());
    VLOG(3) << "run send graph";
    auto before_run_send_graph = GetCurrentUS();
    for (auto &iter : send_varname_to_queue_) {
      auto &var_name = iter.first;
      auto &var_queue = iter.second;
      if (var_queue->Size() > 0) {
        auto send_task = [this, &var_name, &var_queue] {
          if(!is_geo_sgd_) {
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
          }
          else if (is_geo_sgd_) {
            VLOG(1) << "geo sgd send var: "<< var_name;
            auto before_send = GetCurrentUS();
            auto send_functor = distributed::ParameterSend<float>();
            while(var_queue->Size()>0) {
              var_queue->Pop();
            }       
            auto &ctx = send_varname_to_ctx_.at(var_name);
            if (!FLAGS_communicator_fake_rpc) {
              send_functor(ctx, *delta_scope_.get(), true);
            }
            auto after_send = GetCurrentUS();
            VLOG(1) << "send " << var_name << " use time "
                    << after_send - before_send;
          }
        };
        task_futures.emplace_back(
            send_threadpool_->enqueue(std::move(send_task)));
      } else {
        VLOG(4) << var_name << " queue empty";
      }
    }
    for (auto &task_f : task_futures) {
      task_f.wait();
      have_push_.fetch_add(1, std::memory_order_relaxed);
    }
    auto after_run_send_graph = GetCurrentUS();

    VLOG(3) << "run send graph use time "
            << after_run_send_graph - before_run_send_graph;
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
    auto push_num = have_push_.load();
    if (push_num >= var_nums_ ) {
      RecvAll();
      have_push_.store(0);
    } else {
      std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }
  }
  else {
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
  VLOG(1) << "parallel run recv graph";
  if (!running_) return;
  auto before_send = GetCurrentUS();
  std::vector<std::future<void>> task_futures;
  task_futures.reserve(recv_varname_to_ctx_.size());
  for (auto &iter : recv_varname_to_ctx_) {
    auto recv_task = [this, &iter] {
      auto &var_name = iter.first;
      VLOG(1) << "recv var " << var_name;
      auto recv_functor = distributed::ParameterRecv<float>();
      if (!FLAGS_communicator_fake_rpc && !is_geo_sgd_) {
        recv_functor(iter.second, *recv_scope_);
      }
      // for geo-sgd
      if(!FLAGS_communicator_fake_rpc && is_geo_sgd_) {
        recv_functor(iter.second, *pserver_scope_.get());
        RecvUpdateVars(var_name);
      }
    };
    task_futures.emplace_back(recv_threadpool_->enqueue(std::move(recv_task)));
  }
  for (auto &task : task_futures) {
    task.wait();
  }
  auto after_recv = GetCurrentUS();
  VLOG(1) << "run recv graph use time " << after_recv - before_send;
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
  if (is_geo_sgd_){
    VLOG(3) << "run into geo sgd communicator Send()";
    GeoSgdSend(var_name,scope);
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
                           Scope *recv_scope, int& trainers, int &geo_need_push_nums)
    : send_varname_to_ctx_(send_varname_to_ctx),
      recv_varname_to_ctx_(recv_varname_to_ctx),
      recv_scope_(recv_scope),
      trainer_nums_(trainers),geo_need_push_nums_(geo_need_push_nums) {
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

  if (send_varname_to_ctx.size() == 0) {
    VLOG(0) << "nothing need to be send, will not start send_thread";
  } else {
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
  is_geo_sgd_ = true;
  FLAGS_communicator_independent_recv_thread = false;
  var_nums_ = send_varname_to_ctx.size();

  VLOG(1) <<"var nums is: "<<var_nums_;
  VLOG(1) <<"geo sgd push nums: "<<geo_need_push_nums;

  delta_scope_.reset(new Scope()); //parameter local: recv(train) - old
  VLOG(1) << "Init delta scope, for send";
  GeoSgdParamInit(delta_scope_.get(), true);
  
  old_scope_.reset(new Scope()); //parameter local, storage the param after last recv
  VLOG(1) << "Init old scope";
  GeoSgdParamInit(old_scope_.get(), false);

  pserver_scope_.reset(new Scope()); //parameter on pserver, global scope
  VLOG(1) << "Init pserver(global) scope";
  GeoSgdParamInit(pserver_scope_.get(), false);
}

void Communicator::GeoSgdInit(const paddle::framework::ProgramDesc& program, Scope* param_scope,
                        std::map<std::string,std::map<std::string,std::vector<std::string>>> &vars_info, 
                        int &trainers, int &geo_need_push_nums){
  // param_scope is global scope
  // In geo-sgd model, we using pserver_scope to send&recv
  VLOG(0) << "ProcessGraph Geo_Sgd_Communicator";
  RpcCtxMap send_varname_to_ctx;
  RpcCtxMap recv_varname_to_ctx;
  for (auto &iter : vars_info) {
      std::string var_name = iter.first;
      std::string send_var_name = var_name.append(".delta");
      std::vector<std::string> vars_names = iter.second["var_names"];
      std::vector<std::string> vars_sections_str = iter.second["sections"];
      std::vector<int64_t> vars_sections_int = {};
      for (std::string str : vars_sections_str){
          int64_t str2i = std::stol(str.c_str());
          vars_sections_int.push_back(str2i);
      }
      std::vector<std::string> vars_epmap = iter.second["epmap"];
      int trainer_id = 0;
      send_varname_to_ctx[var_name] = operators::distributed::RpcContext(
          send_var_name,vars_names,vars_epmap,vars_sections_int,trainer_id);
      recv_varname_to_ctx[var_name] = operators::distributed::RpcContext(
          var_name,vars_names,vars_epmap,{},trainer_id);
      VLOG(1) << "find and init an send&recv param: "<< send_varname_to_ctx[var_name];
  }
  // init communicator here
  if (send_varname_to_ctx.size() == 0 && recv_varname_to_ctx.size() == 0) {
    LOG(WARNING) << "no var need to send and recv!!";
  }
  Communicator::Init(send_varname_to_ctx,recv_varname_to_ctx, param_scope, trainers, geo_need_push_nums);  
}

void Communicator::GeoSgdSend(const std::string& var_name, 
                              const framework::Scope& scope) {
  VLOG(1) << "geo sgd communicator get loop num "<< var_name;
  if(var_name == "param_init"){
    // when execute trainer startup program, recv init parameter from pserver
    // old_scope param will copy it for storage
    VLOG(1) <<"Parameter init from recv_scope";
    for(auto &iter:recv_varname_to_ctx_){
      auto var_name = iter.first;
      GeoSgdParamCopy(*recv_scope_,*old_scope_.get(),var_name, false);
      GeoSgdParamCopy(*recv_scope_,*pserver_scope_.get(),var_name, false);
      GeoSgdParamCopy(*recv_scope_,*delta_scope_.get(),VarToDeltaVar(var_name), true);
    }
    return;
  }
  else if (var_name == "batch_num" ) {
    need_push_.fetch_add(1, std::memory_order_relaxed);
    auto need_push = need_push_.load();
    if (need_push >= geo_need_push_nums_){
      for (auto &iter:recv_varname_to_ctx_) {
        std::string local_var_name = iter.first;
        auto &queue = send_varname_to_queue_.at(VarToDeltaVar(local_var_name));
        VLOG(1) << "send " << local_var_name << " queue size " << queue->Size();   
        SendUpdateVars(local_var_name);
        auto *delta_var = delta_scope_->FindVar(VarToDeltaVar(local_var_name));
        auto tmp_param_var = std::make_shared<Variable>();
        framework::CopyVariable(*delta_var, tmp_param_var.get());
        queue->Push(tmp_param_var);
      }
      need_push_.store(0);
    }
  }
}

void Communicator::GeoSgdParamInit(framework::Scope *scope,bool send){
  VLOG(3) << "Init scope parameter from send_varname_to_ctx_, Scope ptr: "<< scope;
  for(auto &iter:recv_varname_to_ctx_){
    auto var_name = iter.first;
    if (send) {
      auto send_var_name = VarToDeltaVar(var_name);
      scope->Var(send_var_name);
    } else {
      scope->Var(var_name);
    }
  }
}

void Communicator::GeoSgdParamCopy(const framework::Scope &scope_x,
                                   const framework::Scope &scope_y,
                                   const std::string var_name, bool send) {
  // copy var(send_varname_to_ctx_) from x to y
  VLOG(3) <<"Copy parameter from scope: "<< &scope_x 
          <<"To scope: "<< &scope_y 
          <<"Parameter name: "<< var_name; 
  auto *var_x = scope_x.FindVar(var_name);
  auto copy_var_name = send ? VarToDeltaVar(var_name) : var_name;
  auto *var_y = scope_y.FindVar(copy_var_name);
  framework::CopyVariable(*var_x,var_y);
}

void Communicator::SendUpdateVars(const std::string& var_name) {
  // calc var_delata = (var_recv - var_old)/trainer_nums
  // calc var_old += var_delta
  VLOG(1) << "Geo-Sgd Communicator Send update Vars: "<< var_name;
  // Todo: add check
  auto *var_x = recv_scope_->FindVar(var_name);
  auto *var_y = old_scope_.get()->FindVar(var_name);
  auto *var_z = delta_scope_.get()->FindVar(VarToDeltaVar(var_name));

  if (var_x->IsType<framework::LoDTensor>() && var_y->IsType<framework::LoDTensor>()){
    auto var_x_tensor = var_x->Get<framework::LoDTensor>();
    auto var_y_tensor = var_y->Get<framework::LoDTensor>();
    auto var_z_tensor = var_z->Get<framework::LoDTensor>();
    int element_number = var_x_tensor.numel();
    float* x_mutable_data = var_x_tensor.mutable_data<float>(var_x_tensor.place());
    float* y_mutable_data = var_y_tensor.mutable_data<float>(var_y_tensor.place());
    float* z_mutable_data = var_z_tensor.mutable_data<float>(var_z_tensor.place());
    VLOG(1) << "Geo-Sgd Send " << var_name<< " before update Vars recv_scope: "<< *x_mutable_data
            <<" ;old_scope: "<< *y_mutable_data
            <<" ;delta_scope(param local delta): "<< *z_mutable_data;
    for(int i = 0; i < element_number; i++){
      z_mutable_data[i] = (x_mutable_data[i] - y_mutable_data[i])/(float)(trainer_nums_);
      y_mutable_data[i] += z_mutable_data[i];
    }
    VLOG(1) << "Geo-Sgd Send " << var_name<< " after update Vars recv_scope: "<< *x_mutable_data
            <<" ;old_scope: "<< *y_mutable_data
            <<" ;delta_scope(param local delta): "<< *z_mutable_data;
  }
  // Todo: add Sparse param sub method 
}

void Communicator::RecvUpdateVars(const std::string& var_name) {
  // calc var_recv = var_pserver - var_old
  // calc var_old = var_pserver
  VLOG(1) << "Geo-Sgd Communicator Recv update Vars: "<< var_name;
  // Todo: add check
  auto *var_x = recv_scope_->FindVar(var_name);
  auto *var_y = old_scope_.get()->FindVar(var_name);
  auto *var_z = pserver_scope_.get()->FindVar(var_name);

  if (var_x->IsType<framework::LoDTensor>() && var_y->IsType<framework::LoDTensor>()){
    auto var_x_tensor = var_x->Get<framework::LoDTensor>();
    auto var_y_tensor = var_y->Get<framework::LoDTensor>();
    auto var_z_tensor = var_z->Get<framework::LoDTensor>();

    int element_number = var_x_tensor.numel();
    float* x_mutable_data = var_x_tensor.mutable_data<float>(var_x_tensor.place());
    float* y_mutable_data = var_y_tensor.mutable_data<float>(var_y_tensor.place());
    float* z_mutable_data = var_z_tensor.mutable_data<float>(var_z_tensor.place());
    VLOG(1) << "Geo-Sgd Recv " << var_name<< " before update Vars recv_scope: "<< *x_mutable_data
            <<" ;old_scope: "<< *y_mutable_data
            <<" ;delta_scope(param on pserver): "<< *z_mutable_data;
    for(int i = 0; i < element_number; i++){
      x_mutable_data[i] += (z_mutable_data[i] - y_mutable_data[i]);
      y_mutable_data[i] = z_mutable_data[i];
    }
    VLOG(1) << "Geo-Sgd Recv " << var_name<< " after update Vars recv_scope: "<< *x_mutable_data
            <<" ;old_scope: "<< *y_mutable_data
            <<" ;delta_scope(param on pserver): "<< *z_mutable_data;
  }
  // Todo: add Sparse param sub method 
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
