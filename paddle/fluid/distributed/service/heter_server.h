/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#pragma once
#include <atomic>
#include <ctime>
#include <map>
#include <memory>
#include <mutex>
#include <random>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "brpc/channel.h"
#include "brpc/controller.h"
#include "brpc/server.h"
#include "paddle/fluid/distributed/service/brpc_utils.h"
#include "paddle/fluid/distributed/service/sendrecv.pb.h"
#include "paddle/fluid/framework/blocking_queue.h"
#include "paddle/fluid/framework/executor.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/macros.h"  // for DISABLE_COPY_AND_ASSIGN
#include "paddle/fluid/platform/profiler.h"

namespace google {
namespace protobuf {
class Closure;
class RpcController;
}  // namespace protobuf
}  // namespace google
namespace paddle {
namespace framework {
class Executor;
class ProgramDesc;
}  // namespace framework
namespace platform {
class DeviceContext;
}  // namespace platform
}  // namespace paddle

DECLARE_double(eager_delete_tensor_gb);
namespace paddle {
namespace distributed {

static void split(const std::string& str, char sep,
                  std::vector<std::string>* pieces) {
  pieces->clear();
  if (str.empty()) {
    return;
  }
  size_t pos = 0;
  size_t next = str.find(sep, pos);
  while (next != std::string::npos) {
    pieces->push_back(str.substr(pos, next - pos));
    pos = next + 1;
    next = str.find(sep, pos);
  }
  if (!str.substr(pos).empty()) {
    pieces->push_back(str.substr(pos));
  }
}

using MultiVarMsg = ::paddle::distributed::MultiVariableMessage;
using VarMsg = ::paddle::distributed::VariableMessage;

class HeterService;

typedef int32_t (HeterService::*serviceHandlerFunc)(
    const PsRequestMessage& request, PsResponseMessage& response,
    brpc::Controller* cntl);

typedef std::function<void(void*)> HeterRpcCallbackFunc;
typedef std::function<int(const MultiVarMsg*, MultiVarMsg*, brpc::Controller*)>
    HeterServiceHandler;

class HeterService : public ::paddle::distributed::PsService {
 public:
  HeterService() {
    _service_handler_map[PS_STOP_SERVER] = &HeterService::stop_heter_worker;
    _service_handler_map[PS_START_PROFILER] = &HeterService::start_profiler;
    _service_handler_map[PS_STOP_PROFILER] = &HeterService::stop_profiler;
  }

  virtual ~HeterService() {}

  virtual void service(::google::protobuf::RpcController* controller,
                       const PsRequestMessage* request,
                       PsResponseMessage* response,
                       ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    std::string log_label("ReceiveCmd-");

    response->set_err_code(0);
    response->set_err_msg("");
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    auto itr = _service_handler_map.find(request->cmd_id());
    if (itr == _service_handler_map.end()) {
      std::string err_msg(
          "undefined cmd_id, should match PsCmdID in ps.proto, cmd_id:");
      err_msg.append(std::to_string(request->cmd_id()));
      return;
    }
    serviceHandlerFunc handler_func = itr->second;
    int service_ret = (this->*handler_func)(*request, *response, cntl);
    if (service_ret != 0) {
      response->set_err_code(service_ret);
      response->set_err_msg("server internal error");
    }
  }

  void SendAndRecvVariable(::google::protobuf::RpcController* controller,
                           const MultiVarMsg* request, MultiVarMsg* response,
                           ::google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    std::string message_name = request->message_name();
    auto itr = handler_map_.find(message_name);
    brpc::Controller* cntl = static_cast<brpc::Controller*>(controller);
    PADDLE_ENFORCE_NE(
        itr, handler_map_.end(),
        platform::errors::InvalidArgument(
            "HeterService::SendAndRecvVariable Get illegal message_name: %s "
            "which is not in HeterService::handler_map_",
            message_name));
    itr->second(request, response, cntl);
  }

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func) {
    handler_map_[message_name] = func;
  }

  void SetEndpoint(const std::string& end_point) { endpoint_ = end_point; }
  void SetFanin(const int& fan_in) { fan_in_ = fan_in; }
  bool IsExit() { return is_exit_; }

 private:
  int32_t stop_profiler(const PsRequestMessage& request,
                        PsResponseMessage& response, brpc::Controller* cntl);

  int32_t start_profiler(const PsRequestMessage& request,
                         PsResponseMessage& response, brpc::Controller* cntl);

  int32_t stop_heter_worker(const PsRequestMessage& request,
                            PsResponseMessage& response,
                            brpc::Controller* cntl);

 private:
  std::string endpoint_;
  std::unordered_map<std::string, HeterServiceHandler> handler_map_;
  std::unordered_map<int32_t, serviceHandlerFunc> _service_handler_map;
  std::unordered_set<int> stop_cpu_worker_set_;
  int fan_in_;
  bool is_exit_ = false;
};

class HeterServer {
 public:
  virtual ~HeterServer() {}

  void Stop() {
    VLOG(3) << "HeterServer Stop()";
    std::unique_lock<std::mutex> lock(mutex_);
    stoped_ = true;
    cv_.notify_all();
    server_.Stop(1000);
    server_.Join();
  }

  bool IsExit() { return service_.IsExit(); }

  HeterServer() {}

  void RegisterServiceHandler(std::string message_name,
                              HeterServiceHandler func);

  void StartHeterService();

  void SetEndPoint(std::string& endpoint);
  void SetFanin(int& fan_in);

  // HeterWrapper singleton
  static std::shared_ptr<HeterServer> GetInstance() {
    if (NULL == s_instance_) {
      s_instance_.reset(new HeterServer());
    }
    return s_instance_;
  }

  void WaitServerReady();

 private:
  static std::shared_ptr<HeterServer> s_instance_;
  mutable std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable condition_ready_;
  bool stoped_ = false;
  std::string endpoint_;

 protected:
  brpc::Server server_;
  HeterService service_;
  DISABLE_COPY_AND_ASSIGN(HeterServer);
  std::mutex mutex_ready_;

  int ready_;
};

class HeterRequestHandler {
 public:
  HeterRequestHandler()
      : dev_ctx_(nullptr),
        executor_(nullptr),
        scope_(nullptr),
        program_(nullptr) {}

  virtual ~HeterRequestHandler() {}

  void SetScope(const framework::Scope* scope) { scope_ = scope; }
  void SetDevCtx(const platform::DeviceContext* dev_ctx) { dev_ctx_ = dev_ctx; }
  void SetProgram(framework::ProgramDesc* program) { program_ = program; }
  void SetExecutor(std::vector<framework::Executor>* executor) { executor_ = executor; }
  void SetMicroNum(int num_microbatch) { num_microbatch_ = num_microbatch; }
  void SetMiniNum(int num_minibatch) { num_minibatch_ = num_minibatch; }
  void SetTrainers(int trainers) { trainers_ = trainers; }
  void SetTrainerId(int trainer_id) { trainer_id_ = trainer_id; }
  virtual void Start() {}
  virtual void Process(int minibatch_idx) {}
  virtual void batch_finished(int minibatch_idx) {}

  void SetGradToPreparedCtx(
      std::unordered_map<
          std::string, std::shared_ptr<framework::ExecutorPrepareContext>>* g) {
    message_to_prepared_ctx_ = g;
  }

  virtual int Handle(const MultiVarMsg* request, MultiVarMsg* response,
                     brpc::Controller* cntl) = 0;

 protected:
  const platform::DeviceContext* dev_ctx_;
  std::vector<framework::Executor>* executor_;
  const framework::Scope* scope_;
  framework::ProgramDesc* program_;
  int num_microbatch_;
  int num_minibatch_;
  int trainers_;
  int trainer_id_;

  std::unordered_map<std::string,
                     std::shared_ptr<framework::ExecutorPrepareContext>>*
      message_to_prepared_ctx_;
};

class RequestSendAndRecvHandler final : public HeterRequestHandler {
 public:
  RequestSendAndRecvHandler() {
    this->num_microbatch_ = 0;
    this->num_minibatch_ = 0;
  }

  virtual ~RequestSendAndRecvHandler() { 
    for (int i = 0; i < this->num_minibatch_; i++) {
      (process_thread_[i].get())->join(); 
    }
  }

  void Start() {
    real_microbatch_.resize(num_minibatch_);
    for (int i = trainer_id_; i < num_microbatch_ * num_minibatch_; i += trainers_) {     
       int mini_idx = i / num_microbatch_;
       //int micro_idx = i % num_microbatch_;
       real_microbatch_[mini_idx]++;
    }
    done_.resize(num_minibatch_, 0);
    batch_finished_.resize(num_minibatch_, false);
    task_queue_.resize(this->num_minibatch_);
    for (int i = 0; i < this->num_minibatch_; i++) {
      task_queue_[i].reset(
          new ::paddle::framework::BlockingQueue<std::pair<std::string, int>>());
    } 
    bool has_backward = false;
    bool has_forward = false;
    for (auto& mpair : *message_to_prepared_ctx_) {
      if (mpair.first.find("forward") != mpair.first.npos) {
        has_forward = true;
      }
      if (mpair.first.find("backward") != mpair.first.npos) {
        has_backward = true;
      }
    }
    if (!has_forward) is_first_stage_ = true;
    if (!has_backward) is_last_stage_ = true;

    process_thread_.resize(this->num_minibatch_);
    for (int i = 0; i < this->num_minibatch_; i++) {
      process_thread_[i].reset(
          new std::thread(std::bind(&RequestSendAndRecvHandler::Process, this, i)));
    }
  }

  void batch_finished(int minibatch_idx) {
    std::unique_lock<std::mutex> lk(this->batch_finished_mutex);
    this->batch_finished_cond_var.wait(
        lk, [&]() { return this->batch_finished_[minibatch_idx] == true; });
  }

  void Process(int minibatch_idx) {
    int target_val = 2;
    if (is_first_stage_ || is_last_stage_) target_val = 1;
    while (true) {
      if (task_queue_[minibatch_idx]->Size() > 0) {
        if (batch_finished_[minibatch_idx] == true) {
          std::unique_lock<std::mutex> lk(this->batch_finished_mutex);
          batch_finished_[minibatch_idx] = false;
        }
        auto task = task_queue_[minibatch_idx]->Pop();
        auto message_name = task.first;
        auto micro_id = task.second;
        int minibatch_index = micro_id / num_microbatch_;
        int microbatch_index = micro_id % num_microbatch_;
        auto& mini_scope = scope_->KidScope(minibatch_index);
        auto& micro_scope = (&mini_scope)->KidScope(microbatch_index);
      
        PADDLE_ENFORCE_EQ(minibatch_idx, minibatch_index,
                        platform::errors::InvalidArgument(
                            "minibatch_idx should match"));

        micro_cnt_[minibatch_index][microbatch_index]++;
        if (micro_cnt_[minibatch_index][microbatch_index] >= target_val) {
          {
            std::unique_lock<std::mutex> lk(this->batch_finished_mutex);
            done_[minibatch_index]++;
            if (is_first_stage_) {
              if (done_[minibatch_index] == real_microbatch_[minibatch_index]) {
                micro_cnt_[minibatch_index].clear();
                done_[minibatch_index] = 0;
                batch_finished_[minibatch_index] = true;
                this->batch_finished_cond_var.notify_all();
              }
            }
          }
        }
        (*executor_)[minibatch_idx].RunPreparedContext(
            (*message_to_prepared_ctx_)[message_name].get(), &micro_scope,
            false);
      }
    }
  }

  int Handle(const MultiVarMsg* request, MultiVarMsg* response,
             brpc::Controller* cntl) override {
    platform::RecordEvent record_event("RequestSendAndRecvHandler->Handle");
    FLAGS_eager_delete_tensor_gb = -1;
    // get microID from request
    // deserialize variable to micro scope
    // Push to heter worker's task_queue
    //auto& local_scope = scope_->NewScope();
    std::unique_ptr<paddle::framework::Scope> local_scope_ptr(new paddle::framework::Scope());
    auto& local_scope = *(local_scope_ptr.get());
    auto message_name = request->message_name();
      auto& request_io_buffer = cntl->request_attachment();
      distributed::DeserializeFromMultiVarMsgAndIOBuf(
          *request, &request_io_buffer, *dev_ctx_, &local_scope);

      auto* var = local_scope.FindVar("microbatch_id");
      PADDLE_ENFORCE_NE(var, nullptr,
                        platform::errors::InvalidArgument(
                            "Not find variable microbatch_id in scope."));
      auto* tensor = var->GetMutable<framework::LoDTensor>();
      const auto place = dev_ctx_->GetPlace();
      int micro_id = -1;
      if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
        char* temp_ptr =
            new char[tensor->numel() * framework::SizeOfType(tensor->type())];
        auto stream =
            reinterpret_cast<const platform::CUDADeviceContext&>(*dev_ctx_)
                .stream();
        memory::Copy(platform::CPUPlace(), temp_ptr,
                     BOOST_GET_CONST(platform::CUDAPlace, tensor->place()),
                     tensor->data<void>(),
                     tensor->numel() * framework::SizeOfType(tensor->type()),
                     stream);
        float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
        micro_id = static_cast<int>(temp_ptr_float[0]);
        delete[] temp_ptr;
#endif
      } else {
        auto data = reinterpret_cast<const float*>(tensor->data<void>());
        micro_id = static_cast<int>(data[0]);
      }

      // get minibatch id & local micro id from global id
      PADDLE_ENFORCE_LT(micro_id, num_microbatch_ * num_minibatch_,
                        platform::errors::InvalidArgument(
                            "micro_id should less than num_microbatch_ * num_minibatch_."));

      int minibatch_index = micro_id / num_microbatch_;
      int microbatch_index = micro_id % num_microbatch_;


      if (message_name == "barrier_batch_finish") {
        PADDLE_ENFORCE_EQ(is_first_stage_, true,
                          platform::errors::InvalidArgument(
                             "only trainer in first stage will call barrier_batch_finish")); 
        batch_finished(minibatch_index);
      } else {
      auto& mini_scope = scope_->KidScope(minibatch_index);
      auto& micro_scope = (&mini_scope)->KidScope(microbatch_index);
      distributed::DeserializeFromMultiVarMsgAndIOBuf(
          *request, &request_io_buffer, *dev_ctx_, &micro_scope);
      // blocking queue handles multi thread
      task_queue_[minibatch_index]->Push(std::make_pair(message_name, micro_id));
    }
    auto response_var_nums = request->recv_var_names_size();
    std::vector<std::string> response_var_names(response_var_nums),
        empty_var_names{};
    for (int var_idx = 0; var_idx < response_var_nums; ++var_idx) {
      response_var_names[var_idx] = request->recv_var_names(var_idx);
    }
    auto& response_io_buffer = cntl->response_attachment();
    distributed::SerializeToMultiVarMsgAndIOBuf(
        message_name, response_var_names, empty_var_names, *dev_ctx_,
        &local_scope, response, &response_io_buffer);
    return 0;
  }

 private:
  std::unordered_map<int, std::unordered_map<int,int>> micro_cnt_;
  std::vector<int> done_;
  std::vector<int> real_microbatch_;
  std::mutex batch_finished_mutex;
  std::condition_variable batch_finished_cond_var;



  bool is_first_stage_ = false;
  bool is_last_stage_ = false;
  std::vector<bool> batch_finished_;

  std::vector<std::shared_ptr<std::thread>> process_thread_;
  std::vector<std::shared_ptr<
      ::paddle::framework::BlockingQueue<std::pair<std::string, int>>> >
      task_queue_;
};

}  // end namespace distributed
}  // end namespace paddle
