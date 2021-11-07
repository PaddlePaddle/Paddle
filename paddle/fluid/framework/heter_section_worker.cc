/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#if defined(PADDLE_WITH_PSCORE)
#include <float.h>
#include "paddle/fluid/distributed/service/heter_server.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

void SetMicroId(paddle::framework::Scope* scope,
                platform::DeviceContext* dev_ctx, const platform::Place& place,
                int micro_id) {
  // create microbatch_id variable
  // and set micro id value
  auto* ptr = scope->Var("microbatch_id");
  InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
  framework::Variable* var = scope->FindVar("microbatch_id");
  PADDLE_ENFORCE_EQ(var->IsType<framework::LoDTensor>(), 1,
                    platform::errors::InvalidArgument(
                        "the type of microbatch_id  should be LoDTensor"));
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  std::vector<int> dims{1};
  tensor->Resize(framework::make_ddim(dims));
  void* tensor_data =
      tensor->mutable_data(place, framework::proto::VarType::FP32);
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    std::vector<char> temp;
    temp.resize(tensor->numel() * framework::SizeOfType(tensor->type()));
    char* temp_ptr = temp.data();
    float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
    temp_ptr_float[0] = micro_id;
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(*dev_ctx).stream();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place), tensor_data,
                 platform::CPUPlace(), reinterpret_cast<void*>(temp_ptr),
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
#endif
  } else {
    float* temp_ptr = reinterpret_cast<float*>(tensor_data);
    temp_ptr[0] = micro_id;
  }
}

class TrainerDesc;

uint64_t HeterSectionWorker::batch_id_(0);

void HeterSectionWorker::Initialize(const TrainerDesc& desc) {
  trainer_desc_ = desc;
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(new ProgramDesc(
      desc.heter_section_param().section_config().program_desc()));
  thread_queue_.reset(
      new ::paddle::framework::BlockingQueue<std::pair<std::string, int>>());
  bool is_first_stage = (pipeline_stage_ == 0);
  bool is_last_stage = (pipeline_stage_ + 1 == num_pipeline_stages_);

  if (is_first_stage) {
    for (auto& op_desc : program_->Block(0).AllOps()) {
      auto op = std::move(OpRegistry::CreateOp(*op_desc));
      auto op_type = op->Type();
      if (listen_op_ == nullptr && op_type == "heter_listen_and_serv") {
        listen_op_ = std::move(op);
      } else {
        forward_ops_.push_back(std::move(op));
      }
    }
    for (auto& op_desc : program_->Block(1).AllOps()) {
      backward_ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
  } else if (is_last_stage) {
    for (auto& op_desc : program_->Block(0).AllOps()) {
      if (listen_op_ == nullptr) {
        listen_op_ = std::move(OpRegistry::CreateOp(*op_desc));
      }
    }
    for (auto& op_desc : program_->Block(1).AllOps()) {
      auto op = std::move(OpRegistry::CreateOp(*op_desc));
      int op_role = op->Attr<int>(std::string("op_role"));
      bool is_forward_op = (op_role == static_cast<int>(OpRole::kForward)) ||
                           (op_role == (static_cast<int>(OpRole::kForward) |
                                        static_cast<int>(OpRole::kLoss))) ||
                           (op_role == static_cast<int>(OpRole::kLRSched));
      if (is_forward_op) {
        forward_ops_.push_back(std::move(op));
      } else {
        backward_ops_.push_back(std::move(op));
      }
    }
  } else {
    for (auto& op_desc : program_->Block(0).AllOps()) {
      if (listen_op_ == nullptr) {
        listen_op_ = std::move(OpRegistry::CreateOp(*op_desc));
      }
    }
    for (auto& op_desc : program_->Block(1).AllOps()) {
      forward_ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
    for (auto& op_desc : program_->Block(2).AllOps()) {
      backward_ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
  }
}

void HeterSectionWorker::RunBackward(int micro_id) {
  for (auto& op : backward_ops_) {
    VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
            << micro_id;
    op->Run(*((*microbatch_scopes_)[micro_id]), place_);
  }
}

void HeterSectionWorker::MiniBatchBarrier() {
  // get micro id & deserialize data
  std::set<int> micro_ids;
  while (micro_ids.size() < micro_ids_.size()) {
    auto task = (*thread_queue_).Pop();
    auto message_name = task.first;
    auto micro_id = task.second;
    PADDLE_ENFORCE_EQ(message_name.find("backward") != std::string::npos, true,
                      platform::errors::InvalidArgument(
                          "cpu trainers only receive backward data"));
    PADDLE_ENFORCE_EQ(
        micro_ids.find(micro_id) == micro_ids.end(), true,
        platform::errors::InvalidArgument("minibatch_scope_ can not be nullptr "
                                          "when create MicroBatch Scope"));
    micro_ids.insert(micro_id);
    // backward data has been deserialized to micro scope
    // now run backward computation
    RunBackward(micro_id);
  }
}

void HeterSectionWorker::RunListen() { listen_op_->Run(*root_scope_, place_); }

void HeterSectionWorker::RunForward(int micro_id) {
  bool is_first_stage = (pipeline_stage_ == 0);
  if (is_first_stage) {
    BindingDataFeedMemory(micro_id);
    int cur_micro_batch = device_reader_->Next();
    if (cur_micro_batch <= 0) {
      epoch_finish_ = true;
      return;
    }
    total_ins_num_ += cur_micro_batch;
  }
  for (auto& op : forward_ops_) {
    VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
            << micro_id;
    op->Run(*((*microbatch_scopes_)[micro_id]), place_);
  }
}

void HeterSectionWorker::BindingDataFeedMemory(int micro_id) {
  const std::vector<std::string>& input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar((*microbatch_scopes_)[micro_id]->FindVar(name),
                               name);
  }
}

void HeterSectionWorker::CreateMicrobatchScopes() {
  PADDLE_ENFORCE_NOT_NULL(
      minibatch_scope_,
      platform::errors::InvalidArgument(
          "minibatch_scope_ can not be nullptr when create MicroBatch Scopes"));
  microbatch_scopes_.reset(new std::vector<paddle::framework::Scope*>{});
  (*microbatch_scopes_).resize(num_microbatches_);
  VLOG(3) << "Create microbatch scopes...";
  std::shared_ptr<framework::ProgramDesc> program;
  program.reset(new ProgramDesc(
      trainer_desc_.heter_section_param().section_config().program_desc()));
  for (int j = 0; j < num_microbatches_; ++j) {
    (*microbatch_scopes_)[j] = &minibatch_scope_->NewScope();
    CopyParameters(j, *program, place_);
  }
}

void HeterSectionWorker::CopyParameters(int microbatch_id,
                                        const ProgramDesc& program,
                                        const platform::Place& place) {
  auto& global_block = program.Block(0);
  auto var_list = global_block.AllVars();
  if (program.Size() > 1) {
    auto& heter_block = program.Block(1);
    auto heter_var_list = heter_block.AllVars();
    var_list.insert(var_list.end(), heter_var_list.begin(),
                    heter_var_list.end());
  }
  if (program.Size() > 2) {
    auto& heter_block = program.Block(2);
    auto heter_var_list = heter_block.AllVars();
    var_list.insert(var_list.end(), heter_var_list.begin(),
                    heter_var_list.end());
  }
  auto global_micro_id = thread_id_ * 10 + microbatch_id;
  SetMicroId((*microbatch_scopes_)[microbatch_id], dev_ctx_, place,
             global_micro_id);
  for (auto& var : var_list) {
    if (var->Persistable() && microbatch_id == 0) {
      if (root_scope_->FindVar(var->Name()) != nullptr) continue;
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(5) << "Create persistable var: " << var->Name()
              << ", which pointer is " << ptr;
    } else if (!var->Persistable()) {
      auto* ptr = (*microbatch_scopes_)[microbatch_id]->Var(var->Name());
      VLOG(5) << "Create variable " << var->Name() << " for microbatch "
              << microbatch_id << ", which pointer is " << ptr;
      InitializeVariable(ptr, var->GetType());
    }
  }
}

void HeterSectionWorker::Run() {
  bool is_first_stage = (pipeline_stage_ == 0);
  bool is_last_stage = (pipeline_stage_ + 1 == num_pipeline_stages_);
  if (is_first_stage) {  // for cpu trainer
    while (!epoch_finish_) {
      // forward
      // std::vector<int> micro_ids;
      for (int i = 0; i < num_microbatches_; i++) {
        VLOG(5) << "Run " << i << " microbatch";
        RunForward(i);
        if (epoch_finish_ == true) {
          break;
        }
        micro_ids_.push_back(i);
      }
      // backward
      if (micro_ids_.size() > 0) {
        MiniBatchBarrier();
        PrintFetchVars();
        micro_ids_.clear();
      }
    }
  } else {  // for heter worker
    while (true) {
      auto heter_server = paddle::distributed::HeterServer::GetInstance();
      if (heter_server->IsStop()) {
        epoch_finish_ = true;
        break;
      }
      auto task = (*thread_queue_).Pop();
      auto message_name = task.first;
      auto micro_id = task.second;
      if (is_last_stage) {
        PADDLE_ENFORCE_EQ(message_name.find("forward") != std::string::npos, 1,
                          platform::errors::InvalidArgument(
                              "last stage only receive forward data"));
        RunForward(micro_id);
        RunBackward(micro_id);
      } else {
        if (message_name.find("forward") != std::string::npos) {
          RunForward(micro_id);
        } else if (message_name.find("backward") != std::string::npos) {
          RunBackward(micro_id);
        }
      }
    }
  }
}

void HeterSectionWorker::TrainFiles() {
  VLOG(3) << "begin section_worker TrainFiles";
  epoch_finish_ = false;
  bool is_first_stage = (pipeline_stage_ == 0);
  if (is_first_stage) {
    device_reader_->Start();
  }
  while (!epoch_finish_) {
    Run();
    dev_ctx_->Wait();
  }
}

void HeterSectionWorker::PrintFetchVars() {
  // call count
  batch_num_ += micro_ids_.size();
  int batch_per_print = fetch_config_.print_period();
  int fetch_var_num = fetch_config_.fetch_var_names_size();
  if (fetch_var_num == 0) {
    return;
  }
  if (thread_id_ == 0 && batch_num_ % batch_per_print == 0) {
    time_t curtime;
    time(&curtime);
    char mbstr[80];
    std::strftime(mbstr, sizeof(mbstr), "%Y-%m-%d %H:%M:%S",
                  std::localtime(&curtime));
    std::stringstream ss;
    ss << "time: [" << mbstr << "], ";
    ss << "batch: [" << batch_num_ << "], ";
    for (int i = 0; i < fetch_var_num; ++i) {
      platform::PrintVar(minibatch_scope_, fetch_config_.fetch_var_names(i),
                         fetch_config_.fetch_var_str_format(i), &ss);
      if (i < fetch_var_num - 1) {
        ss << ", ";
      }
    }
    std::cout << ss.str() << std::endl;
  }
}

/*
void HogwildWorker::TrainFilesWithProfiler() {
  platform::SetNumThreads(1);
  device_reader_->Start();
  std::vector<double> op_total_time;
  std::vector<std::string> op_name;
  for (auto &op : ops_) {
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
  uint64_t total_inst = 0;
  while ((cur_batch = device_reader_->Next()) > 0) {
    VLOG(3) << "read a batch in thread " << thread_id_;
    timeline.Pause();
    read_time += timeline.ElapsedSec();
    total_time += timeline.ElapsedSec();
    for (size_t i = 0; i < ops_.size(); ++i) {
      bool need_skip = false;
      for (auto t = 0u; t < skip_ops_.size(); ++t) {
        if (ops_[i]->Type().find(skip_ops_[t]) != std::string::npos) {
          need_skip = true;
          break;
        }
      }
      timeline.Start();
      VLOG(3) << "Going to run op " << op_name[i];
      if (!need_skip) {
        ops_[i]->Run(*thread_scope_, place_);
#ifdef PADDLE_WITH_HETERPS
        dev_ctx_->Wait();
#endif
      }
      VLOG(3) << "Op " << op_name[i] << " Finished";
      timeline.Pause();
      op_total_time[i] += timeline.ElapsedSec();
      total_time += timeline.ElapsedSec();
    }

    if (need_dump_field_) {
      DumpField(*thread_scope_, dump_mode_, dump_interval_);
    }
    if (need_dump_param_ && thread_id_ == 0) {
      DumpParam(*thread_scope_, batch_cnt);
    }

    total_inst += cur_batch;
    ++batch_cnt;
    PrintFetchVars();
#ifdef PADDLE_WITH_HETERPS
    dev_ctx_->Wait();
    VLOG(1) << "GpuPs worker " << thread_id_ << " train cost " << total_time
            << " seconds, ins_num: " << total_inst;
    for (size_t i = 0; i < op_name.size(); ++i) {
      VLOG(1) << "card:" << thread_id_ << ", op: " << op_name[i]
              << ", mean time: " << op_total_time[i] / total_inst
              << "s, totol time:" << op_total_time[i] << "sec";
    }
#else
    if (thread_id_ == 0) {
      if (batch_cnt > 0 && batch_cnt % 100 == 0) {
        for (size_t i = 0; i < ops_.size(); ++i) {
          fprintf(stderr, "op_name:[%zu][%s], op_mean_time:[%fs]\n", i,
                  op_name[i].c_str(), op_total_time[i] / batch_cnt);
        }
        fprintf(stderr, "mean read time: %fs\n", read_time / batch_cnt);
        fprintf(stderr, "IO percent: %f\n", read_time / total_time * 100);
        fprintf(stderr, "%6.2f instances/s\n", total_inst / total_time);
      }
    }
#endif
    thread_scope_->DropKids();
    timeline.Start();
  }

  if (need_dump_field_ || need_dump_param_) {
    writer_.Flush();
  }

#if defined PADDLE_WITH_PSCORE
  if (thread_barrier_) {
    paddle::distributed::Communicator::GetInstance()->BarrierTriggerDecrement();
  }
#endif
}

*/

}  // namespace framework
}  // namespace paddle
#endif
