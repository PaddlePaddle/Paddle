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
#include <cfloat>

#include "paddle/fluid/distributed/ps/service/heter_server.h"
#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/platform/densetensor_printer.h"
#include "paddle/phi/core/platform/cpu_helper.h"
#include "paddle/phi/core/platform/device_context.h"

namespace paddle::framework {

void SetMicroId(paddle::framework::Scope* scope,
                phi::DeviceContext* dev_ctx,
                const phi::Place& place,
                int micro_id) {
  // create microbatch_id variable
  // and set micro id value
  auto* ptr = scope->Var("microbatch_id");
  InitializeVariable(ptr, proto::VarType::DENSE_TENSOR);
  framework::Variable* var = scope->FindVar("microbatch_id");
  PADDLE_ENFORCE_EQ(
      var->IsType<phi::DenseTensor>(),
      1,
      common::errors::InvalidArgument(
          "the type of microbatch_id  should be phi::DenseTensor"));
  auto* tensor = var->GetMutable<phi::DenseTensor>();
  std::vector<int> dims{1};
  tensor->Resize(common::make_ddim(dims));
  void* tensor_data = tensor->mutable_data(
      place, phi::TransToPhiDataType(framework::proto::VarType::FP32));
  if (phi::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    std::vector<char> temp;
    temp.resize(tensor->numel() * phi::SizeOf(tensor->dtype()));
    char* temp_ptr = temp.data();
    float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
    temp_ptr_float[0] = static_cast<float>(micro_id);
    auto stream = reinterpret_cast<const phi::GPUContext&>(*dev_ctx).stream();
    memory::Copy(
        place,
        tensor_data,
        phi::CPUPlace(),
        reinterpret_cast<void*>(temp_ptr),
        tensor->numel() * framework::SizeOfType(
                              framework::TransToProtoVarType(tensor->dtype())),
        stream);
#endif
  } else {
    float* temp_ptr = reinterpret_cast<float*>(tensor_data);
    temp_ptr[0] = static_cast<float>(micro_id);
  }
}

class TrainerDesc;

uint64_t HeterSectionWorker::batch_id_(0);

#ifdef PADDLE_WITH_FLPS
void HeterSectionWorker::Initialize(const TrainerDesc& desc) {
  trainer_desc_ = desc;
  fetch_config_ = desc.fetch_config();
  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);
  program_.reset(new ProgramDesc(
      desc.heter_section_param().section_config().program_desc()));
  thread_queue_.reset(
      new ::paddle::framework::BlockingQueue<std::pair<std::string, int>>());
  VLOG(4) << "addr of thread_queue_ is: " << thread_queue_.get();
  bool is_first_stage = (pipeline_stage_ == 0);
  bool is_last_stage = (pipeline_stage_ + 1 == num_pipeline_stages_);

  if (is_first_stage) {
    VLOG(0) << "entering first stage";
    for (auto& op_desc : program_->Block(0).AllOps()) {
      forward_ops_.push_back(std::move(OpRegistry::CreateOp(*op_desc)));
    }
    for (auto& op_desc : program_->Block(1).AllOps()) {
      auto op = std::move(OpRegistry::CreateOp(*op_desc));
      auto op_type = op->Type();
      if (listen_op_ == nullptr && op_type == "heter_listen_and_serv") {
        listen_op_ = std::move(op);
      } else {
        backward_ops_.push_back(std::move(op));
      }
    }
  } else if (is_last_stage) {
    VLOG(0) << "HeterSectionWorker::Initialize for the last stage";
    for (auto& op_desc : program_->Block(0).AllOps()) {
      auto op = std::move(OpRegistry::CreateOp(*op_desc));
      auto op_type = op->Type();
      if (listen_op_ == nullptr && op_type == "heter_listen_and_serv") {
        listen_op_ = std::move(op);
      } else {
        forward_ops_.push_back(std::move(op));
      }
    }
    VLOG(0) << "test111";
    for (auto& op_desc : program_->Block(1).AllOps()) {
      auto op = std::move(OpRegistry::CreateOp(*op_desc));
      backward_ops_.push_back(std::move(op));
    }
  }
}
#else
void HeterSectionWorker::Initialize(const TrainerDesc& desc) {
  trainer_desc_ = desc;
  fetch_config_ = desc.fetch_config();
  dev_ctx_ = phi::DeviceContextPool::Instance().Get(place_);
  program_.reset(new ProgramDesc(
      desc.heter_section_param().section_config().program_desc()));
  thread_queue_.reset(
      new ::paddle::framework::BlockingQueue<std::pair<std::string, int>>());
  bool is_first_stage = (pipeline_stage_ == 0);
  bool is_last_stage = (pipeline_stage_ + 1 == num_pipeline_stages_);

  if (is_first_stage) {  // NOLINT
    for (auto& op_desc : program_->Block(0).AllOps()) {
      auto op = OpRegistry::CreateOp(*op_desc);
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
        listen_op_ = OpRegistry::CreateOp(*op_desc);
      }
    }
    for (auto& op_desc : program_->Block(1).AllOps()) {
      auto op = OpRegistry::CreateOp(*op_desc);
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
        listen_op_ = OpRegistry::CreateOp(*op_desc);
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
#endif

void HeterSectionWorker::RunBackward(int micro_id) {
  for (size_t i = 0; i < backward_ops_.size(); i++) {
    auto& op = backward_ops_[i];
    VLOG(3) << "Backward: start to run op " << op->Type() << " for micro-batch "
            << micro_id;
    if (debug_) {
      timeline_.Start();
    }
    op->Run(*((*microbatch_scopes_)[micro_id]), place_);
    dev_ctx_->Wait();
    if (debug_) {
      timeline_.Pause();
      int offset = forward_ops_.size();
      op_total_time_[i + offset] += timeline_.ElapsedSec();
      total_time_ += timeline_.ElapsedSec();
    }
    VLOG(3) << "Backward: finish running op " << op->Type()
            << " for micro-batch " << micro_id;
  }
}

void HeterSectionWorker::MiniBatchBarrier() {
  // get micro id & deserialize data
  std::set<int> micro_ids;
  VLOG(4) << "entering MiniBatchBarrier";
  VLOG(4) << "micro_ids_.size(): " << micro_ids_.size();
  while (micro_ids.size() < micro_ids_.size()) {
    auto task = (*thread_queue_).Pop();
    VLOG(4) << "got one task from task que in cpu worker";
    auto message_name = task.first;
    auto micro_id = task.second;
    PADDLE_ENFORCE_EQ(message_name.find("backward") != std::string::npos,
                      true,
                      common::errors::InvalidArgument(
                          "cpu trainers only receive backward data"));
    PADDLE_ENFORCE_EQ(
        micro_ids.find(micro_id) == micro_ids.end(),
        true,
        common::errors::InvalidArgument("minibatch_scope_ can not be nullptr "
                                        "when create MicroBatch Scope"));
    micro_ids.insert(micro_id);
    // backward data has been deserialized to micro scope
    // now run backward computation
    RunBackward(micro_id);
    batch_num_++;
    BatchPostProcess();
    VLOG(0) << "one task in cpu worker overed!";
  }
  micro_ids_.clear();
}

void HeterSectionWorker::RunListen() {
  VLOG(4) << ">>> run listen_op";
  listen_op_->Run(*root_scope_, place_);
  VLOG(4) << "<<< run listen_op over";
}

void HeterSectionWorker::RunForward(int micro_id) {
#ifdef PADDLE_WITH_FLPS
  BindingDataFeedMemory(micro_id);
  if (debug_) {
    timeline_.Start();
  }
  int cur_micro_batch = device_reader_->Next();
  if (cur_micro_batch <= 0) {
    VLOG(0) << "no more data in device_reader_";
    epoch_finish_ = true;
    return;
  }
  if (debug_) {
    timeline_.Pause();
    read_time_ += timeline_.ElapsedSec();
    total_time_ += timeline_.ElapsedSec();
    total_ins_num_ += cur_micro_batch;
  }
  VLOG(3) << "read a batch in thread " << thread_id_ << " micro " << micro_id;
#else
  if (pipeline_stage_ == 0) {
    BindingDataFeedMemory(micro_id);
    if (debug_) {
      timeline_.Start();
    }
    int cur_micro_batch =
        device_reader_->Next();  // batch_size is just micro_batch_size
    if (cur_micro_batch <= 0) {
      epoch_finish_ = true;
      return;
    }
    if (debug_) {
      timeline_.Pause();
      read_time_ += timeline_.ElapsedSec();
      total_time_ += timeline_.ElapsedSec();
      total_ins_num_ += cur_micro_batch;
    }
    VLOG(3) << "read a batch in thread " << thread_id_ << " micro " << micro_id;
  }
#endif
  for (size_t i = 0; i < forward_ops_.size(); i++) {
    auto& op = forward_ops_[i];
    VLOG(3) << "Forward: start to run op " << op->Type() << " for micro-batch "
            << micro_id;
    if (debug_) {
      timeline_.Start();
    }
    op->Run(*((*microbatch_scopes_)[micro_id]), place_);
    dev_ctx_->Wait();
    if (debug_) {
      timeline_.Pause();
      op_total_time_[i] += timeline_.ElapsedSec();
      total_time_ += timeline_.ElapsedSec();
    }
    VLOG(3) << "Forward: finish running op " << op->Type()
            << " for micro-batch " << micro_id;
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
      common::errors::InvalidArgument(
          "minibatch_scope_ can not be nullptr when create MicroBatch Scopes"));
  if (microbatch_scopes_.get() == nullptr) {
    microbatch_scopes_.reset(new std::vector<paddle::framework::Scope*>{});
    (*microbatch_scopes_).resize(num_microbatches_);
    VLOG(3) << "Create microbatch scopes...";
    for (int j = 0; j < num_microbatches_; ++j) {
      (*microbatch_scopes_)[j] = &minibatch_scope_->NewScope();
    }
  }
  if (thread_id_ >= 0) {
    std::shared_ptr<framework::ProgramDesc> program;
    program.reset(new ProgramDesc(
        trainer_desc_.heter_section_param().section_config().program_desc()));
    for (int j = 0; j < num_microbatches_; ++j) {
      CopyParameters(j, *program, place_);
    }
  }
}

void HeterSectionWorker::CopyParameters(int microbatch_id,
                                        const ProgramDesc& program,
                                        const phi::Place& place) {
  auto& global_block = program.Block(0);
  auto var_list = global_block.AllVars();
  if (program.Size() > 1) {
    auto& heter_block = program.Block(1);
    auto heter_var_list = heter_block.AllVars();
    var_list.insert(
        var_list.end(), heter_var_list.begin(), heter_var_list.end());
  }
  if (program.Size() > 2) {
    auto& heter_block = program.Block(2);
    auto heter_var_list = heter_block.AllVars();
    var_list.insert(
        var_list.end(), heter_var_list.begin(), heter_var_list.end());
  }
  auto global_micro_id = thread_id_ * 10 + microbatch_id;
  SetMicroId(
      (*microbatch_scopes_)[microbatch_id], dev_ctx_, place, global_micro_id);
  for (auto& var : var_list) {
    if (var->Persistable() && microbatch_id == 0) {
      if (root_scope_->FindVar(var->Name()) != nullptr) continue;
      auto* ptr = root_scope_->Var(var->Name());
      InitializeVariable(ptr, var->GetType());
      VLOG(5) << "Create persistable var: " << var->Name()
              << ", which pointer is " << ptr;
    } else if (!var->Persistable()) {
      if ((*microbatch_scopes_)[microbatch_id]->FindVar(var->Name()) != nullptr)
        continue;
      auto* ptr = (*microbatch_scopes_)[microbatch_id]->Var(var->Name());
      VLOG(5) << "Create variable " << var->Name() << " for microbatch "
              << microbatch_id << ", which pointer is " << ptr;
      InitializeVariable(ptr, var->GetType());
    }
  }
}

void HeterSectionWorker::Run() {
  if (debug_) {
    size_t total_ops_size = forward_ops_.size() + backward_ops_.size();
    op_name_.reserve(total_ops_size);
    op_total_time_.resize(total_ops_size);
    platform::SetNumThreads(1);
    // forward op + backward op
    for (auto& op : forward_ops_) {
      op_name_.push_back(op->Type());
    }
    for (auto& op : backward_ops_) {
      op_name_.push_back(op->Type());
    }
    for (double& op_time : op_total_time_) {
      op_time = 0.0;
    }
  }
  bool is_first_stage = (pipeline_stage_ == 0);
  bool is_last_stage = (pipeline_stage_ + 1 == num_pipeline_stages_);
  if (is_first_stage) {  // for cpu trainer
    while (!epoch_finish_) {
      // forward
      for (int i = 0; i < num_microbatches_; i++) {
        VLOG(4) << "Run " << i << " microbatch";
        RunForward(i);
        if (epoch_finish_ == true) {
          break;
        }
        micro_ids_.push_back(i);
      }
      // backward
      if (!micro_ids_.empty()) {
        MiniBatchBarrier();
      }
      VLOG(0) << "one batch run over! micro_ids_size: " << micro_ids_.size();
    }
  } else {  // for heter worker
    VLOG(4) << "entering heter Run...";
    auto heter_server = paddle::distributed::HeterServer::GetInstance();
    while (true) {
      if (heter_server->IsStop()) {
        VLOG(0) << "heter_server is stopped!!";
        epoch_finish_ = true;
        break;
      }
      auto task = (*thread_queue_).Pop();
      VLOG(4) << "got one task from task que in heter worker";
      auto message_name = task.first;
      auto micro_id = task.second;
      if (is_last_stage) {
        PADDLE_ENFORCE_EQ(message_name.find("forward") != std::string::npos,
                          1,
                          common::errors::InvalidArgument(
                              "last stage only receive forward data"));
        RunForward(micro_id);
        RunBackward(micro_id);
        batch_num_++;
        BatchPostProcess();
        VLOG(0) << "one batch run over! micro_id: " << micro_id
                << " batch_num: " << batch_num_;
      } else {
        if (message_name.find("forward") != std::string::npos) {
          RunForward(micro_id);
        } else if (message_name.find("backward") != std::string::npos) {
          RunBackward(micro_id);
          batch_num_++;
          BatchPostProcess();
        }
      }
    }
  }
}

void HeterSectionWorker::BatchPostProcess() {
  PrintFetchVars();
  // dump param & field
  if (need_dump_field_) {
    DumpField(*((*microbatch_scopes_)[0]), dump_mode_, dump_interval_);
  }
  if (need_dump_param_ && thread_id_ == 0) {
    DumpParam(*((*microbatch_scopes_)[0]), batch_num_);
  }
  // print each op time
  if (debug_ && thread_id_ == 0) {
    size_t total_ops_size = forward_ops_.size() + backward_ops_.size();
    if (batch_num_ > 0 && batch_num_ % 100 == 0) {
      for (size_t i = 0; i < total_ops_size; ++i) {
        fprintf(stderr,
                "op_name:[%zu][%s], op_mean_time:[%fs]\n",
                i,
                op_name_[i].c_str(),
                op_total_time_[i] / batch_num_);
      }
      if (pipeline_stage_ == 0) {
        fprintf(stderr,
                "mean read time: %fs\n",
                read_time_ / batch_num_);  // NOLINT
        fprintf(stderr, "IO percent: %f\n", read_time_ / total_time_ * 100);
      }
      fprintf(stderr,
              "%6.2f instances/s\n",
              total_ins_num_ / total_time_);  // NOLINT
    }
  }
}

void HeterSectionWorker::TrainFiles() {
  VLOG(4) << "entering HeterSectionWorker::TrainFiles";
  if (thread_id_ >= 0) {
    total_ins_num_ = 0;
    batch_num_ = 0;
    platform::SetNumThreads(1);
    timeline_.Start();
    VLOG(3) << "begin section_worker TrainFiles";
    epoch_finish_ = false;
#ifdef PADDLE_WITH_FLPS
    if (device_reader_ == nullptr) {
      VLOG(4) << "device_reader_ is null!!";
    }
    device_reader_->Start();
#else
    if (pipeline_stage_ == 0) {
      device_reader_->Start();
    }
#endif
    VLOG(4) << "Run in TrainFiles:";
    while (!epoch_finish_) {
      Run();
      dev_ctx_->Wait();
    }
    timeline_.Pause();
    VLOG(3) << "worker " << thread_id_ << " train cost "
            << timeline_.ElapsedSec()
            << " seconds, ins_num: " << total_ins_num_;
  }
}

void HeterSectionWorker::PrintFetchVars() {
  // call count
  int batch_per_print = fetch_config_.print_period();
  int fetch_var_num = fetch_config_.fetch_var_names_size();
  if (fetch_var_num == 0) {
    return;
  }
  if (thread_id_ == 0 && batch_num_ % batch_per_print == 0) {
    time_t curtime;
    time(&curtime);
    char mbstr[80];  // NOLINT
    std::strftime(
        mbstr, sizeof(mbstr), "%Y-%m-%d %H:%M:%S", std::localtime(&curtime));
    std::stringstream ss;
    ss << "time: [" << mbstr << "], ";
    ss << "batch: [" << batch_num_ << "], ";
    for (int i = 0; i < fetch_var_num; ++i) {
      platform::PrintVar((*microbatch_scopes_)[0],
                         fetch_config_.fetch_var_names(i),
                         fetch_config_.fetch_var_str_format(i),
                         &ss);
      if (i < fetch_var_num - 1) {
        ss << ", ";
      }
    }
    std::cout << ss.str() << std::endl;
  }
}

void HeterSectionWorker::TrainFilesWithProfiler() {
  if (thread_id_ >= 0) {
    VLOG(3) << "begin section_worker TrainFilesWithProfiler";
    batch_num_ = 0;
    epoch_finish_ = false;
    total_ins_num_ = 0;
    op_name_.clear();
    op_total_time_.clear();
#ifdef PADDLE_WITH_FLPS
    device_reader_->Start();
#else
    if (pipeline_stage_ == 0) {
      device_reader_->Start();
    }
#endif
    while (!epoch_finish_) {
      Run();
      dev_ctx_->Wait();
      if (epoch_finish_) {
        // dump param for debug
        if (need_dump_field_ || need_dump_param_) {
          writer_.Flush();
        }
      }
    }
  }
}

}  // namespace paddle::framework
#endif
