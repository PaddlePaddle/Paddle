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

#include <float.h>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/platform/device_context.h"


namespace paddle {
namespace framework {

class TrainerDesc;

uint64_t HeterSectionWorker::batch_id_(0);

void HeterSectionWorker::Initialize(const TrainerDesc &desc) {
  trainer_desc_ = desc;
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(
      new ProgramDesc(desc.heter_section_param().section_config().program_desc()));
  thread_queue_.reset(
      new ::paddle::framework::BlockingQueue<std::pair<std::string, int>>());
  bool is_first_stage = (pipeline_stage_ == 0);
  bool is_last_stage = (pipeline_stage_ + 1 == num_pipeline_stages_);

  if (is_first_stage) {
    for (auto &op_desc : program_->Block(0).AllOps()) {
      auto op = std::move(OpRegistry::CreateOp(*op_desc));
      auto op_type = op->Type();
      if (listen_op_ == nullptr && op_type == "heter_listen_and_serv") {
         listen_op_ = std::move(op);
      } else {
          forward_ops_.push_back(std::move(op));
      }
    }
    for (auto &op_desc : program_->Block(1).AllOps()) {
        backward_ops_.push_back(OpRegistry::CreateOp(*op_desc));
    } 
  } else if (is_last_stage) {
    for (auto &op_desc : program_->Block(0).AllOps()) {
      if (listen_op_ == nullptr) {
          listen_op_ = std::move(OpRegistry::CreateOp(*op_desc));
      }
    }
    for (auto &op_desc : program_->Block(1).AllOps()) {
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
    for (auto &op_desc : program_->Block(0).AllOps()) {
      if (listen_op_ == nullptr) {
          listen_op_ = std::move(OpRegistry::CreateOp(*op_desc));
      }
    }
    for (auto &op_desc : program_->Block(1).AllOps()) {
        forward_ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
    for (auto &op_desc : program_->Block(2).AllOps()) {
        backward_ops_.push_back(OpRegistry::CreateOp(*op_desc));
    }
  }
}

void HeterSectionWorker::RunBackward(int micro_id) {
  
  //bool is_first_stage = (pipeline_stage_ == 0);
  //if (is_first_stage) {
  //  BindingDataFeedMemory(micro_id);
  //  int cur_micro_batch = device_reader_->Next();
  //  if (cur_micro_batch <= 0) {
  //    epoch_finish_ = true;
  //    return;
  //  }
  //  total_ins_num_ += cur_micro_batch;
  //}

  for (auto &op : backward_ops_) {
    //int op_role = op->Attr<int>(std::string("op_role"));
    //auto op_type = op->Type();

    //if (op_type == "heter_listen_and_serv") continue;
    //if (op_type == "send_and_recv") {
    //  auto op_mode = op->Attr<std::string>(std::string("mode"));
    //  if (op_mode == "barrier") continue;
    //}
    //if (op_type == "trainer_barrier") continue;
    //bool run_first_mbatch = (op_role == static_cast<int>(OpRole::kForward)) ||
    //                        (op_role == (static_cast<int>(OpRole::kForward) |
    //                                     static_cast<int>(OpRole::kLoss))) ||
    //                        (op_role == static_cast<int>(OpRole::kRPC)) ||
    //                        (op_role == static_cast<int>(OpRole::kLRSched));

    //bool run_others = (op_role == static_cast<int>(OpRole::kForward)) ||
    //                  (op_role == (static_cast<int>(OpRole::kForward) |
    //                               static_cast<int>(OpRole::kLoss))) ||
    //                  (op_role == static_cast<int>(OpRole::kRPC));

    //if ((micro_id == 0 && run_first_mbatch) || (micro_id != 0 && run_others)) {
      VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*((*microbatch_scopes_)[micro_id]), place_);
    //}
  }
}

void HeterSectionWorker::MiniBatchBarrier(const std::vector<int>& barrier_ids) {
    // get micro id & deserialize data
    std::set<int> micro_ids;
    while(micro_ids.size() < barrier_ids.size()) {
      auto task = (*thread_queue_).Pop();
      auto message_name = task.first;
      auto micro_id = task.second;
	  PADDLE_ENFORCE_EQ(message_name.find("backward") != std::string::npos, true,
                        platform::errors::InvalidArgument(
				            "cpu trainers only receive backward data"));
	  PADDLE_ENFORCE_EQ(micro_ids.find(micro_id) == micro_ids.end(), true,
                        platform::errors::InvalidArgument(
				            "minibatch_scope_ can not be nullptr when create MicroBatch Scope"));
      micro_ids.insert(micro_id);
      // backward data has been deserialized to micro scope
      // now run backward computation 
      RunBackward(micro_id);
    }
}

//void HeterSectionWorker::TrainerBarrier() {
//  for (auto &op : ops_) {
//    auto op_type = op->Type();
//    if (op_type != "trainer_barrier") continue;
//    op->Run(*root_scope_, place_);
//  }
//}

void HeterSectionWorker::RunListen() {
  //bool is_first_stage = (pipeline_stage_ == 0);
  listen_op_->Run(*root_scope_, place_); 
  //for (auto &op : ops_) {
  //  auto op_type = op->Type();
  //  if (op_type == "heter_listen_and_serv") {
  //    if (is_first_stage) {
  //      if (thread_id_ == 0)
  //        op->Run(*root_scope_, place_);
  //    } else { // for heter worker
  //      op->Run(*root_scope_, place_);
  //    }
  //  }
  //}
}

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

  for (auto &op : forward_ops_) {
    //int op_role = op->Attr<int>(std::string("op_role"));
    //auto op_type = op->Type();

    //if (op_type == "heter_listen_and_serv") continue;
    //if (op_type == "send_and_recv") {
    //  auto op_mode = op->Attr<std::string>(std::string("mode"));
    //  if (op_mode == "barrier") continue;
    //}
    //if (op_type == "trainer_barrier") continue;
    //bool run_first_mbatch = (op_role == static_cast<int>(OpRole::kForward)) ||
    //                        (op_role == (static_cast<int>(OpRole::kForward) |
    //                                     static_cast<int>(OpRole::kLoss))) ||
    //                        (op_role == static_cast<int>(OpRole::kRPC)) ||
    //                        (op_role == static_cast<int>(OpRole::kLRSched));

    //bool run_others = (op_role == static_cast<int>(OpRole::kForward)) ||
    //                  (op_role == (static_cast<int>(OpRole::kForward) |
    //                               static_cast<int>(OpRole::kLoss))) ||
    //                  (op_role == static_cast<int>(OpRole::kRPC));

    //if ((micro_id == 0 && run_first_mbatch) || (micro_id != 0 && run_others)) {
      VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*((*microbatch_scopes_)[micro_id]), place_);
    //}
  }
}

void HeterSectionWorker::BindingDataFeedMemory(int micro_id) {
  const std::vector<std::string> &input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar((*microbatch_scopes_)[micro_id]->FindVar(name),
                               name);
  }
}


void HeterSectionWorker::CreateMicrobatchScopes() {
  PADDLE_ENFORCE_NOT_NULL(minibatch_scope_, platform::errors::InvalidArgument(
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

  // create microbatch_id variable
  // and set micro id value
  // TODO(zhangminxu): performance optimization 
  auto* ptr = (*microbatch_scopes_)[microbatch_id]->Var("microbatch_id");
  InitializeVariable(ptr, proto::VarType::LOD_TENSOR);
  framework::Variable* var =
      (*microbatch_scopes_)[microbatch_id]->FindVar("microbatch_id");
  PADDLE_ENFORCE_EQ(var->IsType<framework::LoDTensor>(), 1,
                    platform::errors::InvalidArgument(
                        "the type of microbatch_id  should be LoDTensor"));
  auto* tensor = var->GetMutable<framework::LoDTensor>();
  std::vector<int> dims{1};
  tensor->Resize(framework::make_ddim(dims));
  void* tensor_data =
      tensor->mutable_data(place, framework::proto::VarType::FP32);
  auto global_micro_id = thread_id_ * 10 + microbatch_id;    
  //auto global_micro_id = thread_id_ * num_microbatches_ + microbatch_id; 
  if (platform::is_gpu_place(place)) {
#ifdef PADDLE_WITH_CUDA
    char* temp_ptr =
        new char[tensor->numel() * framework::SizeOfType(tensor->type())];
    float* temp_ptr_float = reinterpret_cast<float*>(temp_ptr);
    temp_ptr_float[0] = global_micro_id;
    auto stream =
        reinterpret_cast<const platform::CUDADeviceContext&>(*dev_ctx_)
            .stream();
    memory::Copy(BOOST_GET_CONST(platform::CUDAPlace, place), tensor_data,
                 platform::CPUPlace(), reinterpret_cast<void*>(temp_ptr),
                 tensor->numel() * framework::SizeOfType(tensor->type()),
                 stream);
    delete[] temp_ptr;
#endif
  } else {
    float* temp_ptr = reinterpret_cast<float*>(tensor_data);
    temp_ptr[0] = global_micro_id;
  }

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
  if (is_first_stage) { // for cpu trainer
    // forward
    std::vector<int> micro_ids;
    for (int i = 0; i < num_microbatches_; i++) {
        VLOG(5) << "Run " << i << " stage" << std::endl;
        RunForward(i);
        if (epoch_finish_ == true) { break;}
        micro_ids.push_back(i);
    }
    // backward
    MiniBatchBarrier(micro_ids);
  } else { // for heter worker
      int cnt = 0;
      int target_ = -1;
      if (is_last_stage) {
        target_ = num_microbatches_;
      } else {
        target_ = 2 * num_microbatches_;
      } 
      while(cnt < target_) {
        auto task = (*thread_queue_).Pop();
        auto message_name = task.first;
        auto micro_id = task.second;

        if (message_name.find("forward") != std::string::npos) {
            RunForward(micro_id);
        } else if (message_name.find("backward") != std::string::npos) {
            RunBackward(micro_id);
        }
        cnt++;
      }

  }
}

void HeterSectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  if (max_memory_size >= 0) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
    }
#elif defined(PADDLE_WITH_ASCEND_CL)
    if (IsFastEagerDeletionModeEnabled()) {
      VLOG(4) << "Use unsafe fast gc for NPU.";
      gc.reset(new NPUUnsafeFastGarbageCollector(
          BOOST_GET_CONST(platform::NPUPlace, place_), max_memory_size));
    } else {
      PADDLE_THROW(platform::errors::Unimplemented(
          "Please set FLAGS_fast_eager_deletion_mode=true to use "
          "GarbageCollector on NPU."));
      // TODO(zhiqiu): fix bugs and enable NPUDefaultStreamGarbageCollector.
      VLOG(4) << "Use default stream gc for NPU.";
      gc.reset(new NPUDefaultStreamGarbageCollector(
          BOOST_GET_CONST(platform::NPUPlace, place_), max_memory_size));
    }
#endif
  }  // max_memory_size >= 0
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

}  // namespace framework
}  // namespace paddle
