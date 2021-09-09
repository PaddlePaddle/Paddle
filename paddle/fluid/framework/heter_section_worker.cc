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

#include <float.h>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class TrainerDesc;

uint64_t HeterSectionWorker::batch_id_(0);

void HeterSectionWorker::Initialize(const TrainerDesc &desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(
      new ProgramDesc(desc.section_param().section_config().program_desc()));
  for (auto &op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void HeterSectionWorker::BatchBarrier() {
  for (auto &op : ops_) {
    auto op_type = op->Type();
    if (op_type != "send_and_recv") continue;
    auto op_mode = op->Attr<std::string>(std::string("mode"));
    if (op_mode == "barrier") {
      VLOG(3) << op_mode << " "
              << op->Attr<std::string>(std::string("message_name"));
      op->Run(*microbatch_scopes_[trainer_id_], place_);
    }
  }
}

void HeterSectionWorker::TrainerBarrier() {
  for (auto &op : ops_) {
    auto op_type = op->Type();
    if (op_type != "trainer_barrier") continue;
    op->Run(*root_scope_, place_);
  }
}

void HeterSectionWorker::RunListen() {
  for (auto &op : ops_) {
    auto op_type = op->Type();
    if (op_type == "heter_listen_and_serv") {
      op->Run(*root_scope_, place_);
    }
  }
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
  }

  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    auto op_type = op->Type();

    if (op_type == "heter_listen_and_serv") continue;
    if (op_type == "send_and_recv") {
      auto op_mode = op->Attr<std::string>(std::string("mode"));
      if (op_mode == "barrier") continue;
    }
    if (op_type == "trainer_barrier") continue;
    bool run_first_mbatch = (op_role == static_cast<int>(OpRole::kForward)) ||
                            (op_role == (static_cast<int>(OpRole::kForward) |
                                         static_cast<int>(OpRole::kLoss))) ||
                            (op_role == static_cast<int>(OpRole::kRPC)) ||
                            (op_role == static_cast<int>(OpRole::kLRSched));

    bool run_others = (op_role == static_cast<int>(OpRole::kForward)) ||
                      (op_role == (static_cast<int>(OpRole::kForward) |
                                   static_cast<int>(OpRole::kLoss))) ||
                      (op_role == static_cast<int>(OpRole::kRPC));

    if ((micro_id == 0 && run_first_mbatch) || (micro_id != 0 && run_others)) {
      VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*microbatch_scopes_[micro_id], place_);
    }
  }
}

void HeterSectionWorker::BindingDataFeedMemory(int micro_id) {
  const std::vector<std::string> &input_feed =
      device_reader_->GetUseSlotAlias();
  for (auto name : input_feed) {
    device_reader_->AddFeedVar(microbatch_scopes_[micro_id]->FindVar(name),
                               name);
  }
}

void HeterSectionWorker::Run() {
  bool is_first_stage = (pipeline_stage_ == 0);
  if (listen_ptr == nullptr) {
    listen_ptr.reset(
        new std::thread(std::bind(&HeterSectionWorker::RunListen, this)));
  }

  if (is_first_stage) {
    for (int i = trainer_id_; i < num_microbatches_; i += trainers_) {
      VLOG(5) << "Run " << i << " stage" << std::endl;
      RunForward(i);
      if (epoch_finish_ == true) return;
    }
  }
}

void HeterSectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";
  // VLOG(2) << "mini batch steps:" << batch_id_;

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
  while (true) {
    Run();
    dev_ctx_->Wait();
    if (epoch_finish_ == true) return;
    if (is_first_stage) {
      BatchBarrier();
    }
    ++batch_id_;
  }
}
}  // namespace framework
}  // namespace paddle
