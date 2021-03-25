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

#if defined(PADDLE_WITH_NCCL) || defined(PADDLE_WITH_RCCL)
#include <float.h>
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {

class TrainerDesc;

uint64_t SectionWorker::batch_id_(0);

void SectionWorker::Initialize(const TrainerDesc &desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(
      new ProgramDesc(desc.section_param().section_config().program_desc()));
  for (auto &op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void SectionWorker::RunForward(
    int micro_id, std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    // We run op with op_role = kLRSched only for the first microbatch
    // to avoid increasing the @LR_DECAY_STEP@ multiple times.
    bool run_first_mbatch = op_role == static_cast<int>(OpRole::kForward) ||
                            op_role == (static_cast<int>(OpRole::kForward) |
                                        static_cast<int>(OpRole::kLoss)) ||
                            op_role == static_cast<int>(OpRole::kLRSched);
    bool run_others = op_role == static_cast<int>(OpRole::kForward) ||
                      op_role == (static_cast<int>(OpRole::kForward) |
                                  static_cast<int>(OpRole::kLoss));
    if ((micro_id == 0 && run_first_mbatch) || (micro_id != 0 && run_others)) {
      VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*microbatch_scopes_[micro_id], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[micro_id], op.get(),
                            unused_vars_, gc.get());
      }
    }
  }
}

void SectionWorker::RunBackward(
    int micro_id, std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    if (op_role == static_cast<int>(OpRole::kBackward) ||
        op_role == (static_cast<int>(OpRole::kBackward) |
                    static_cast<int>(OpRole::kLoss))) {
      VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
              << micro_id;
      op->Run(*microbatch_scopes_[micro_id], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[micro_id], op.get(),
                            unused_vars_, gc.get());
      }
    }
  }
}

void SectionWorker::RunUpdate(
    std::unique_ptr<GarbageCollector> &gc,
    std::unordered_map<const OperatorBase *, std::vector<std::string>>
        &unused_vars_) {
  for (auto &op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "Update: running op " << op->Type();
      op->Run(*microbatch_scopes_[num_microbatches_ - 1], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[num_microbatches_ - 1],
                            op.get(), unused_vars_, gc.get());
      }
    }
  }
}

void SectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
  if (max_memory_size >= 0) {
#if defined(PADDLE_WITH_CUDA) || defined(PADDLE_WITH_HIP)
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
    }
#endif
  }

  if (schedule_mode_ == 0) {
    // F-then-B scheduler which runs Forward phase for all microbatches,
    // then runs Backward phase for all microbatches.
    // step1: run forward
    for (int i = 0; i < num_microbatches_; ++i) {
      RunForward(i, gc, unused_vars_);
    }
    // step2: run backward
    for (int i = 0; i < num_microbatches_; ++i) {
      RunBackward(i, gc, unused_vars_);
    }
    // step3: run update
    RunUpdate(gc, unused_vars_);
  } else {
    // 1F1B scheduler, which runs forward phase and backward phase altertively
    // after startup phase. For a stage, the number of microbatches for
    // startup is num_pipeline_stages_ - pipeline_stage_ - 1, where
    // num_pipeline_stages_ is the total number of pipeline stages and
    // pipeline_stage_ is the pipeline stage of the current device.
    auto startup_steps = num_pipeline_stages_ - pipeline_stage_ - 1;
    VLOG(3) << "startup_steps:" << startup_steps
            << ", num_stages: " << num_pipeline_stages_
            << ", stage:" << pipeline_stage_;
    PADDLE_ENFORCE_GT(
        num_microbatches_, startup_steps,
        platform::errors::InvalidArgument(
            "To use pipeline with 1F1B scheduler, please make sure number of "
            "microbatches (%d) is than startup steps (%d).",
            num_microbatches_, startup_steps));
    int fw_step = 0;
    int bw_step = 0;
    // startup phase
    while (fw_step < startup_steps) {
      RunForward(fw_step, gc, unused_vars_);
      fw_step += 1;
    }

    // 1f1b phase
    while (fw_step < num_microbatches_) {
      RunForward(fw_step, gc, unused_vars_);
      fw_step += 1;
      RunBackward(bw_step, gc, unused_vars_);
      bw_step += 1;
    }
    // backward phase
    while (bw_step < num_microbatches_) {
      RunBackward(bw_step, gc, unused_vars_);
      bw_step += 1;
    }
    RunUpdate(gc, unused_vars_);
  }
  dev_ctx_->Wait();
  ++batch_id_;
}

}  // namespace framework
}  // namespace paddle
#endif
