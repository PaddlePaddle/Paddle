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

#if defined(PADDLE_WITH_NCCL)
#include <float.h>
#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/program_desc.h"

#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/message.h"
#include "google/protobuf/text_format.h"

#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/fleet/box_wrapper.h"
#include "paddle/fluid/framework/tensor_util.h"
#include "paddle/fluid/framework/trainer_desc.pb.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

uint64_t SectionWorker::batch_id_(0);

void SectionWorker::Initialize(const TrainerDesc& desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  program_.reset(
      new ProgramDesc(desc.section_param().section_config().program_desc()));
  for (auto& op_desc : program_->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
}

void SectionWorker::TrainFiles() {
  VLOG(5) << "begin section_worker TrainFiles";

  int64_t max_memory_size = GetEagerDeletionThreshold();
  std::unique_ptr<GarbageCollector> gc;
  auto unused_vars_ = GetUnusedVars(program_->Block(0), ops_, skip_vars_);
  if (max_memory_size >= 0) {
#ifdef PADDLE_WITH_CUDA
    if (platform::is_gpu_place(place_)) {
      if (IsFastEagerDeletionModeEnabled()) {
        gc.reset(new UnsafeFastGPUGarbageCollector(
            BOOST_GET_CONST(platform::CUDAPlace, place_), max_memory_size));
      }
    }
#endif
  }

  for (int i = 0; i < num_microbatches_; ++i) {
    for (auto& op : ops_) {
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
      if ((i == 0 && run_first_mbatch) || (i != 0 && run_others)) {
        VLOG(3) << "Forward: running op " << op->Type() << " for micro-batch "
                << i;
        op->Run(*microbatch_scopes_[i], place_);
        if (gc) {
          DeleteUnusedTensors(*microbatch_scopes_[i], op.get(), unused_vars_,
                              gc.get());
        }
      }
    }
    cudaDeviceSynchronize();
  }

  // backward pass
  for (int i = 0; i < num_microbatches_; ++i) {
    for (auto& op : ops_) {
      int op_role = op->Attr<int>(std::string("op_role"));
      if (op_role == static_cast<int>(OpRole::kBackward) ||
          op_role == (static_cast<int>(OpRole::kBackward) |
                      static_cast<int>(OpRole::kLoss))) {
        VLOG(3) << "Backward: running op " << op->Type() << " for micro-batch "
                << i;
        op->Run(*microbatch_scopes_[i], place_);
        if (gc) {
          DeleteUnusedTensors(*microbatch_scopes_[i], op.get(), unused_vars_,
                              gc.get());
        }
      }
    }
    cudaDeviceSynchronize();
  }

  // update pass
  for (auto& op : ops_) {
    int op_role = op->Attr<int>(std::string("op_role"));
    if (op_role == static_cast<int>(OpRole::kOptimize)) {
      VLOG(3) << "Update: running op " << op->Type();
      op->Run(*microbatch_scopes_[0], place_);
      if (gc) {
        DeleteUnusedTensors(*microbatch_scopes_[0], op.get(), unused_vars_,
                            gc.get());
      }
    }
  }
  dev_ctx_->Wait();
  ++batch_id_;
}

}  // namespace framework
}  // namespace paddle
#endif
