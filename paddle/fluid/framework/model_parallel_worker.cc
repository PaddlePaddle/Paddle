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

std::atomic<int> ModelParallelWorker::cpu_id_(0);

void ModelParallelWorker::Initialize(const TrainerDesc& trainer_desc) {
  dev_ctx_ = platform::DeviceContextPool::Instance().Get(place_);
  std::shared_ptr<framework::ProgramDesc> program;
  program.reset(new ProgramDesc(
      trainer_desc.section_param().section_config(section_id_).program_desc()));
  for (auto& op_desc : program->Block(0).AllOps()) {
    ops_.push_back(OpRegistry::CreateOp(*op_desc));
  }
  VLOG(3) << "program size: " << program->Size();
}

void ModelParallelWorker::AutoSetCPUAffinity(bool reuse) {
  int thread_cpu_id = cpu_id_.fetch_add(1);

  unsigned concurrency_cap = std::thread::hardware_concurrency();
  unsigned proc = thread_cpu_id;

  if (proc >= concurrency_cap) {
    if (reuse) {
      proc %= concurrency_cap;
    } else {
      LOG(INFO) << "All " << concurrency_cap
                << " CPUs have been set affinities. Fail to set the "
                << thread_cpu_id << "th thread";
      return;
    }
  }

  cpu_set_t mask;
  CPU_ZERO(&mask);
  CPU_SET(proc, &mask);

  if (-1 == sched_setaffinity(0, sizeof(mask), &mask)) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
    return;
  }

  CPU_ZERO(&mask);
  if ((0 != sched_getaffinity(0, sizeof(mask), &mask)) ||
      (0 == CPU_ISSET(proc, &mask))) {
    LOG(WARNING) << "Fail to set thread affinity to CPU " << proc;
  }
  VLOG(3) << "Set " << thread_cpu_id << "th thread affinity to CPU " << proc;
}

void ModelParallelWorker::TrainFiles() {
  AutoSetCPUAffinity(true);

  if (thread_id_ == 0) {
    // For the first section.
    PADDLE_ENFORCE_NOT_NULL(device_reader_,
                            "The device reader for the first "
                            "thread should not be null.");
    device_reader_->Start();
    device_reader_->AssignFeedVar(*minibatch_scope_);
    while (true) {
      // Start a minibatch.
      // forward pass:
      for (int i = 0; i < num_macrobatches_; ++i) {
        device_reader_->Next();
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          if (op_role == static_cast<int>(OpRole::kForward) ||
              op_role == (static_cast<int>(OpRole::kForward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*macrobatch_scopes_[i], place_);
          }
        }
      }
      // backward pass
      for (int i = 0; i < num_macrobatches_; ++i) {
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*macrobatch_scopes_[i], place_);
          }
        }
      }
      // update pass
      for (auto& op : ops_) {
        int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
        if (op_role == static_cast<int>(OpRole::kOptimize) ||
            op_role == static_cast<int>(OpRole::kLRSched)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope";
          // op->Run(*minibatch_scope_, place_);
          op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
        }
      }
      dev_ctx_->Wait();
    }
  } else {
    while (true) {
      // forward pass:
      for (int i = 0; i < num_macrobatches_; ++i) {
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          if (op_role == static_cast<int>(OpRole::kForward) ||
              op_role == (static_cast<int>(OpRole::kForward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*macrobatch_scopes_[i], place_);
          }
        }
      }
      // backward pass
      for (int i = 0; i < num_macrobatches_; ++i) {
        for (auto& op : ops_) {
          int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
          if (op_role == static_cast<int>(OpRole::kBackward) ||
              op_role == (static_cast<int>(OpRole::kBackward) |
                          static_cast<int>(OpRole::kLoss))) {
            VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                    << " for scope " << i;
            op->Run(*macrobatch_scopes_[i], place_);
          }
        }
      }
      // update pass
      for (auto& op : ops_) {
        int op_role = boost::get<int>(op->Attr<int>(std::string("op_role")));
        if (op_role == static_cast<int>(OpRole::kOptimize) ||
            op_role == static_cast<int>(OpRole::kLRSched)) {
          VLOG(3) << "running an op " << op->Type() << " for " << thread_id_
                  << " for minibatch scope ";
          op->Run(*macrobatch_scopes_[num_macrobatches_ - 1], place_);
          cudaDeviceSynchronize();
        }
      }
      dev_ctx_->Wait();
    }
  }
}

void ModelParallelWorker::TrainFilesWithProfiler() {
  VLOG(3) << "Not implemented now.";
}

}  // namespace framework
}  // namespace paddle
