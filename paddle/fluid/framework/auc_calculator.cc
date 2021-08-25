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

#include <sstream>
#include "paddle/fluid/framework/context_callback.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/trainer.h"
#include "paddle/fluid/framework/trainer_context.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

namespace paddle {
namespace framework {

void FeedAucCalculator::thread_pulled_callback(TrainerContextInterface *context,
                                               DeviceWorker *worker_base) {
  auto *worker = (DownpourWorker *)worker_base;  // NOLINT
  auto device_reader = worker->device_reader_;
  int batch_size = device_reader->GetCurBatchSize();
  instance_count_ += batch_size;
}

template <typename T>
void MergeToRootScope(LoDTensor *root_tensor, LoDTensor *tensor) {
  T *root_data = root_tensor->data<T>();
  T *data = tensor->data<T>();
  for (int i = 0; i < tensor->numel(); i++) {
    root_data[i] += data[i];
  }
}

void FeedAucCalculator::trainer_end_callback(TrainerContextInterface *context) {
  auto *trainer = (DistMultiTrainer *)(context->trainer_);  // NOLINT
  // auto* trainer_context = (FeedTrainerContext*)context;
  auto &need_merge_var_names = trainer->need_merge_var_names_;
  for (size_t i = 0; i < need_merge_var_names.size(); i++) {
    Variable *root_var = context->root_scope_->FindVar(need_merge_var_names[i]);
    if (root_var == nullptr) {
      continue;
    }
    LoDTensor *root_tensor = root_var->GetMutable<LoDTensor>();
    for (size_t j = 1; j < context->thread_num_; j++) {
      Scope *cur_thread_scope = trainer->workers_[j]->GetThreadScope();
      Variable *thread_var = cur_thread_scope->FindVar(need_merge_var_names[i]);
      LoDTensor *thread_tensor = thread_var->GetMutable<LoDTensor>();
      if (root_tensor->numel() != thread_tensor->numel()) {
        continue;
      }
#define MergeCallback(cpp_type, proto_type)                                   \
  do {                                                                        \
    if (root_tensor->type() == proto_type) {                                  \
      if (thread_tensor->type() != proto_type) {                              \
        VLOG(0) << "Error: thread id=" << j << ", need_merge_var_names[" << i \
                << "] " << need_merge_var_names[i]                            \
                << ", root tensor type=" << root_tensor->type()               \
                << ", thread tensor type=" << thread_tensor->type();          \
        exit(-1);                                                             \
      }                                                                       \
      MergeToRootScope<cpp_type>(root_tensor, thread_tensor);                 \
    }                                                                         \
  } while (0)
      _ForEachDataType_(MergeCallback);
    }
  }
  // double ins_num = instance_count_;
  // double all_ins = trainer_context->MPI()->AllReduce(ins_num, MPI_SUM,
  // trainer_context->comm_);
}

}  // namespace framework
}  // namespace paddle
