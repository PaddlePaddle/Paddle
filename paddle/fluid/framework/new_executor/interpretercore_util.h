// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

/*************************************************************************
  > File Name: interpretercore_util.h
  > Author: guanshanshan@baidu.com
  > Created Time: Fri 23 Jul 2021 06:19:19 AM UTC
 ************************************************************************/

#pragma once

#include <chrono>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "paddle/fluid/framework/executor_gc_helper.h"
#include "paddle/fluid/framework/garbage_collector.h"
#include "paddle/fluid/framework/new_executor/new_executor_defs.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue.h"
#include "paddle/fluid/framework/new_executor/workqueue/workqueue_utils.h"
#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/program_desc.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/tensor.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/fluid/platform/init.h"

using AtomicVectorSizeT = std::vector<std::atomic<size_t>>;

namespace paddle {
namespace framework {

namespace interpreter {

class AsyncWorkQueue {
 public:
  AsyncWorkQueue(size_t host_num_threads, size_t deivce_num_threads,
                 EventsWaiter* waiter);

  std::future<std::unique_ptr<AtomicVectorSizeT>> PrepareAtomicDeps(
      const std::vector<size_t>& dependecy_count);
  std::future<std::unique_ptr<AtomicVectorSizeT>> PrepareAtomicVarRef(
      const std::vector<VariableMetaInfo>& vec_meta_info);

  // void WaitEmpty() { queue_group_->WaitQueueGroupEmpty(); }

  void AddTask(const OpFuncType& op_func_type, std::function<void()> fn);

  void Cancel() { queue_group_->Cancel(); }

 private:
  size_t host_num_thread_;
  std::unique_ptr<WorkQueueGroup> queue_group_;
};

std::unique_ptr<AtomicVectorSizeT> PrepareAtomicDeps(
    const std::vector<size_t>& dependecy_count);
std::unique_ptr<AtomicVectorSizeT> PrepareAtomicVarRef(
    const std::vector<VariableMetaInfo>& vec_meta_info);

void build_variable_scope(const framework::BlockDesc& block,
                          VariableScope* var_scope,
                          bool use_local_scope = true);

void build_op_func_list(const platform::Place& place,
                        const framework::BlockDesc& block,
                        const std::set<std::string>& skip_gc_vars,
                        std::vector<OpFuncNode>* vec_func_list,
                        VariableScope* var_scope, bool use_local_scope = true);

std::map<int, std::list<int>> build_op_downstream_map(
    const std::vector<Instruction>& vec_instruction,
    std::vector<std::vector<bool>>* op_happens_before);

void add_fetch(const std::vector<std::string>& fetch_names,
               framework::BlockDesc* block);

std::vector<size_t> merge_vector(const std::vector<size_t>& first,
                                 const std::vector<size_t>& second);

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
