// Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/details/double_queue_threaded_ssa_graph_executor.h"
#include "paddle/fluid/framework/details/multi_devices_helper.h"

namespace paddle {
namespace framework {
namespace details {

DoubleQueueThreadedSSAGraphExecutor::DoubleQueueThreadedSSAGraphExecutor(
    const ExecutionStrategy &strategy, const std::vector<Scope *> &local_scopes,
    const std::vector<platform::Place> &places,
    std::unique_ptr<ir::Graph> &&graph)
    : strategy_(strategy),
      local_scopes_(local_scopes),
      places_(places),
      graph_(std::move(graph)),
      fetch_ctxs_(places),
      prepare_next_pool_(1) {
  PADDLE_ENFORCE_EQ(strategy_.type_,
                    ExecutionStrategy::ExecutorType::kDoubleQueue);
  PADDLE_ENFORCE_GT(strategy_.max_queue_size_, 0);

  auto &ops = graph_->Get<details::GraphOps>(details::kGraphOps);

  for (auto &op : ops) {
    pending_ops_[op.get()];  // create empty set for each op
    op_deps_[op.get()] = 0;
    for (auto *input : op->Inputs()) {
      if (input->GeneratedOp() != nullptr) {
        pending_ops_[input->GeneratedOp()].insert(op.get());
      }
    }
  }

  for (auto &pending_op_pair : pending_ops_) {
    for (auto *pending_op : pending_op_pair.second) {
      ++op_deps_[pending_op];
    }
  }

  local_queues_.resize(strategy_.num_threads_);
  for (auto &queue : local_queues_) queue.ResetAndReserve(ops.size());
  global_queue_.ResetAndReserve(ops.size());

  for (auto &op : ops) {
    if (op_deps_[op.get()] == 0) {
      if (local_queues_[init_queue_idx_].size() < strategy_.max_queue_size_) {
        local_queues_[init_queue_idx_].Push(op.get());
      } else {
        global_queue_.Push(op.get());
      }
      init_queue_idx_ = (init_queue_idx_ + 1) % local_queues_.size();
    }
  }

  global_queue_.MarkAsInit();
  for (size_t i = 0; i < strategy_.num_threads_; ++i) {
    auto &local_queue = local_queues_[i];
    local_queue.MarkAsInit();
    thread_pool_.emplace_back([&, i, this]() {
      std::string thread_id = "Thread-" + std::to_string(i) + "/" +
                              std::to_string(strategy_.num_threads_) +
                              "@ParallelExecutor: ";
      std::vector<OpHandleBase *> tmp;
      while (control_flag_.Wait()) {
        ++running_threads_;
        auto &op_deps = *rt_op_deps_;
        auto &pending_ops = *rt_pending_ops_;
        for (;;) {
          bool is_ended;
          do {
            size_t cur_tail = local_queue.tail_;
            size_t new_queue_size = 0;
            tmp.clear();
            is_ended = false;
            while (!local_queue.IsEmpty()) {
              auto *op = local_queue.Pop();
              try {
                VLOG(10) << thread_id << "begin to run op: " << op->Name();
                op->Run(strategy_.use_cuda_);
                VLOG(10) << thread_id << "end to run op: " << op->Name();
                // Handle op_deps
                for (auto *pending_op : pending_ops.at(op)) {
                  if (op_deps.at(pending_op).fetch_sub(1) == 1) {
                    // add it to local_queue
                    if (new_queue_size < strategy_.max_queue_size_) {
                      local_queue.queue_[cur_tail++] = pending_op;
                      ++new_queue_size;
                    } else {
                      tmp.push_back(pending_op);
                    }
                  }
                }

                // NOTE(zjl): not quite sure, but using fetch_and_add may be
                // faster than add_and_fetch
                // Fetch_and_add is usually an atomic instruction in some CPUs,
                // (see: https://en.wikipedia.org/wiki/Fetch-and-add)
                // while add_and_fetch may be implemented by CAS and while-loop
                // (see: https://en.wikipedia.org/wiki/Compare-and-swap).
                if (++run_op_num_ >= op_deps.size()) {
                  is_ended = true;
                  break;
                }
              } catch (...) {
                VLOG(5) << thread_id
                        << "raise exception when running op: " << op->Name();
                {
                  std::lock_guard<std::mutex> lock(exit_mutex_);
                  exception_ptr_ = std::current_exception();
                }
                run_op_num_ += (op_deps.size() + 1);
                is_ended = true;
                break;
              }
            }

            if (is_ended) {
              control_flag_.Pause();
              {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                queue_cv_.notify_all();
              }
              VLOG(10) << thread_id << "queue_cv_.notify_all() is called";
              break;
            }

            local_queue.tail_ = cur_tail;
            if (!tmp.empty()) {
              // NOTE(zjl): This can be done async
              std::lock_guard<std::mutex> lock(queue_mutex_);
              global_queue_.PushAll(tmp);
              if (global_queue_.size() <= strategy_.max_queue_size_) {
                queue_cv_.notify_one();
              } else {
                queue_cv_.notify_all();
              }
            }
          } while (!local_queue.IsEmpty());

          if (is_ended) break;

          size_t pop_head, pop_num;
          {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [&] {
              return !global_queue_.IsEmpty() || !control_flag_.IsStarted();
            });
            if (global_queue_.IsEmpty() || !control_flag_.IsStarted()) {
              break;
            }
            pop_num = global_queue_.PopN(strategy_.max_queue_size_, &pop_head);
          }
          local_queue.Push(global_queue_, pop_head, pop_head + pop_num);
        }

        if (running_threads_.fetch_sub(1) == 1) {
          std::lock_guard<std::mutex> lock(exit_mutex_);
          exit_cv_.notify_one();
        }
      }
      VLOG(10) << thread_id << "stops, destructor called";
    });
  }

  PrepareNext();
}

DoubleQueueThreadedSSAGraphExecutor::~DoubleQueueThreadedSSAGraphExecutor() {
  control_flag_.Stop();
  for (auto &th : thread_pool_) {
    th.join();
  }
  prepare_next_future_.wait();
}

FeedFetchList DoubleQueueThreadedSSAGraphExecutor::Run(
    const std::vector<std::string> &fetch_tensors) {
  prepare_next_future_.wait();

  paddle::framework::FeedFetchList fetches;
  fetches.resize(fetch_tensors.size());
  std::unordered_map<std::string, std::vector<VarHandleBase *>> fetched_vars;
  std::vector<std::unique_ptr<FetchOpHandle>> fetch_ops;

  for (auto &fetch_var_name : fetch_tensors) {
    for (auto &var_map : graph_->Get<details::GraphVars>(details::kGraphVars)) {
      auto it = var_map.find(fetch_var_name);
      if (it != var_map.end()) {
        fetched_vars[fetch_var_name].push_back(it->second.rbegin()->get());
      }
    }
  }

  {
    // init queues
    size_t reserve_size = rt_op_deps_->size() + fetch_tensors.size();
    for (auto &queue : local_queues_) queue.ResetAndReserve(reserve_size);
    global_queue_.ResetAndReserve(reserve_size);
  }

  size_t queue_idx = init_queue_idx_;
  for (size_t i = 0; i < fetch_tensors.size(); ++i) {
    auto &var_name = fetch_tensors[i];
    auto fetched_var_it = fetched_vars.find(var_name);
    PADDLE_ENFORCE(fetched_var_it != fetched_vars.end(),
                   "Cannot find fetched variable.(Perhaps the main_program "
                   "is not set to ParallelExecutor)");

    auto &vars = fetched_var_it->second;

    ir::Node *fetch_node =
        graph_->CreateEmptyNode("fetch", ir::Node::Type::kOperation);
    auto *op = new FetchOpHandle(fetch_node, &fetches, i, &local_scopes_);
    fetch_ops.emplace_back(op);

    for (auto &p : places_) {
      op->SetDeviceContext(p, fetch_ctxs_.Get(p));
    }

    (*rt_pending_ops_)[op];
    std::unordered_set<OpHandleBase *> preceding_ops;
    for (auto *var : vars) {
      op->AddInput(var);
      if (var->GeneratedOp() != nullptr) {
        preceding_ops.insert(var->GeneratedOp());
        (*rt_pending_ops_)[var->GeneratedOp()].insert(op);
      }
    }

    (*rt_op_deps_)[op] = preceding_ops.size();

    if (preceding_ops.empty()) {
      if (local_queues_[queue_idx].size() < strategy_.max_queue_size_) {
        local_queues_[queue_idx].Push(op);
      } else {
        global_queue_.Push(op);
      }
      queue_idx = (queue_idx + 1) % local_queues_.size();
    }
  }

  {
    run_op_num_ = 0;
    running_threads_ = 0;
    control_flag_.Start();
    std::unique_lock<std::mutex> lock(exit_mutex_);
    exit_cv_.wait(lock, [&] {
      return run_op_num_ >= rt_op_deps_->size() && running_threads_ == 0;
    });
  }

  ClearFetchOp(graph_.get(), &fetch_ops);
  if (UNLIKELY(run_op_num_ > rt_op_deps_->size())) {
    VLOG(5) << "Exception raised, there are "
            << run_op_num_ % (rt_op_deps_->size() + 1) << " have run";
    PrepareNext();
    std::rethrow_exception(exception_ptr_);
  } else {
    PrepareNext();
  }
  return fetches;
}

const ir::Graph &DoubleQueueThreadedSSAGraphExecutor::Graph() const {
  return *graph_;
}

void DoubleQueueThreadedSSAGraphExecutor::PrepareNext() {
  prepare_next_future_ = prepare_next_pool_.enqueue([&] {
    using RuntimeOpDepsType =
        std::remove_reference<decltype(*rt_op_deps_)>::type;
    using RuntimePendingOpsType =
        std::remove_reference<decltype(*rt_pending_ops_)>::type;

    rt_op_deps_.reset(new RuntimeOpDepsType());
    for (auto &pair : op_deps_) {
      (*rt_op_deps_)[pair.first] = pair.second;
    }

    rt_pending_ops_.reset(new RuntimePendingOpsType(pending_ops_));
  });
}

}  // namespace details
}  // namespace framework
}  // namespace paddle
