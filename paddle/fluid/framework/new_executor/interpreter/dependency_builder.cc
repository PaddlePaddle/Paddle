// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/new_executor/interpreter/dependency_builder.h"

#include <queue>

// The difference between "sequential_run" and "serial_run":
// "sequential_run" dispatches OPs one by one according to the sequence in the
// Program, while "serial_run" ensures that all Ops are scheduled in a singal
// thread. In standalone executor, "sequential_run" is also "serial_run", while
// "serial_run" is not necessarily "sequential_run".
PADDLE_DEFINE_EXPORTED_bool(new_executor_sequential_run,
                            false,
                            "Enable sequential execution for standalone "
                            "executor, only applied to GPU OPs.");

namespace paddle {
namespace framework {
namespace interpreter {

size_t CountDownstreamMap(const std::map<int, std::set<int>>& downstream_map) {
  size_t count = 0;
  for (auto pair : downstream_map) {
    count += pair.second.size();
  }
  return count;
}

bool IsCommunicationOp(const std::string& op_name) {
  const std::set<std::string> special_comm_op_set = {
      "send",
      "recv",
      "send_v2",
      "recv_v2",
  };
  const std::string communication_op_prefix = "c_";
  if (op_name.find(communication_op_prefix) != std::string::npos ||
      special_comm_op_set.count(op_name)) {
    return true;
  }
  return false;
}

const std::string StringizeDownstreamMap(
    const std::map<int, std::set<int>>& downstream_map) {
  std::ostringstream oss;
  for (auto pair : downstream_map) {
    oss << pair.first << " -> ";
    std::copy(pair.second.begin(),
              pair.second.end(),
              std::ostream_iterator<int>(oss, " "));
    oss << std::endl;
  }
  return oss.str();
}

const std::map<int, std::set<int>>& DependencyBuilder::Build(
    const std::vector<Instruction>& instructions) {
  PADDLE_ENFORCE_EQ(
      is_build_,
      false,
      phi::errors::AlreadyExists("The op dependency has been built"));

  instructions_ = &instructions;
  op_num_ = instructions_->size();

  BuildDownstreamMap();
  BuildOpHappensBefore();
  ShrinkDownstreamMap();

  AddDependencyForCoalesceTensorOp();
  AddDependencyForCommunicationOp();
  AddDependencyForRandomOp();
  AddDependencyForReadOp();

  if (FLAGS_new_executor_sequential_run) {
    AddDependencyForSequentialRun();
  }

  is_build_ = true;

  VLOG(8) << "Finish build dependency";
  VLOG(8) << "downstream count: " << CountDownstreamMap(op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(op_downstream_map_);

  return op_downstream_map_;
}

bool DependencyBuilder::OpHappensBefore(int prior_op_idx,
                                        int posterior_op_idx) {
  PADDLE_ENFORCE_GE(
      op_happens_before_.size(),
      0,
      phi::errors::Unavailable("op_happen_before is not yet built"));
  return op_happens_before_.at(prior_op_idx).at(posterior_op_idx);
}

void DependencyBuilder::AddDependencyForCoalesceTensorOp() {
  const std::string kCoalesceTensor = "coalesce_tensor";
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBase()->Type() == kCoalesceTensor) {
      VLOG(4) << "Add depend for " << kCoalesceTensor << " " << op_idx;
      auto fused_out = instructions_->at(op_idx).Outputs().at("FusedOutput")[0];
      auto outputs = instructions_->at(op_idx).Outputs().at("Output");

      auto is_read = [](const Instruction& inst, int var_id) -> bool {
        for (auto pair : inst.Inputs()) {
          for (auto item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      auto is_write = [](const Instruction& inst, int var_id) -> bool {
        for (auto pair : inst.Outputs()) {
          for (auto item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      // find first op that reads fused_out
      auto first_read_fused_out_op = -1;
      for (auto j = op_idx + 1; j < op_num_; ++j) {
        if (is_read(instructions_->at(j), fused_out)) {
          first_read_fused_out_op = j;
          break;
        }
      }

      if (UNLIKELY(first_read_fused_out_op == -1)) {
        VLOG(4) << "No op read FusedOutput";
        continue;
      }

      // find ops that write 'outputs' between (op_index,
      // first_read_fused_out_op)
      // add depend: them->first_read_fused_out_op
      for (auto j = op_idx + 1;
           j < static_cast<size_t>(first_read_fused_out_op);
           ++j) {
        for (auto var_id : outputs) {
          if (is_write(instructions_->at(j), var_id)) {
            AddDownstreamOp(j, first_read_fused_out_op);
          }
        }
      }

      // find first op read 'outputs' between (first_read_fused_out_op, end)
      // add depned:  first_read_fused_out_op -> first op that reads 'outputs'

      // special case for consecutive communication ops, for example,
      // FusedOutput = c_sync_calc_stream(FusedOutput)
      // FusedOutput= c_allreduce_sum(FusedOutput)
      // FusedOutput = c_sync_comm_stream(FusedOutput)
      // we should take the last one to add depned instead of
      // 'first_read_fused_out_op'
      size_t target = first_read_fused_out_op;
      for (size_t j = first_read_fused_out_op + 1; j < op_num_; ++j) {
        if (j == target + 1 &&
            IsCommunicationOp(instructions_->at(target).OpBase()->Type()) &&
            IsCommunicationOp(instructions_->at(j).OpBase()->Type())) {
          VLOG(4) << "Found consecutive communication ops, "
                  << instructions_->at(target).OpBase()->Type() << " -> "
                  << instructions_->at(j).OpBase()->Type();
          target = j;
          continue;
        }

        for (auto var_id : outputs) {
          if (is_read(instructions_->at(j), var_id)) {
            AddDownstreamOp(target, j);
          }
        }
      }
    }
  }
}

void DependencyBuilder::AddDependencyForCommunicationOp() {
  auto IsCommunicationOp = [](std::string op) -> bool {
    const std::set<std::string> special_comm_op_set = {
        "send",
        "recv",
        "send_v2",
        "recv_v2",
    };
    const std::string communication_op_prefix = "c_";
    if (op.find(communication_op_prefix) != std::string::npos ||
        special_comm_op_set.count(op)) {
      return true;
    }
    return false;
  };

  int dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (IsCommunicationOp(instructions_->at(op_idx).OpBase()->Type())) {
      if (dependence_op_idx != -1) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }

  // TODO(zhiqiu): there still some cases not handled
  // add dependency for c_sync_comm_stream

  // in program, we can add only one c_sync_comm_stream to sync all
  // communication ops.
  // c_allreduce_sum(a)
  // c_allreduce_sum(b)
  // c_allreduce_sum(c)
  // c_sync_comm_stream(a)
  const std::string kSyncComm = "c_sync_comm_stream";
  dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBase()->Type() == kSyncComm) {
      dependence_op_idx = op_idx;
    } else {
      if (dependence_op_idx != -1) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
    }
  }
}

// make sure that the random op is scheduled sequentially
void DependencyBuilder::AddDependencyForRandomOp() {
  const std::set<std::string> random_op_set = {"bernoulli",
                                               "poisson",
                                               "multinomial",
                                               "gaussian_random",
                                               "truncated_gaussian_random",
                                               "uniform_random",
                                               "randint",
                                               "randperm",
                                               "exponential",
                                               "sampling_id",
                                               "dropout",
                                               "class_center_sample"};

  int dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (random_op_set.count(instructions_->at(op_idx).OpBase()->Type())) {
      if (dependence_op_idx != -1) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }
}

// equivalent to add_reader_dependency_pass
void DependencyBuilder::AddDependencyForReadOp() {
  std::vector<bool> is_startup_ops(op_num_, true);
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    auto it = op_downstream_map_.find(op_idx);
    if (it != op_downstream_map_.end()) {
      for (size_t downstream_op_idx : it->second) {
        is_startup_ops[downstream_op_idx] = false;
      }
    }
  }

  std::vector<size_t> read_ops;
  std::vector<size_t> startup_ops;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBase()->Type() == "read") {
      read_ops.push_back(op_idx);
    }

    if (is_startup_ops[op_idx]) {
      startup_ops.push_back(op_idx);
    }
  }

  for (size_t read_op_idx : read_ops) {
    for (size_t downstream_op_idx : startup_ops) {
      if (read_op_idx != downstream_op_idx &&
          !op_happens_before_[downstream_op_idx][read_op_idx]) {
        AddDownstreamOp(read_op_idx, downstream_op_idx);
      }
    }
  }
}

void DependencyBuilder::AddDependencyForSequentialRun() {
  int dependence_op_idx = -1;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (!IsCpuOp(instructions_->at(op_idx))) {
      if (dependence_op_idx != -1) {
        AddDownstreamOp(dependence_op_idx, op_idx);
      }
      dependence_op_idx = op_idx;
    }
  }
}

void DependencyBuilder::AddDownstreamOp(int prior_op_idx,
                                        int posterior_op_idx) {
  std::set<int>& downstream_ops = op_downstream_map_[prior_op_idx];

  if (op_happens_before_.size() != 0) {
    PADDLE_ENFORCE_EQ(
        op_happens_before_[posterior_op_idx][prior_op_idx],
        false,
        phi::errors::Unavailable(
            "Can not add dependency %d->%d because %d is run before %d",
            prior_op_idx,
            posterior_op_idx,
            posterior_op_idx,
            prior_op_idx));

    for (int op_idx : downstream_ops) {
      if (op_happens_before_[op_idx][posterior_op_idx]) {
        VLOG(7) << "Find dependencies " << prior_op_idx << "->" << op_idx
                << "->" << posterior_op_idx << ", skip adding " << prior_op_idx
                << "->" << posterior_op_idx;
        return;
      }
    }
  }

  downstream_ops.insert(posterior_op_idx);

  if (op_happens_before_.size() != 0) {
    for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
      if (op_happens_before_[posterior_op_idx][op_idx]) {
        op_happens_before_[prior_op_idx][op_idx] = true;
      }
    }
  }
  VLOG(8) << prior_op_idx << "->" << posterior_op_idx;
  VLOG(8) << "Add dependency from "
          << instructions_->at(prior_op_idx).OpBase()->Type() << "("
          << prior_op_idx << ") to "
          << instructions_->at(posterior_op_idx).OpBase()->Type() << "("
          << posterior_op_idx << ")";
}

void DependencyBuilder::BuildDownstreamMap() {
  auto var2min_rw_op =
      std::map<int, std::list<int>>();  // # map from variable id to read /
                                        // write op id.
  auto var2recent_write_op =
      std::map<int, int>();  // # map from variable to recent write op.
  auto op2dependences =
      std::map<int, std::set<int>>();  //# map from op to the dependence list,
                                       // op must run after the dependence.
  std::set<int>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    op2dependences[op_idx] = std::set<int>();
  }

  auto update_var_min_rw_op =
      [](const std::map<int, std::set<int>>& op2dependences,
         std::map<int, std::list<int>>* var2min_rw_op,
         int cur_op,
         int rw_var) {
        // rw_var is inputs or outputs of cur_op
        // this function update the var2min_rw_op set .
        if (var2min_rw_op->find(rw_var) == var2min_rw_op->end()) {
          (*var2min_rw_op)[rw_var] = std::list<int>();
        }
        for (auto dep_op : op2dependences.at(cur_op)) {
          var2min_rw_op->at(rw_var).remove(dep_op);
        }
        var2min_rw_op->at(rw_var).push_back(cur_op);
      };

  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    remove_duplicate.clear();
    // step1: update the op2dependences structure
    for (auto& item :
         instructions_->at(op_idx).Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (var2recent_write_op.count(var))
          op2dependences[op_idx].insert(var2recent_write_op[var]);
      }
    }

    for (auto& item :
         instructions_->at(op_idx).Outputs()) {  // for all write vars
      for (auto var : item.second) {
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }
    // the original output of inplace op is also change.
    if (!instructions_->at(op_idx).InplaceBackMap().empty()) {
      auto& m = instructions_->at(op_idx).InplaceBackMap();
      for (auto& p : m) {
        auto& var = p.second;
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }

    // step2: update 2 var2xxxx data structure
    for (auto& item :
         instructions_->at(op_idx).Outputs()) {  // for all write vars
      for (auto var : item.second) {
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<int>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    // NOTE(zhiqiu): The inplace op with `transfer` also changes
    // original output after that so add original output as well
    // original: a->op->a
    // after: a->data_transfer->a'->op->a'->transfer_back->a
    // which means op writes a and a'
    if (!instructions_->at(op_idx).InplaceBackMap().empty()) {
      auto& m = instructions_->at(op_idx).InplaceBackMap();
      for (auto& p : m) {
        auto var = p.second;
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<int>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         instructions_->at(op_idx).Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          update_var_min_rw_op(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // convert op2dependences to downstream_map directly. op2dependences is op ->
  // it's dependences, we want to get op -> [next ops] map, where ops is the
  // next instruction of op. The size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  for (auto& item : op2dependences) {
    int op = item.first;
    for (auto dep_op : item.second) {
      AddDownstreamOp(dep_op, op);
    }
  }

  VLOG(6) << "downstream count: " << CountDownstreamMap(op_downstream_map_);
  VLOG(6) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(op_downstream_map_);
}

void DependencyBuilder::BuildOpHappensBefore() {
  // happens_before[i][j] means i should be executed before j
  op_happens_before_.assign(op_num_, std::vector<bool>(op_num_, false));

  // bfs to get all next ops
  auto bfs = [&](size_t op_idx) {
    std::queue<size_t> q;
    std::vector<bool> visited(op_num_, false);
    q.push(op_idx);
    while (!q.empty()) {
      size_t op = q.front();
      q.pop();
      visited[op] = true;
      if (!op_downstream_map_.count(op)) {
        continue;
      }
      for (auto next : op_downstream_map_.at(op)) {
        if (!visited[next]) {
          PADDLE_ENFORCE_EQ(op_happens_before_[next][op_idx],
                            false,
                            paddle::platform::errors::AlreadyExists(
                                "There exists circle in graph, expected "
                                "%d->%d, but already got %d->%d",
                                op_idx,
                                next,
                                next,
                                op_idx));
          op_happens_before_[op_idx][next] = true;
          VLOG(8) << "happens before: " << op_idx << " " << next;
          q.push(next);
        }
      }
    }
  };

  for (size_t i = 0; i < op_num_; ++i) {
    bfs(i);
  }
}

void DependencyBuilder::ShrinkDownstreamMap() {
  // remove unnecessary downstream ops
  // for example, a->b->c
  // a: b, c
  // b: c
  // =>
  // a: b
  // b: c

  // shrink, find the downstream op that has no other op in the
  // downstream list happens before it
  for (size_t i = 0; i < op_num_; ++i) {
    if (op_downstream_map_.find(i) == op_downstream_map_.end()) {
      continue;
    }

    std::set<int> minumum_nexts;
    for (size_t item : op_downstream_map_.at(i)) {
      bool not_after_any = true;
      // find the op that is not executed after any
      for (size_t other_item : op_downstream_map_.at(i)) {
        if (op_happens_before_[other_item][item]) {
          VLOG(8) << "happens_before: " << other_item << "->" << item
                  << ", so skip " << item;
          not_after_any = false;
          break;
        }
      }
      if (not_after_any) {
        VLOG(8) << "downstream op of " << i << ": " << item;
        minumum_nexts.insert(item);
      }
    }
    op_downstream_map_.at(i) = minumum_nexts;
  }
  VLOG(6) << "downstream count: " << CountDownstreamMap(op_downstream_map_);
  VLOG(6) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(op_downstream_map_);
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
