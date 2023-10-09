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
#include "paddle/fluid/framework/new_executor/instruction/instruction_base.h"
#include "paddle/fluid/framework/new_executor/interpreter/interpreter_util.h"
#include "paddle/fluid/platform/flags.h"
PADDLE_DEFINE_EXPORTED_bool(
    add_dependency_for_communication_op,
    true,
    "Whether to add dependency for communication Ops. It is just a temporary "
    "FLAGS especially for auto parallel to avoid the concurrency damage by the "
    "communication dependency added in standalone executor.");

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

size_t CountDownstreamMap(
    const std::map<size_t, std::set<size_t>>& downstream_map) {
  size_t count = 0;
  for (auto pair : downstream_map) {
    count += pair.second.size();
  }
  return count;
}
const std::string StringizeDownstreamMap(
    const std::map<size_t, std::set<size_t>>& downstream_map) {
  std::ostringstream oss;
  for (auto pair : downstream_map) {
    oss << pair.first << " -> ";
    std::copy(pair.second.begin(),
              pair.second.end(),
              std::ostream_iterator<size_t>(oss, " "));
    oss << std::endl;
  }
  return oss.str();
}

DependencyBuilder::DependencyBuilder()
    : is_build_(false), instructions_(nullptr) {
  op_downstream_map_ = std::make_shared<std::map<size_t, std::set<size_t>>>();
  op_happens_before_ = std::make_shared<std::vector<std::vector<bool>>>();
}

const std::map<size_t, std::set<size_t>>& DependencyBuilder::Build(
    const std::vector<Instruction>& instructions) {
  if (is_build_) {
    return *op_downstream_map_;
  }

  std::tie(op_downstream_map_, op_happens_before_) = GetDependency();

  instructions_ = &instructions;
  op_num_ = instructions_->size();

  ops_before_.assign(op_num_, {});
  ops_behind_.assign(op_num_, {});
  op_happens_before_->assign(op_num_, std::vector<bool>(op_num_, false));

  BuildDownstreamMap();
  VLOG(6) << "Finish BuildDownstreamMap";

  ShrinkDownstreamMap();
  VLOG(6) << "Finish ShrinkDownstreamMap";

  if (FLAGS_new_executor_sequential_run) {
    AddDependencyForSequentialRun();
  }

  AddDependencyForCoalesceTensorOp();

  if (FLAGS_add_dependency_for_communication_op) {
    AddDependencyForCommunicationOp();
    VLOG(6) << "Finish AddDependencyForSequentialRun";
  }

  AddDependencyForRandomOp();
  VLOG(6) << "Finish AddDependencyForRandomOp";

  AddDependencyForReadOp();
  VLOG(6) << "Finish AddDependencyForReadOp";

  VLOG(6) << "Finish build dependency";
  VLOG(8) << "downstream count: " << CountDownstreamMap(*op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(*op_downstream_map_);

  is_build_ = true;

  return *op_downstream_map_;
}

std::tuple<std::shared_ptr<std::map<size_t, std::set<size_t>>>,
           std::shared_ptr<std::vector<std::vector<bool>>>>
DependencyBuilder::GetDependency() const {
  return std::make_tuple(op_downstream_map_, op_happens_before_);
}

void DependencyBuilder::ShareDependencyFrom(const DependencyBuilder& src) {
  std::tie(op_downstream_map_, op_happens_before_) = src.GetDependency();
  is_build_ = true;
}

const std::map<size_t, std::set<size_t>>& DependencyBuilder::OpDownstreamMap()
    const {
  PADDLE_ENFORCE_EQ(
      is_build_,
      true,
      phi::errors::Unavailable(
          "DependencyBuilder is not yet built, call Build() firstly."));
  return *op_downstream_map_;
}

void DependencyBuilder::AddDependencyForCoalesceTensorOp() {
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        instructions_->at(op_idx).OpBase()->Type() == kCoalesceTensor) {
      VLOG(4) << "Add depend for " << kCoalesceTensor << " " << op_idx;
      auto fused_out = instructions_->at(op_idx).Outputs().at("FusedOutput")[0];
      auto outputs = instructions_->at(op_idx).Outputs().at("Output");

      auto is_read = [](const Instruction& inst, size_t var_id) -> bool {
        for (auto pair : inst.Inputs()) {
          for (size_t item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      auto is_write = [](const Instruction& inst, size_t var_id) -> bool {
        for (auto pair : inst.Outputs()) {
          for (size_t item : pair.second) {
            if (item == var_id) {
              return true;
            }
          }
        }
        return false;
      };

      // find first op that reads fused_out
      auto first_read_fused_out_op = ULLONG_MAX;
      for (auto j = op_idx + 1; j < op_num_; ++j) {
        if (is_read(instructions_->at(j), fused_out)) {
          first_read_fused_out_op = j;
          break;
        }
      }

      if (UNLIKELY(first_read_fused_out_op == ULLONG_MAX)) {
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
        if (j == target + 1 && IsCommunicationOp(instructions_->at(target)) &&
            IsCommunicationOp(instructions_->at(j))) {
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
  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (IsCommunicationOp(instructions_->at(op_idx))) {
      if (dependence_op_idx != ULLONG_MAX) {
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
  dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        instructions_->at(op_idx).OpBase()->Type() == kSyncComm) {
      dependence_op_idx = op_idx;
    } else {
      if (dependence_op_idx != ULLONG_MAX) {
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

  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        random_op_set.count(instructions_->at(op_idx).OpBase()->Type())) {
      if (dependence_op_idx != ULLONG_MAX) {
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
    auto it = op_downstream_map_->find(op_idx);
    if (it != op_downstream_map_->end()) {
      for (size_t downstream_op_idx : it->second) {
        is_startup_ops[downstream_op_idx] = false;
      }
    }
  }

  std::vector<size_t> read_ops;
  std::vector<size_t> startup_ops;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (instructions_->at(op_idx).OpBaseValid() &&
        instructions_->at(op_idx).OpBase()->Type() == "read") {
      read_ops.push_back(op_idx);
    }

    if (is_startup_ops[op_idx]) {
      startup_ops.push_back(op_idx);
    }
  }

  for (size_t read_op_idx : read_ops) {
    for (size_t downstream_op_idx : startup_ops) {
      if (read_op_idx != downstream_op_idx &&
          !OpHappensBefore(downstream_op_idx, read_op_idx)) {
        AddDownstreamOp(read_op_idx, downstream_op_idx);
      }
    }
  }
}

void DependencyBuilder::AddDependencyForSequentialRun() {
  size_t dependence_op_idx = ULLONG_MAX;
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    if (dependence_op_idx != ULLONG_MAX) {
      AddDownstreamOp(dependence_op_idx, op_idx);
    }
    dependence_op_idx = op_idx;
  }
}

void DependencyBuilder::AddDownstreamOp(size_t prior_op_idx,
                                        size_t posterior_op_idx) {
  PADDLE_ENFORCE_EQ(
      OpHappensBefore(posterior_op_idx, prior_op_idx),
      false,
      phi::errors::Unavailable(
          "Can not add dependency %d->%d because %d is run before %d",
          prior_op_idx,
          posterior_op_idx,
          posterior_op_idx,
          prior_op_idx));
  std::set<size_t>& downstream_ops = (*op_downstream_map_)[prior_op_idx];
  // NOTE(Ruibiao): Here the downstream map shrinking is best-effort, therefore
  // ShrinkDownstreamMap after BuildDownstreamMap is still helpful. For example,
  // a->c will not be shrinked in the following case: AddDownstreamOp(a, b) ->
  // AddDownstreamOp(a, c) -> AddDownstreamOp(b, c), it should be shrinked by
  // ShrinkDownstreamMap.
  for (size_t op_idx : downstream_ops) {
    if (OpHappensBefore(op_idx, posterior_op_idx)) {
      VLOG(7) << "Find dependencies " << prior_op_idx << "->" << op_idx << "->"
              << posterior_op_idx << ", skip adding " << prior_op_idx << "->"
              << posterior_op_idx;
      return;
    }
  }
  downstream_ops.insert(posterior_op_idx);

  std::vector<size_t> prior_of_prior = ops_before_[prior_op_idx];
  std::vector<size_t> posterior_of_posterior = ops_behind_[posterior_op_idx];

  auto update_op_happen_before = [this](size_t prior_op_idx,
                                        size_t posterior_op_idx) {
    if (!(*op_happens_before_)[prior_op_idx][posterior_op_idx]) {
      (*op_happens_before_)[prior_op_idx][posterior_op_idx] = true;
      ops_before_[posterior_op_idx].push_back(prior_op_idx);
      ops_behind_[prior_op_idx].push_back(posterior_op_idx);
    }
  };

  update_op_happen_before(prior_op_idx, posterior_op_idx);

  // All ops before prior-op are also before posterior-op
  for (size_t op_idx : prior_of_prior) {
    update_op_happen_before(op_idx, posterior_op_idx);
  }

  // All ops after posterior-op are also after prior-op
  for (size_t op_idx : posterior_of_posterior) {
    update_op_happen_before(prior_op_idx, op_idx);
  }

  VLOG(8) << prior_op_idx << "->" << posterior_op_idx;
  VLOG(8) << "Add dependency from "
          << "prior_op_idx(" << prior_op_idx << ") to "
          << "posterior_op_idx(" << posterior_op_idx << ")";
}

void DependencyBuilder::BuildDownstreamMap() {
  auto var2min_rw_op =
      std::map<size_t, std::list<size_t>>();  // # map from variable id to read
                                              //  write op id.
  auto var2recent_write_op =
      std::map<size_t, size_t>();  // # map from variable to recent write op.
  auto op2dependences =
      std::map<size_t,
               std::set<size_t>>();  // # map from op to the dependence list,
                                     //  op must run after the dependence.
  std::set<size_t>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    op2dependences[op_idx] = std::set<size_t>();
  }

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
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
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
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         instructions_->at(op_idx).Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          UpdateVarMinRwOp(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // convert op2dependences to downstream_map directly. op2dependences is op ->
  // it's dependences, we want to get op -> [next ops] map, where ops is the
  // next instruction of op. The size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  for (auto& item : op2dependences) {
    size_t op = item.first;
    for (auto dep_op : item.second) {
      AddDownstreamOp(dep_op, op);
    }
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
    if (op_downstream_map_->find(i) == op_downstream_map_->end()) {
      continue;
    }

    std::set<size_t> minumum_nexts;
    for (size_t item : op_downstream_map_->at(i)) {
      bool not_after_any = true;
      // find the op that is not executed after any
      for (size_t other_item : op_downstream_map_->at(i)) {
        if (OpHappensBefore(other_item, item)) {
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
    // NOTE(Ruibiao): op_happens_before will not be changed when shrink
    // dowstream map
    (*op_downstream_map_)[i] = minumum_nexts;
  }
  VLOG(8) << "Finish shrink downstream map";
  VLOG(8) << "downstream count: " << CountDownstreamMap(*op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(*op_downstream_map_);
}

void DependencyBuilder::UpdateVarMinRwOp(
    const std::map<size_t, std::set<size_t>>& op2dependences,
    std::map<size_t, std::list<size_t>>* var2min_rw_op,
    size_t cur_op,
    size_t rw_var) {
  // rw_var is inputs or outputs of cur_op
  // this function update the var2min_rw_op set .
  if (var2min_rw_op->find(rw_var) == var2min_rw_op->end()) {
    (*var2min_rw_op)[rw_var] = std::list<size_t>();
  }
  for (auto dep_op : op2dependences.at(cur_op)) {
    var2min_rw_op->at(rw_var).remove(dep_op);
  }
  var2min_rw_op->at(rw_var).push_back(cur_op);
}

/// ======================== ///
///        For new ir        ///
/// ======================== ///
NewIrDependencyBuilder::NewIrDependencyBuilder() {
  is_build_ = false;
  op_downstream_map_ = std::make_shared<std::map<size_t, std::set<size_t>>>();
  op_happens_before_ = std::make_shared<std::vector<std::vector<bool>>>();
}

const std::map<size_t, std::set<size_t>>& NewIrDependencyBuilder::Build(
    std::vector<paddle::framework::InstructionBase*> instructions) {
  if (is_build_) {
    return *op_downstream_map_;
  }

  std::tie(op_downstream_map_, op_happens_before_) = GetDependency();

  instructions_ = instructions;
  op_num_ = instructions_.size();

  ops_before_.assign(op_num_, {});
  ops_behind_.assign(op_num_, {});
  op_happens_before_->assign(op_num_, std::vector<bool>(op_num_, false));

  BuildDownstreamMap();
  VLOG(6) << "Finish BuildDownstreamMap";

  ShrinkDownstreamMap();
  VLOG(6) << "Finish ShrinkDownstreamMap";

  if (FLAGS_new_executor_sequential_run) {
    AddDependencyForSequentialRun();
  }

  // TODO(zhangbo): Add dependency for special op ？

  VLOG(6) << "Finish build dependency";
  VLOG(8) << "downstream count: " << CountDownstreamMap(*op_downstream_map_);
  VLOG(8) << "downstream_map: " << std::endl
          << StringizeDownstreamMap(*op_downstream_map_);

  is_build_ = true;

  return *op_downstream_map_;
}

void NewIrDependencyBuilder::BuildDownstreamMap() {
  auto var2min_rw_op =
      std::map<size_t, std::list<size_t>>();  // # map from variable id to read
                                              //  write op id.
  auto var2recent_write_op =
      std::map<size_t, size_t>();  // # map from variable to recent write op.

  auto op2dependences =
      std::map<size_t,
               std::set<size_t>>();  //# map from op to the dependence list,
                                     // op must run after the dependence.
  std::set<size_t>
      remove_duplicate;  // remove the duplicate between inputs and outputs

  // reserve
  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    op2dependences[op_idx] = std::set<size_t>();
  }

  for (size_t op_idx = 0; op_idx < op_num_; ++op_idx) {
    remove_duplicate.clear();
    // step1: update the op2dependences structure
    for (auto& item :
         instructions_.at(op_idx)->Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (var2recent_write_op.count(var))
          op2dependences[op_idx].insert(var2recent_write_op[var]);
      }
    }

    for (auto& item :
         instructions_.at(op_idx)->Outputs()) {  // for all write vars
      for (auto var : item.second) {
        if (var2min_rw_op.count(var)) {
          for (auto dep_op : var2min_rw_op[var]) {
            op2dependences[op_idx].insert(dep_op);
          }
        }
      }
    }

    // step2: update 2 var2xxxx data structure
    for (auto& item :
         instructions_.at(op_idx)->Outputs()) {  // for all write vars
      for (auto var : item.second) {
        var2recent_write_op[var] = op_idx;
        var2min_rw_op[var] = {static_cast<size_t>(op_idx)};
        remove_duplicate.insert(var);
      }
    }

    for (auto& item :
         instructions_.at(op_idx)->Inputs()) {  // for all inputs(read only)
      for (auto var : item.second) {
        if (remove_duplicate.count(var) ==
            0) {  // var in input list and in output list, so remove it.
          UpdateVarMinRwOp(op2dependences, &var2min_rw_op, op_idx, var);
        }
      }
    }
  }

  // convert op2dependences to downstream_map directly. op2dependences is op ->
  // it's dependences, we want to get op -> [next ops] map, where ops is the
  // next instruction of op. The size of downstream != size of op2dependences
  // since there are some ops that have no downstream-op.
  for (auto& item : op2dependences) {
    size_t op = item.first;
    for (auto dep_op : item.second) {
      AddDownstreamOp(dep_op, op);
    }
  }
}

void NewIrDependencyBuilder::ShareDependencyFrom(
    const NewIrDependencyBuilder& src) {
  std::tie(op_downstream_map_, op_happens_before_) = src.GetDependency();
  is_build_ = true;
}

}  // namespace interpreter
}  // namespace framework
}  // namespace paddle
