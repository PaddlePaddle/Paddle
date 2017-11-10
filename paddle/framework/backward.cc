/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License. */

#include "paddle/framework/backward.h"
#include "paddle/operators/net_op.h"

#include <deque>
#include <list>
#include <memory>
#include <unordered_set>

#include "paddle/framework/block_desc.h"
#include "paddle/framework/op_registry.h"
#include "paddle/operators/dynamic_recurrent_op.h"
#include "paddle/operators/net_op.h"

namespace paddle {
namespace framework {

static inline std::unique_ptr<OperatorBase> CreateGradOp(
    const OperatorBase& op, const std::unordered_set<std::string>& no_grad_set,
    std::unordered_map<std::string, std::string>* grad_to_var) {
  OpDescBind op_desc;
  op_desc.SetInputMap(op.Inputs());
  op_desc.SetOutputMap(op.Outputs());
  op_desc.SetType(op.Type());
  op_desc.SetAttrMap(op.Attrs());
  auto& info = OpInfoMap::Instance().Get(op.Type());
  auto grad_descs = info.GradOpMaker()(op_desc, no_grad_set, grad_to_var, {});
  std::vector<std::unique_ptr<OperatorBase>> grad_ops;
  grad_ops.reserve(grad_descs.size());
  std::transform(grad_descs.begin(), grad_descs.end(),
                 std::back_inserter(grad_ops),
                 [](const std::unique_ptr<OpDescBind>& grad_desc) {
                   return OpRegistry::CreateOp(*grad_desc);
                 });
  PADDLE_ENFORCE(!grad_ops.empty());
  if (grad_ops.size() == 1) {
    return std::move(grad_ops[0]);
  } else {
    auto net_op = new operators::NetOp();
    for (auto& grad_op : grad_ops) {
      net_op->AppendOp(std::move(grad_op));
    }
    net_op->CompleteAddOp();
    return std::unique_ptr<OperatorBase>(net_op);
  }
}

template <typename Map, typename T>
static void ForEachVarName(const Map& names, T callback) {
  for (auto& name : names) {
    for (auto& n : name.second) {
      if (callback(n)) return;
    }
  }
}

// return whether all the names + suffixes in the set
static bool AllInSet(
    const std::map<std::string, std::vector<std::string>>& names,
    const std::string& suffix, const std::unordered_set<std::string>& set) {
  bool all_in_set = true;
  ForEachVarName(names, [&all_in_set, &set, &suffix](const std::string& n) {
    all_in_set = set.find(n + suffix) != set.end();
    return !all_in_set;
  });
  return all_in_set;
}

static std::unique_ptr<OperatorBase> NOP() {
  auto net_op = new operators::NetOp();
  net_op->SetType("@NOP@");
  net_op->CompleteAddOp();
  return std::unique_ptr<OperatorBase>(net_op);
}

//  Get backward operator from a forward operator, a recursive implementation.
//
//  no_grad_names the gradient variable names without gradient calculating.
//
//  uniq_id is a unique index used inside recursively calling
//  BackwardRecursive. use `uid = uniq_id++;` to get the unique index, and
//  pass `uniq_id` through recursive calling.
//
//  returns The backward operator. In a simple situation, it may be a simple
//  operator, in a complex situation, it maybe a NetOp.
//
//  See Backward.h for details
static std::unique_ptr<OperatorBase> BackwardRecursive(
    const OperatorBase& forwardOp,
    std::unordered_set<std::string>& no_grad_names,
    std::unordered_map<std::string, std::string>* grad_to_var,
    size_t& uniq_id) {
  //  If all input gradients of forwarding operator do not need to calculate,
  //  just return an NOP. Not return null ptr because NOP does not take
  //  too much time for calculation, but it is useful for simplifying logic.
  if (AllInSet(forwardOp.Inputs() /*names*/, kGradVarSuffix /*suffix*/,
               no_grad_names /*set*/)) {
    return NOP();
  }

  //  All output gradients of forwarding operator do not need to calculate.
  //  Then all input gradients cannot be computed at all, and we put them into
  //  `no_grad_names` set. Return an NOP.
  if (AllInSet(forwardOp.Outputs() /*names*/, kGradVarSuffix /*suffix*/,
               no_grad_names /*set*/)) {
    ForEachVarName(forwardOp.Inputs(),
                   [&no_grad_names](const std::string& name) -> bool {
                     no_grad_names.insert(GradVarName(name));
                     return false;
                   });
    return NOP();
  }

  // Returned gradient network
  auto net = std::unique_ptr<operators::NetOp>(new operators::NetOp());

  if (forwardOp.IsNetOp()) {
    // Because forwardOp is a net op, it can static_cast.
    auto& forwardNet = static_cast<const operators::NetOp&>(forwardOp);

    // Map from output gradient variable name to operator's indices in
    // backward net's ops_. That operator generates that variable.
    std::unordered_map<std::string, std::vector<size_t>> dup_output_ops;

    size_t local_op_id = 0;
    // reversely travel forwardNet and collect all duplicate outputs.
    for (auto it = forwardNet.ops_.rbegin(); it != forwardNet.ops_.rend();
         ++it, ++local_op_id) {
      auto& fwd = *it;
      auto bwd = BackwardRecursive(*fwd, no_grad_names, grad_to_var, uniq_id);
      ForEachVarName(bwd->Outputs(),
                     [&dup_output_ops, local_op_id](const std::string& out) {
                       dup_output_ops[out].emplace_back(local_op_id);
                       return false;
                     });
      net->AppendOp(std::move(bwd));
    }
    // Get unique ID for this method.
    auto uid = uniq_id++;
    // TODO(dzh): more comment
    // multiple operators which have the same output (y for example) may
    // overwrite the same y variable when backward, special operations are token
    // to handle this case. For each duplicate output, rename it to an alias
    // (original name with a offset), append an `add` op for its operator,
    // and finally sum all the alias variable to the final output variable y.
    using Pos = std::pair<size_t, std::unique_ptr<OperatorBase>>;
    std::list<Pos> insert_position;
    for (auto& dup_output_op : dup_output_ops) {
      const std::string& name = dup_output_op.first;
      // duplicate @Empty@ don't need to be added
      if (name == kEmptyVarName) continue;

      auto& dup_op = dup_output_op.second;
      // no duplicate output
      if (dup_op.size() == 1) continue;

      // process the duplicate outputs
      std::vector<std::string> dup_outputs;
      for (size_t i = 0; i < dup_op.size(); ++i) {
        // rename each duplicate output to an alias
        auto op_offset = dup_op[i];
        dup_outputs.push_back(name + "@RENAME@" + std::to_string(uid) + "@" +
                              std::to_string(i));
        net->ops_[op_offset]->Rename(name, dup_outputs.back());
      }
      // collect all the offset for each alias,
      // insert a sum operator to add all aliases to output
      insert_position.push_back(
          {dup_op.back(), OpRegistry::CreateOp("sum", {{"X", dup_outputs}},
                                               {{"Out", {name}}}, {})});
    }

    // make sure the inserted `sum` ops follow the BFS order.
    insert_position.sort(
        [](const Pos& l, const Pos& r) { return l.first > r.first; });

    for (auto& pos : insert_position) {
      net->InsertOp(pos.first + 1, std::move(pos.second));
    }
  } else {
    std::unique_ptr<OperatorBase> grad_op(
        CreateGradOp(forwardOp, no_grad_names, grad_to_var));

    ForEachVarName(grad_op->Inputs(), [&no_grad_names, &net, &grad_op](
                                          const std::string& grad_input) {
      if (no_grad_names.count(grad_input)) {
        // +1 for \0
        std::string prefix = grad_input.substr(
            0, grad_input.size() - sizeof(kGradVarSuffix) / sizeof(char) + 1);
        grad_op->Rename(grad_input, prefix + kZeroVarSuffix);

        // If part of input gradient of that operator is not calculated, fill
        // zero variables to that input gradient.
        net->AppendOp(OpRegistry::CreateOp("fill_zeros_like", {{"X", {prefix}}},
                                           {{"Y", {grad_input}}}, {}));
      }
      return false;
    });

    ForEachVarName(grad_op->Outputs(),
                   [&no_grad_names, &grad_op](const std::string& grad_output) {
                     if (no_grad_names.count(grad_output)) {
                       grad_op->Rename(grad_output, kEmptyVarName);
                     }
                     return false;
                   });

    // process recurrent gradient op as a special operator.
    if (forwardOp.Type() == "dynamic_recurrent") {
      // NOTE clean up cycle call somewhere (RNN's stepnet constains itself),
      // or this will result in infinite loop.
      const auto& rnnop =
          *static_cast<const operators::DynamicRecurrentOp*>(&forwardOp);
      auto rnn_grad_op =
          static_cast<operators::DynamicRecurrentGradientOp*>(grad_op.get());
      const auto& stepnet_op =
          *static_cast<const OperatorBase*>(&rnnop.rnn.GetStepUnit());
      // create stepnet's gradient op
      rnn_grad_op->rnn.SetStepUnit(
          BackwardRecursive(stepnet_op, no_grad_names, grad_to_var, uniq_id));
    }

    if (net->ops_.empty()) {  // Current no aux op is added to network
      return grad_op;
    }
    net->AppendOp(std::move(grad_op));
  }
  net->SetType("@GENERATED_BACKWARD@");
  net->CompleteAddOp();
  return std::unique_ptr<OperatorBase>(
      static_cast<OperatorBase*>(net.release()));
}

// See header for comments
std::unique_ptr<OperatorBase> Backward(
    const OperatorBase& forwardOp,
    const std::unordered_set<std::string>& no_grad_vars) {
  std::unordered_set<std::string> no_grad_names;
  no_grad_names.reserve(no_grad_vars.size() + 1);

  no_grad_names.insert(std::string(kEmptyVarName) + kGradVarSuffix);

  for (auto& name : no_grad_vars) {
    no_grad_names.insert(name + kGradVarSuffix);
  }
  size_t uid = 0;
  std::unordered_map<std::string, std::string> grad_to_var;
  return BackwardRecursive(forwardOp, no_grad_names, &grad_to_var, uid);
}

// ====================================  //

static bool AllGradInSet(const std::vector<std::string>& names,
                         const std::unordered_set<std::string>& set) {
  for (const std::string& name : names) {
    if (!set.count(GradVarName(name))) {
      return false;
    }
  }
  return true;
}

static std::string FwdName(const std::string& grad_name) {
  auto pos = grad_name.find("@GRAD");
  if (pos == std::string::npos) {
    return "";
  } else {
    return grad_name.substr(0, pos);
  }
}

static void CreateGradVarInBlock(
    size_t grad_op_start_index,
    const std::unordered_map<std::string, std::string>& param_name_map,
    BlockDescBind* block_desc,
    std::unordered_map<std::string, GradVarInfo>* grad_var_record) {
  auto ops = block_desc->AllOps();
  for (size_t op_index = grad_op_start_index; op_index < ops.size();
       ++op_index) {
    bool need_infer_shape = false;
    std::unordered_set<std::string> new_vars;
    ForEachVarName(ops[op_index]->Outputs(),
                   [&](const std::string& grad_var_name) {
                     if (block_desc->HasVar(grad_var_name)) {
                       return false;
                     }
                     need_infer_shape = true;
                     auto var = block_desc->Var(grad_var_name);
                     new_vars.insert(var->Name());
                     auto it = param_name_map.find(grad_var_name);
                     if (it == param_name_map.end()) {
                       return false;
                     }
                     auto param_var_name = it->second;
                     auto& grad_record = (*grad_var_record)[param_var_name];
                     grad_record.name_ = grad_var_name;
                     grad_record.block_idx_ = block_desc->ID();
                     grad_record.op_idx_ = static_cast<int>(op_index);
                     return false; /* not break */
                   });
    if (need_infer_shape) {
      ops[op_index]->InferVarType(block_desc);
      for (auto& arg : ops[op_index]->OutputArgumentNames()) {
        if (new_vars.find(arg) == new_vars.end()) {
          continue;
        }
        auto pname = FwdName(arg);
        auto* param = block_desc->FindVarRecursive(pname);
        auto* grad = block_desc->FindVar(arg);
        if (param == nullptr) {
          grad->SetDataType(DataType::FP32);
        } else {
          grad->SetDataType(param->GetDataType());
        }
      }
      ops[op_index]->InferShape(*block_desc);
    }
  }
}

std::vector<std::unique_ptr<OpDescBind>> MakeOpGrad(
    const OpDescBind* op_desc, std::unordered_set<std::string>* no_grad_vars,
    std::unordered_map<std::string, std::string>* grad_to_var,
    const std::vector<BlockDescBind*>& grad_block =
        std::vector<BlockDescBind*>()) {
  std::vector<std::unique_ptr<OpDescBind>> grad_op_descs;
  // All input gradients of forwarding operator do not need to calculate.
  const std::vector<std::string>& inputs = op_desc->InputArgumentNames();
  if (AllGradInSet(inputs, *no_grad_vars)) {
    return grad_op_descs;  // empty vector
  }
  // All output gradients of forwarding operator do not need to calculate.
  const std::vector<std::string>& outputs = op_desc->OutputArgumentNames();
  if (AllGradInSet(outputs, *no_grad_vars)) {
    for (const std::string& name : inputs) {
      no_grad_vars->insert(GradVarName(name));
    }
    return grad_op_descs;  // empty vector
  }

  grad_op_descs =
      OpInfoMap::Instance()
          .Get(op_desc->Type())
          .GradOpMaker()(*op_desc, *no_grad_vars, grad_to_var, grad_block);

  std::list<std::unique_ptr<OpDescBind>> pending_fill_zeros_ops;
  for (auto& desc : grad_op_descs) {
    for (const std::string& in_name : desc->InputArgumentNames()) {
      if (no_grad_vars->count(in_name)) {
        std::string prefix = in_name.substr(
            0, in_name.size() - sizeof(kGradVarSuffix) / sizeof(char) + 1);
        std::string new_name = prefix + kZeroVarSuffix;
        desc->Rename(in_name, new_name);
        std::unique_ptr<OpDescBind> fill_zeros_op(new OpDescBind(
            "fill_zeros_like", {{"X", {prefix}}}, {{"Y", {new_name}}}, {}));
        pending_fill_zeros_ops.push_back(std::move(fill_zeros_op));
      }
    }
  }

  for (auto& p : pending_fill_zeros_ops) {
    grad_op_descs.insert(grad_op_descs.begin(), std::move(p));
  }
  return grad_op_descs;
}

std::vector<std::unique_ptr<OpDescBind>> MakeBlockBackward(
    ProgramDescBind& program_desc, int block_idx,
    std::unordered_set<std::string>* no_grad_vars,
    std::unordered_map<std::string, std::string>* grad_to_var) {
  BlockDescBind* cur_block = program_desc.MutableBlock(block_idx);
  std::vector<OpDescBind*> op_descs = cur_block->AllOps();
  std::unordered_map<std::string, std::vector<size_t>> dup_out_ops;
  size_t grad_desc_idx = 0;
  std::vector<std::unique_ptr<OpDescBind>> backward_descs;

  for (auto it = op_descs.rbegin(); it != op_descs.rend(); ++it) {
    std::vector<std::unique_ptr<OpDescBind>> op_grads;

    if ((*it)->Type() == "recurrent") {
      int step_block_idx = (*it)->GetBlockAttr("step_block");
      auto backward_block_op_descs = MakeBlockBackward(
          program_desc, step_block_idx, no_grad_vars, grad_to_var);
      BlockDescBind* backward_block =
          program_desc.AppendBlock(*program_desc.MutableBlock(step_block_idx));
      for (auto& ptr : backward_block_op_descs) {
        backward_block->AppendAllocatedOp(std::move(ptr));
      }
      op_grads = MakeOpGrad(*it, no_grad_vars, grad_to_var, {backward_block});
    } else {
      op_grads = MakeOpGrad(*it, no_grad_vars, grad_to_var);
    }

    for (const auto& desc : op_grads) {
      for (const std::string& out_name : desc->OutputArgumentNames()) {
        if (out_name.find("@GRAD") == std::string::npos) {
          // Not all outputs of a backward operator is a gradient. Only gradient
          // need to be sum. Skip variables are not gradient.
          continue;
        }
        dup_out_ops[out_name].emplace_back(grad_desc_idx);
      }
      ++grad_desc_idx;
    }
    std::transform(
        op_grads.begin(), op_grads.end(), std::back_inserter(backward_descs),
        [](std::unique_ptr<OpDescBind>& ptr) { return std::move(ptr); });
  }
  // Check whether some variables are written more than once
  std::list<std::pair<size_t, std::unique_ptr<OpDescBind>>> pending_sum_ops;
  for (const auto& dup : dup_out_ops) {
    const std::string& out_name = dup.first;
    const std::vector<size_t> dup_op = dup.second;
    if (out_name != kEmptyVarName && dup_op.size() > 1) {
      std::vector<std::string> sum_op_inputs;
      for (size_t i = 0; i < dup_op.size(); ++i) {
        std::string new_name = out_name + "@RENAME@" + std::to_string(i);
        backward_descs[dup_op[i]]->Rename(out_name, new_name);
        sum_op_inputs.emplace_back(new_name);
      }
      std::unique_ptr<OpDescBind> sum_op(new OpDescBind(
          "sum", {{"X", sum_op_inputs}}, {{"Out", {out_name}}}, {}));
      pending_sum_ops.push_back({dup_op.back(), std::move(sum_op)});
    }
  }
  pending_sum_ops.sort(
      [](const std::pair<size_t, std::unique_ptr<OpDescBind>>& a,
         const std::pair<size_t, std::unique_ptr<OpDescBind>>& b) {
        return a.first > b.first;
      });
  for (auto& p : pending_sum_ops) {
    backward_descs.insert(backward_descs.begin() + p.first + 1,
                          std::move(p.second));
  }

  return backward_descs;
}

ParamGradInfoMap AppendBackward(
    ProgramDescBind& program_desc, const VarDescBind& target,
    const std::unordered_set<std::string>& no_grad_vars) {
  std::unordered_set<std::string> no_grad_var_names;
  no_grad_var_names.reserve(no_grad_vars.size() + 1);
  no_grad_var_names.insert(std::string(kEmptyVarName) + kGradVarSuffix);
  for (auto& name : no_grad_vars) {
    no_grad_var_names.insert(GradVarName(name));
  }

  const int root_block_idx = 0;
  auto root_block = program_desc.MutableBlock(root_block_idx);

  // insert fill one op for target
  // TODO(qiao) add some check to the target.
  std::string fill_one_op_out = GradVarName(target.Name());
  std::vector<int64_t> target_shape_desc = target.Shape();
  std::vector<int> target_shape;
  std::transform(target_shape_desc.begin(), target_shape_desc.end(),
                 std::back_inserter(target_shape),
                 [](int64_t dim) { return static_cast<int>(dim); });
  VLOG(3) << "backward from loss=" << target.Name()
          << " data_type=" << target.GetDataType();
  std::unique_ptr<OpDescBind> fill_one_op(
      new OpDescBind("fill_constant", {}, {{"Out", {fill_one_op_out}}},
                     {{"shape", target_shape},
                      {"value", static_cast<float>(1.0)},
                      {"data_type", target.GetDataType()}}));
  // infer var type of fill_one_op
  fill_one_op->InferVarType(root_block);

  root_block->AppendAllocatedOp(std::move(fill_one_op));
  size_t forward_op_num = root_block->OpSize();
  size_t forward_block_num = program_desc.Size();

  // Insert backward operators
  std::unordered_map<std::string, std::string> grad_to_var;
  auto backward_op_descs = MakeBlockBackward(program_desc, root_block_idx,
                                             &no_grad_var_names, &grad_to_var);

  for (auto& ptr : backward_op_descs) {
    root_block->AppendAllocatedOp(std::move(ptr));
  }
  // Create Variable

  // Create target gradient variable
  std::unordered_map<std::string, GradVarInfo> retv;

  auto var = root_block->Var(fill_one_op_out);
  var->SetDataType(target.GetDataType());
  var->SetShape(target.Shape());
  auto& target_grad = retv[target.Name()];
  target_grad.name_ = fill_one_op_out;
  target_grad.block_idx_ = root_block_idx;
  target_grad.op_idx_ = static_cast<int>(forward_op_num);

  // create grad_var for all blocks in this program
  CreateGradVarInBlock(forward_op_num, grad_to_var, root_block, &retv);
  for (size_t block_index = forward_block_num;
       block_index < program_desc.Size(); ++block_index) {
    CreateGradVarInBlock(0, grad_to_var, program_desc.MutableBlock(block_index),
                         &retv);
  }
  return retv;
}

}  // namespace framework
}  // namespace paddle
