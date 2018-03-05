/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include "paddle/fluid/framework/executor.h"

#ifdef FLUID_PROFILER
#include <x86intrin.h>
#include <fstream>
#include <iostream>
#include <set>
#include <sstream>
#endif

#include "gflags/gflags.h"
#include "paddle/fluid/framework/channel.h"
#include "paddle/fluid/framework/feed_fetch_method.h"
#include "paddle/fluid/framework/feed_fetch_type.h"
#include "paddle/fluid/framework/lod_rank_table.h"
#include "paddle/fluid/framework/lod_tensor_array.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/reader.h"
#include "paddle/fluid/platform/place.h"

DECLARE_bool(benchmark);
DEFINE_bool(check_nan_inf, false,
            "Checking whether operator produce NAN/INF or not. It will be "
            "extremely slow so please use this flag wisely.");

namespace paddle {
namespace framework {

#ifdef FLUID_PROFILER
static Perf watcher;

Perf::Perf(void) : m_tsc(0) {
  auto process = [](std::string& line, uint64_t* out_tsc) -> bool {
    if (line.find("model name") == std::string::npos) {
      return false;
    }

    // format is model name : <CPU model>
    // we want to extract CPU model only
    auto it = line.find(":");
    auto model_name = line.substr(it);
    std::cout << "MODEL: " << model_name << std::endl;
    it = line.find("@");
    float tsc_clock = stof(line.substr(it + 1)) * 1000000000.0f;
    std::cout << "TSC: " << tsc_clock << std::endl;
    *out_tsc = tsc_clock;
    return true;
  };

  std::ifstream infile("/proc/cpuinfo");
  std::string line;
  bool run = true;
  while (run && std::getline(infile, line)) {
    run = !process(line, &m_tsc);
  }
}

void Perf::addEntry(const std::string& name, uint64_t diff) {
  // Ignore block operators eg. while, while_grad
  std::vector<std::string> exclude_list;
  exclude_list.emplace_back("while");
  exclude_list.emplace_back("while_grad");

  for (auto& exclude_item : exclude_list) {
    if (name.compare(exclude_item) == 0) return;
  }

  auto it = m_my_map.find(name);
  if (it != m_my_map.end()) {
    float tmpVal = m_my_map[name].avg_perf;
    m_my_map[name].avg_perf =
        tmpVal + ((float)(diff)-tmpVal) / (float)m_my_map[name].num_ops;
    m_my_map[name].total_time += diff;
    m_my_map[name].num_ops += 1;
  } else {
    m_my_map[name].avg_perf = (float)(diff);
    m_my_map[name].total_time = diff;
    m_my_map[name].num_ops = 1;
    m_my_map[name].op_type = name;
  }
}

Perf::~Perf() {
  std::vector<entity> sorted_ops;
  float total_average_execution_time = 0.0f;
  for (auto& item : m_my_map) {
    const float total_op_execution_time =
        ((item.second.total_time) * 1000.0f) / (float)m_tsc;
    const float average_execution_time =
        ((item.second.avg_perf) * 1000.0f) / (float)m_tsc;
    total_average_execution_time += total_op_execution_time;
    sorted_ops.push_back({item.first, item.second.num_ops,
                          average_execution_time,
                          (uint64_t)total_op_execution_time});
  }

  // Sort in descending order
  std::sort(sorted_ops.begin(), sorted_ops.end(),
            [](const entity& a, const entity& b) -> bool {
              return a.total_time > b.total_time;
            });

  // Print sorted operators
  for (auto& op_item : sorted_ops) {
    std::cout << "Operator[" << op_item.op_type << "]: "
              << "totalExec: " << op_item.total_time << " ms"
              << " Ratio[totalExec/total_time]: "
              << 100.0f * op_item.total_time / total_average_execution_time
              << " %"
              << " aveExec[ms]: " << op_item.avg_perf
              << " Num Execs: " << op_item.num_ops << std::endl;
  }

  std::cout << "Total average execution time: " << total_average_execution_time
            << "ms" << std::endl;
}
#endif

Executor::Executor(const platform::Place& place) : place_(place) {}

static void CreateTensor(Variable* var, proto::VarType::Type var_type) {
  if (var_type == proto::VarType::LOD_TENSOR) {
    var->GetMutable<LoDTensor>();
  } else if (var_type == proto::VarType::SELECTED_ROWS) {
    var->GetMutable<SelectedRows>();
  } else if (var_type == proto::VarType::FEED_MINIBATCH) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::FETCH_LIST) {
    var->GetMutable<FeedFetchList>();
  } else if (var_type == proto::VarType::STEP_SCOPES) {
    var->GetMutable<std::vector<framework::Scope>>();
  } else if (var_type == proto::VarType::LOD_RANK_TABLE) {
    var->GetMutable<LoDRankTable>();
  } else if (var_type == proto::VarType::LOD_TENSOR_ARRAY) {
    var->GetMutable<LoDTensorArray>();
  } else if (var_type == proto::VarType::PLACE_LIST) {
    var->GetMutable<platform::PlaceList>();
  } else if (var_type == proto::VarType::READER) {
    var->GetMutable<ReaderHolder>();
  } else if (var_type == proto::VarType::CHANNEL) {
    var->GetMutable<ChannelHolder>();
  } else if (var_type == proto::VarType::RAW) {
    // GetMutable will be called in operator
  } else {
    PADDLE_THROW(
        "Variable type %d is not in "
        "[LOD_TENSOR, SELECTED_ROWS, FEED_MINIBATCH, FETCH_LIST, "
        "LOD_RANK_TABLE, PLACE_LIST, READER, CHANNEL, RAW]",
        var_type);
  }
}

static void CheckTensorNANOrInf(const std::string& name,
                                const framework::Tensor& tensor) {
  if (tensor.memory_size() == 0) {
    return;
  }
  if (tensor.type().hash_code() != typeid(float).hash_code() &&
      tensor.type().hash_code() != typeid(double).hash_code()) {
    return;
  }
  PADDLE_ENFORCE(!framework::TensorContainsInf(tensor),
                 "Tensor %s contains Inf", name);
  PADDLE_ENFORCE(!framework::TensorContainsNAN(tensor),
                 "Tensor %s contains NAN", name);
}

void Executor::Run(const ProgramDesc& pdesc, Scope* scope, int block_id,
                   bool create_local_scope, bool create_vars) {
  // TODO(tonyyang-svail):
  //    - only runs on the first device (i.e. no interdevice communication)
  //    - will change to use multiple blocks for RNN op and Cond Op
  PADDLE_ENFORCE_LT(static_cast<size_t>(block_id), pdesc.Size());
  auto& block = pdesc.Block(block_id);

  Scope* local_scope = scope;
  if (create_vars) {
    if (create_local_scope) {
      local_scope = &scope->NewScope();
      for (auto& var : block.AllVars()) {
        if (var->Name() == framework::kEmptyVarName) {
          continue;
        }

        if (var->Persistable()) {
          auto* ptr = scope->Var(var->Name());
          CreateTensor(ptr, var->GetType());
          VLOG(3) << "Create Variable " << var->Name()
                  << " global, which pointer is " << ptr;
        } else {
          auto* ptr = local_scope->Var(var->Name());
          CreateTensor(ptr, var->GetType());
          VLOG(3) << "Create Variable " << var->Name()
                  << " locally, which pointer is " << ptr;
        }
      }
    } else {
      for (auto& var : block.AllVars()) {
        auto* ptr = local_scope->Var(var->Name());
        CreateTensor(ptr, var->GetType());
        VLOG(3) << "Create variable " << var->Name() << ", which pointer is "
                << ptr;
      }
    }  // if (create_local_scope)
  }    // if (create_vars)

  for (auto& op_desc : block.AllOps()) {
    auto op = paddle::framework::OpRegistry::CreateOp(*op_desc);

    VLOG(3) << place_ << " " << op->DebugStringEx(local_scope);
#ifdef FLUID_PROFILER
    uint64_t t0 = __rdtsc();
#endif
    op->Run(*local_scope, place_);
#ifdef FLUID_PROFILER
    uint64_t t1 = __rdtsc();
    watcher.addEntry(op->Type(), t1 - t0);
#endif

    if (FLAGS_benchmark) {
      VLOG(2) << "Memory used after operator " + op->Type() + " running: "
              << memory::memory_usage(place_);
    }
    if (FLAGS_check_nan_inf) {
      for (auto& vname : op->OutputVars(true)) {
        auto* var = local_scope->FindVar(vname);
        if (var == nullptr) continue;
        if (var->IsType<framework::LoDTensor>()) {
          CheckTensorNANOrInf(vname, var->Get<framework::LoDTensor>());
        }
      }
    }
  }
  if (create_vars && create_local_scope) {
    scope->DeleteScope(local_scope);
  }
  if (FLAGS_benchmark) {
    VLOG(2) << "-------------------------------------------------------";
    VLOG(2) << "Memory used after deleting local scope: "
            << memory::memory_usage(place_);
    VLOG(2) << "-------------------------------------------------------";
  }
}

// Check whether the block already has feed operators and feed_holder.
// Return false if the block does not have any feed operators.
// If some feed operators have been prepended to the block, check that
// the info contained in these feed operators matches the feed_targets
// and feed_holder_name. Raise exception when any mismatch is found.
// Return true if the block has feed operators and holder of matching info.
static bool has_feed_operators(
    BlockDesc* block, std::map<std::string, const LoDTensor*>& feed_targets,
    const std::string& feed_holder_name) {
  size_t feed_count = 0;
  for (auto* op : block->AllOps()) {
    if (op->Type() == kFeedOpType) {
      feed_count++;
      PADDLE_ENFORCE_EQ(op->Input("X")[0], feed_holder_name,
                        "Input to feed op should be '%s'", feed_holder_name);
      std::string feed_target_name = op->Output("Out")[0];
      PADDLE_ENFORCE(
          feed_targets.find(feed_target_name) != feed_targets.end(),
          "Feed operator output name '%s' cannot be found in 'feed_targets'",
          feed_target_name);
    }
  }

  if (feed_count > 0) {
    PADDLE_ENFORCE_EQ(
        feed_count, feed_targets.size(),
        "The number of feed operators should match 'feed_targets'");

    // When feed operator are present, so should be feed_holder
    auto var = block->FindVar(feed_holder_name);
    PADDLE_ENFORCE_NOT_NULL(var, "Block should already have a '%s' variable",
                            feed_holder_name);
    PADDLE_ENFORCE_EQ(var->GetType(), proto::VarType::FEED_MINIBATCH,
                      "'%s' variable should be 'FEED_MINIBATCH' type",
                      feed_holder_name);
  }

  return feed_count > 0;
}

// Check whether the block already has fetch operators and fetch_holder.
// Return false if the block does not have any fetch operators.
// If some fetch operators have been appended to the block, check that
// the info contained in these fetch operators matches the fetch_targets
// and fetch_holder_name. Raise exception when any mismatch is found.
// Return true if the block has fetch operators and holder of matching info.
static bool has_fetch_operators(
    BlockDesc* block, std::map<std::string, LoDTensor*>& fetch_targets,
    const std::string& fetch_holder_name) {
  size_t fetch_count = 0;
  for (auto* op : block->AllOps()) {
    if (op->Type() == kFetchOpType) {
      fetch_count++;
      PADDLE_ENFORCE_EQ(op->Output("Out")[0], fetch_holder_name,
                        "Output of fetch op should be '%s'", fetch_holder_name);
      std::string fetch_target_name = op->Input("X")[0];
      PADDLE_ENFORCE(
          fetch_targets.find(fetch_target_name) != fetch_targets.end(),
          "Fetch operator input name '%s' cannot be found in 'fetch_targets'",
          fetch_target_name);
    }
  }

  if (fetch_count > 0) {
    PADDLE_ENFORCE_EQ(
        fetch_count, fetch_targets.size(),
        "The number of fetch operators should match 'fetch_targets'");

    // When fetch operator are present, so should be fetch_holder
    auto var = block->FindVar(fetch_holder_name);
    PADDLE_ENFORCE_NOT_NULL(var, "Block should already have a '%s' variable",
                            fetch_holder_name);
    PADDLE_ENFORCE_EQ(var->GetType(), proto::VarType::FETCH_LIST,
                      "'%s' variable should be 'FETCH_LIST' type",
                      fetch_holder_name);
  }

  return fetch_count > 0;
}

void Executor::Run(const ProgramDesc& program, Scope* scope,
                   std::map<std::string, const LoDTensor*>& feed_targets,
                   std::map<std::string, LoDTensor*>& fetch_targets,
                   const std::string& feed_holder_name,
                   const std::string& fetch_holder_name) {
  auto* copy_program = new ProgramDesc(program);
  auto* global_block = copy_program->MutableBlock(0);

  if (!has_feed_operators(global_block, feed_targets, feed_holder_name)) {
    // create feed_holder variable
    auto* feed_holder = global_block->Var(feed_holder_name);
    feed_holder->SetType(proto::VarType::FEED_MINIBATCH);
    feed_holder->SetPersistable(true);

    int i = 0;
    for (auto& feed_target : feed_targets) {
      std::string var_name = feed_target.first;
      VLOG(3) << "feed target's name: " << var_name;

      // prepend feed op
      auto* op = global_block->PrependOp();
      op->SetType(kFeedOpType);
      op->SetInput("X", {feed_holder_name});
      op->SetOutput("Out", {var_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  // map the data of feed_targets to feed_holder
  for (auto* op : global_block->AllOps()) {
    if (op->Type() == kFeedOpType) {
      std::string feed_target_name = op->Output("Out")[0];
      int idx = boost::get<int>(op->GetAttr("col"));
      SetFeedVariable(scope, *feed_targets[feed_target_name], feed_holder_name,
                      idx);
    }
  }

  if (!has_fetch_operators(global_block, fetch_targets, fetch_holder_name)) {
    // create fetch_holder variable
    auto* fetch_holder = global_block->Var(fetch_holder_name);
    fetch_holder->SetType(proto::VarType::FETCH_LIST);
    fetch_holder->SetPersistable(true);

    int i = 0;
    for (auto& fetch_target : fetch_targets) {
      std::string var_name = fetch_target.first;
      VLOG(3) << "fetch target's name: " << var_name;

      // append fetch op
      auto* op = global_block->AppendOp();
      op->SetType(kFetchOpType);
      op->SetInput("X", {var_name});
      op->SetOutput("Out", {fetch_holder_name});
      op->SetAttr("col", {static_cast<int>(i)});
      op->CheckAttrs();

      i++;
    }
  }

  Run(*copy_program, scope, 0, true, true);

  // obtain the data of fetch_targets from fetch_holder
  for (auto* op : global_block->AllOps()) {
    if (op->Type() == kFetchOpType) {
      std::string fetch_target_name = op->Input("X")[0];
      int idx = boost::get<int>(op->GetAttr("col"));
      *fetch_targets[fetch_target_name] =
          GetFetchVariable(*scope, fetch_holder_name, idx);
    }
  }

  delete copy_program;
}

}  // namespace framework
}  // namespace paddle
