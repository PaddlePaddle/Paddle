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

#include "paddle/fluid/framework/ir/pass.h"

#include <algorithm>
#include "paddle/fluid/framework/ir/graph_helper.h"

namespace paddle {
namespace framework {
namespace ir {
class Graph;
}  // namespace ir
}  // namespace framework
}  // namespace paddle
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif

namespace paddle {
namespace framework {
namespace ir {

Graph *Pass::Apply(Graph *graph) const {
  VLOG(10) << "start to apply pass " << Type() << " to graph";
  CheckPrevPass();
  PADDLE_ENFORCE_NOT_NULL(
      graph, platform::errors::InvalidArgument("Graph cannot be nullptr."));
  for (const std::string &attr : required_pass_attrs_) {
    PADDLE_ENFORCE_NE(
        attrs_.find(attr), attrs_.end(),
        platform::errors::InvalidArgument(
            "Required atrribute %s for pass < %s > is not set.", attr, Type()));
  }
  for (const std::string &attr : required_graph_attrs_) {
    PADDLE_ENFORCE_EQ(graph->Has(attr), true,
                      platform::errors::InvalidArgument(
                          "Required atrribute %s for graph is not set.", attr));
  }
  ApplyImpl(graph);
  // TODO(panyx0718): Add more verifications.
  PADDLE_ENFORCE_EQ(
      HasCircle(*graph), false,
      platform::errors::InvalidArgument(
          "Illegal pass %s. Generated graph shouldn't contain cycle.", Type()));
  PADDLE_ENFORCE_EQ(
      VarDescIsConsistency(*graph), true,
      platform::errors::InvalidArgument(
          "The VarDescs of persistable variable are not consistency."));
  applied_ = true;
  if (!graph->Has(kPassRecorder)) {
    graph->Set<PassRecorder>(kPassRecorder, new PassRecorder);
  }
  graph->Get<PassRecorder>(kPassRecorder).insert(Type());
#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // Passes can change params, tensors, so caching need to be discarded
  ClearMKLDNNCache(paddle::platform::CPUPlace());
#endif
  VLOG(10) << "finish to apply pass " << Type() << " to graph";
  return graph;
}

void Pass::Apply(ProgramDesc *main_program,
                 ProgramDesc *startup_program) const {
  VLOG(10) << "apply pass " << Type() << " to program";
  PADDLE_ENFORCE_NOT_NULL(main_program, platform::errors::InvalidArgument(
                                            "main program must be provided"));
  PADDLE_ENFORCE_NOT_NULL(
      startup_program,
      platform::errors::InvalidArgument("startup program must be provided"));

  ApplyImpl(main_program, startup_program);
  VLOG(10) << "finish to apply pass " << Type() << " to program";
}

template <typename Container, typename Visitor>
static void VisitAllElements(Container &&container, Visitor &&visitor,
                             bool reverse) {
  if (reverse) {
    std::for_each(container.rbegin(), container.rend(), visitor);
  } else {
    std::for_each(container.begin(), container.end(), visitor);
  }
}

void Pass::MergePrograms(ProgramDesc *dst, const details::ProgramDescs &srcs,
                         bool append) {
  PADDLE_ENFORCE_NOT_NULL(
      dst, platform::errors::InvalidArgument("Dst program must be provided."));
  bool reverse = !append;

  auto create_var_visitor = [dst](const ProgramDesc &src) {
    PADDLE_ENFORCE_EQ(src.Size(), 1, platform::errors::Unimplemented(
                                         "MergePrograms can only support to "
                                         "merge program with only one block."));
    const auto &src_block = src.Block(0);
    auto *dst_block = dst->MutableBlock(0);
    for (const auto *src_new_var : src_block.AllVars()) {
      if (dst_block->FindVar(src_new_var->Name())) continue;
      auto *dst_new_var = dst_block->Var(src_new_var->Name());
      *dst_new_var = *src_new_var;
      VLOG(10) << "Create new variable " << dst_new_var->Name();
    }
  };
  VisitAllElements(srcs, create_var_visitor, reverse);

  auto create_op_visitor = [dst, reverse](const ProgramDesc &src) {
    auto ops = src.Block(0).AllOps();
    auto copy_op_visitor = [dst, reverse](const OpDesc *src_op) {
      auto *dst_block = dst->MutableBlock(0);
      auto *op = reverse ? dst_block->PrependOp() : dst_block->AppendOp();
      op->CopyFrom(*src_op);
      VLOG(10) << (reverse ? "Prepend" : "Append") << " op " << op->Type();
      // FIXME(zjl): some passes does not add VarDesc to program,
      // we should fix this bug later...
      for (const auto &in_var_name : op->InputArgumentNames()) {
        dst_block->Var(in_var_name);
      }
      for (const auto &out_var_name : op->OutputArgumentNames()) {
        dst_block->Var(out_var_name);
      }
    };
    VisitAllElements(ops, copy_op_visitor, reverse);
  };
  VisitAllElements(srcs, create_op_visitor, reverse);
}

void Pass::ApplyImpl(ProgramDesc *main_program,
                     ProgramDesc *startup_program) const {
  Graph graph(*main_program);
  Apply(&graph);

  ProgramDesc new_main_program;
  GraphToProgram(graph, &new_main_program);
  main_program->CopyFrom(*new_main_program.Proto());

  if (graph.Has(details::kStartupProgramDescs)) {
    const auto &startups =
        graph.Get<details::ProgramDescs>(details::kStartupProgramDescs);
    VLOG(10) << "Merge startup programs";
    MergePrograms(startup_program, startups, /*append=*/true);
  }

  if (graph.Has(details::kProgramDescs)) {
    const auto &mains =
        graph.Get<details::ProgramDescs>(details::kProgramDescs);
    VLOG(10) << "Merge main programs";
    MergePrograms(main_program, mains, /*append=*/false);
  }

  startup_program->Flush();
  main_program->Flush();
}

PassRegistry &PassRegistry::Instance() {
  static PassRegistry g_pass_info_map;
  return g_pass_info_map;
}
}  // namespace ir
}  // namespace framework
}  // namespace paddle
