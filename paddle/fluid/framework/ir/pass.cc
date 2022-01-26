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
#include "paddle/fluid/framework/op_proto_maker.h"

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
  platform::ClearMKLDNNCache(paddle::platform::CPUPlace());
#endif
  VLOG(10) << "finish to apply pass " << Type() << " to graph";
  return graph;
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

static void MergePrograms(ProgramDesc *dst, const details::ProgramDescs &srcs,
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

static void FillNotSpecifiedOpRole(const ProgramDesc &main_program) {
  for (size_t block_idx = 0; block_idx < main_program.Size(); ++block_idx) {
    auto ops = main_program.Block(block_idx).AllOps();
    size_t n = ops.size();
    std::vector<OpRole> roles;
    roles.reserve(n);
    auto op_role_attr = OpProtoAndCheckerMaker::OpRoleAttrName();
    for (auto *op : ops) {
      OpRole role;
      if (op->HasAttr(op_role_attr)) {
        role = static_cast<OpRole>(op->GetAttrIfExists<int>(op_role_attr));
      } else {
        role = OpRole::kNotSpecified;
      }
      roles.emplace_back(role);
    }

    // NOTE: The following codes may be wrong in some cases.
    // But how can we get the right OpRole? The right way
    // is that all passes should deal with unspecified OpRole.
    auto prev_role = OpRole::kForward;
    for (size_t i = 0; i < n; ++i) {
      if (roles[i] == OpRole::kNotSpecified) {
        VLOG(10) << "Fill op role of " << ops[i]->Type() << " as "
                 << static_cast<int>(prev_role);
        ops[i]->SetAttr(op_role_attr, static_cast<int>(prev_role));
      } else {
        prev_role = roles[i];
      }
    }
  }
}

void Pass::ApplyPassesToProgram(const std::vector<const Pass *> &passes,
                                ProgramDesc *main_program,
                                ProgramDesc *startup_program) {
  VLOG(10) << "ApplyPassesToProgram is called";
  PADDLE_ENFORCE_NOT_NULL(
      main_program,
      platform::errors::InvalidArgument("The main program must be provided."));

  PADDLE_ENFORCE_NOT_NULL(startup_program,
                          platform::errors::InvalidArgument(
                              "The startup program must be provided."));

  for (auto *p : passes) {
    PADDLE_ENFORCE_NOT_NULL(p, platform::errors::InvalidArgument(
                                   "The provided pass cannot be nullptr."));
    VLOG(10) << "Pass " << p->Type();
    if (passes.size() > 1) {
      PADDLE_ENFORCE_EQ(p->SupportApplyProgramViaGraph(), true,
                        platform::errors::PermissionDenied(
                            "Each pass must support to be applied via Graph if "
                            "multi-passes are applied."));
    }
  }

  if (passes.size() == 1 && !passes[0]->SupportApplyProgramViaGraph()) {
    VLOG(10) << "apply pass " << passes[0]->Type() << " to program";
    passes[0]->ApplyImpl(main_program, startup_program);
    FillNotSpecifiedOpRole(*main_program);
    VLOG(10) << "finish to apply pass " << passes[0]->Type() << " to program";
    return;
  }

  Graph graph(*main_program);
  for (auto *p : passes) {
    p->Apply(&graph);
  }
  ConvertToPrograms(&graph, main_program, startup_program);
  FillNotSpecifiedOpRole(*main_program);
}

void Pass::ApplyImpl(ProgramDesc *main_program,
                     ProgramDesc *startup_program) const {
  PADDLE_THROW(platform::errors::Unimplemented(
      "The pass %s does not support to apply ProgramDesc directly", Type()));
}

void Pass::ConvertToPrograms(Graph *graph, ProgramDesc *main_program,
                             ProgramDesc *startup_program) {
  ProgramDesc new_main_program;
  GraphToProgram(*graph, &new_main_program);
  main_program->CopyFrom(*new_main_program.Proto());

  if (graph->Has(details::kStartupProgramDescs)) {
    const auto &startups =
        graph->Get<details::ProgramDescs>(details::kStartupProgramDescs);
    VLOG(10) << "Merge startup programs";
    MergePrograms(startup_program, startups, /*append=*/true);
    graph->Erase(details::kStartupProgramDescs);
  }

  if (graph->Has(details::kProgramDescs)) {
    const auto &mains =
        graph->Get<details::ProgramDescs>(details::kProgramDescs);
    VLOG(10) << "Merge main programs";
    MergePrograms(main_program, mains, /*append=*/false);
    graph->Erase(details::kProgramDescs);
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
