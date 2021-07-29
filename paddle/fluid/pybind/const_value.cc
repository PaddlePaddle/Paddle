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

#include "paddle/fluid/pybind/const_value.h"
#include "paddle/fluid/framework/ir/node.h"
#include "paddle/fluid/framework/op_proto_maker.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/proto_desc.h"

#if defined(PADDLE_WITH_DGC)
#include "paddle/fluid/framework/details/dgc_const_values.h"
#include "paddle/fluid/framework/details/sparse_all_reduce_op_handle.h"
#endif

namespace paddle {
namespace pybind {

void BindConstValue(pybind11::module* m) {
  m->def("kEmptyVarName", [] { return framework::kEmptyVarName; });
  m->def("kTempVarName", [] { return framework::kTempVarName; });
  m->def("kGradVarSuffix", [] { return framework::kGradVarSuffix; });
  m->def("kZeroVarSuffix", [] { return framework::kZeroVarSuffix; });
  m->def("kControlDepVarName",
         [] { return framework::ir::Node::kControlDepVarName; });
  m->def("kNewGradSuffix", [] { return framework::kNewGradSuffix; });
  m->def("kAutoParallelSuffix", [] { return framework::kAutoParallelSuffix; });
  m->def("kNoneProcessMeshIndex",
         [] { return framework::kNoneProcessMeshIndex; });

  auto op_proto_and_checker_maker =
      m->def_submodule("op_proto_and_checker_maker");

  pybind11::enum_<framework::OpRole>(op_proto_and_checker_maker, "OpRole")
      .value("Forward", framework::OpRole::kForward)
      .value("Backward", framework::OpRole::kBackward)
      .value("Optimize", framework::OpRole::kOptimize)
      .value("Loss", framework::OpRole::kLoss)
      .value("RPC", framework::OpRole::kRPC)
      .value("Dist", framework::OpRole::kDist)
      .value("LRSched", framework::OpRole::kLRSched);

  op_proto_and_checker_maker.def(
      "kOpRoleAttrName", framework::OpProtoAndCheckerMaker::OpRoleAttrName);
  op_proto_and_checker_maker.def(
      "kOpRoleVarAttrName",
      framework::OpProtoAndCheckerMaker::OpRoleVarAttrName);
  op_proto_and_checker_maker.def(
      "kOpNameScopeAttrName",
      framework::OpProtoAndCheckerMaker::OpNamescopeAttrName);
  op_proto_and_checker_maker.def(
      "kOpCreationCallstackAttrName",
      framework::OpProtoAndCheckerMaker::OpCreationCallstackAttrName);
  op_proto_and_checker_maker.def(
      "kOpDeviceAttrName", framework::OpProtoAndCheckerMaker::OpDeviceAttrName);
#if defined(PADDLE_WITH_DGC)
  auto dgc = m->def_submodule("dgc");
  dgc.def("kDGCKName", [] { return framework::details::g_dgc_k; });
  dgc.def("kDGCEncodedName", [] { return framework::details::g_dgc_encoded; });
  dgc.def("kDGCGatherName", [] { return framework::details::g_dgc_gather; });
  dgc.def("kDGCCounterName",
          [] { return framework::details::g_dgc_counter_name; });
  dgc.def("kDGCRampUpBeginStepName",
          [] { return framework::details::g_dgc_rampup_begin_step; });
  dgc.def("kDGCNRanksName", [] { return framework::details::g_dgc_nranks; });
#endif
}

}  // namespace pybind
}  // namespace paddle
