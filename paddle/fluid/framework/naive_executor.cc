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

#include "paddle/fluid/framework/naive_executor.h"
#include <string>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/platform/denormal.h"
#ifdef PADDLE_WITH_MKLDNN
#include "paddle/fluid/platform/mkldnn_helper.h"
#endif
#if PADDLE_WITH_TENSORRT
#include "paddle/fluid/operators/tensorrt/tensorrt_engine_op.h"
#endif

namespace paddle {
namespace framework {
void NaiveExecutor::Prepare(Scope *scope, const ProgramDesc &program_desc,
                            int block_id, bool with_feed_fetch_ops) {
  if (!scope) {
    scope_ = new framework::Scope;
  } else {
    scope_ = scope;
  }

  VLOG(3) << "NaiveExecutor init with scope " << scope;
  CreateOps(program_desc, block_id, with_feed_fetch_ops);
}

void NaiveExecutor::Run() {
#ifdef PADDLE_WITH_MKLDNN
  platform::AttachPointerHashToMKLDNNKey(this, place_);
  platform::RegisterModelLayout(ops_, place_);
#endif
  platform::ScopedFlushDenormal flush;
  for (auto &op : ops_) {
    VLOG(4) << std::this_thread::get_id() << " run "
            << op->DebugStringEx(scope_) << " on scope " << scope_;
    op->SetIsCalledByExecutor(false);
    op->Run(*scope_, place_);
  }
}

void NaiveExecutor::CreateVariables(const ProgramDesc &desc, int block_id,
                                    bool persistable, Scope *scope) {
  PADDLE_ENFORCE_NOT_NULL(scope,
                          platform::errors::InvalidArgument(
                              "The Scope to hold variables is nullptr."));

  auto &global_block = desc.Block(block_id);

  const auto *anc = scope;
  PADDLE_ENFORCE_NE(
      anc->parent(), anc,
      platform::errors::InvalidArgument("Input scope should be child scope."));
  while (anc->parent()) {
    anc = anc->parent();
  }

  int num_vars = 0;
  for (auto &var : global_block.AllVars()) {
    if (var->Name() == framework::kEmptyVarName) {
      continue;
    }
    num_vars++;

    if (persistable == var->Persistable()) {
      if (persistable) {
        if (!anc->FindVar(var->Name())) {
          auto *ptr = const_cast<Scope *>(anc)->Var(var->Name());
          VLOG(3) << scope << " Create persistable variable " << var->Name()
                  << ", which pointer is " << ptr;
          InitializeVariable(ptr, var->GetType());
        }
      } else {
        auto *ptr = const_cast<Scope *>(scope)->Var(var->Name());
        VLOG(3) << scope << " Create variable " << var->Name()
                << ", which pointer is " << ptr;
        InitializeVariable(ptr, var->GetType());
      }
    }
  }
  VLOG(4) << "naive executor create " << num_vars << " vars";
}

void NaiveExecutor::CreateOps(const ProgramDesc &desc, int block_id,
                              bool with_feed_fetch_ops) {
  for (const auto &op_desc : desc.Block(block_id).AllOps()) {
    if (!with_feed_fetch_ops &&
        (op_desc->Type() == "feed" || op_desc->Type() == "fetch")) {
      LOG(INFO) << "---  skip [" << op_desc->Input("X")[0] << "], "
                << op_desc->Type() << " -> " << op_desc->Output("Out")[0];
      continue;
    }
    ops_.emplace_back(OpRegistry::CreateOp(*op_desc));
  }
}

LoDTensor *NaiveExecutor::FindTensor(const std::string &name) {
  PADDLE_ENFORCE_NOT_NULL(scope_,
                          platform::errors::PreconditionNotMet(
                              "Need to init scope in NaiveExecutor firstly."));
  auto *var = scope_->FindVar(name);
  PADDLE_ENFORCE_NOT_NULL(var, platform::errors::NotFound(
                                   "No variable [%s] in current scope.", name));
  auto *tensor = const_cast<LoDTensor *>(&var->Get<LoDTensor>());
  return tensor;
}

void NaiveExecutor::CleanFeedFetchOps() {
  std::vector<std::unique_ptr<OperatorBase>> ops;
  for (auto &op : ops_) {
    if (op->Type() != "feed" && op->Type() != "fetch") {
      ops.emplace_back(std::move(op));
    }
  }
  ops_.swap(ops);
}

NaiveExecutor::~NaiveExecutor() {
#ifdef PADDLE_WITH_MKLDNN
  // Clear mkl-dnn cache,
  // this is needed to have mkl-dnn unit tests working
  platform::ClearMKLDNNCache(place_, this);
#endif
}

void NaiveExecutor::ResetTrtOps(int num) {
#if PADDLE_WITH_TENSORRT
  for (auto &op : ops_) {
    if (op->Type() == "tensorrt_engine") {
      operators::TensorRTEngineOp *trtop =
          dynamic_cast<operators::TensorRTEngineOp *>(op.get());
      if (!trtop) return;
      std::string engine_key = trtop->Attr<std::string>("engine_key");
      int engine_predictor_id = trtop->Attr<int>("predictor_id");
      std::string engine_name =
          engine_key + std::to_string(engine_predictor_id);
      operators::TensorRTEngine *trt_engine = nullptr;
      // can't get trt engine if int8 calibration table data process.
      if (paddle::inference::Singleton<
              inference::tensorrt::TRTEngineManager>::Global()
              .Has(engine_name)) {
        trt_engine = paddle::inference::Singleton<
                         inference::tensorrt::TRTEngineManager>::Global()
                         .Get(engine_name);
      }
      if (trt_engine && trt_engine->with_dynamic_shape()) {
        LOG(INFO) << "rebuild trt engine, this may cost a lot of time!";
        trt_engine->ResetContext();
        trt_engine->ClearTensorMap();
        trt_engine->SetProfileNum(num);
        auto *anc = scope_->parent();
        while (anc && anc->parent()) {
          anc = anc->parent();
        }
        if (anc == nullptr) {
          anc = scope_;
        }
        trtop->PrepareTRTEngine(*anc, trt_engine);
      }
    }
  }
#endif
}
}  // namespace framework
}  // namespace paddle
