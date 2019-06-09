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

#include "paddle/fluid/operators/distributed/request_handler_impl.h"
#include <iostream>
#include <string>
#include <vector>

#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/scope.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/framework/variable_helper.h"
#include "paddle/fluid/operators/distributed/async_sparse_param_update_recorder.h"
#include "paddle/fluid/operators/distributed/rpc_server.h"
#include "paddle/fluid/string/piece.h"
#include "paddle/fluid/string/printf.h"

namespace paddle {
namespace operators {
namespace distributed {

// define LOOKUP_TABLE_PATH for checkpoint notify to save lookup table variables
// to directory specified.
constexpr char LOOKUP_TABLE_PATH[] = "kLookupTablePath";

bool RequestSendHandler::Handle(const std::string& varname,
                                framework::Scope* scope,
                                framework::Variable* invar,
                                framework::Variable** outvar,
                                const int trainer_id,
                                const std::string& out_var_name,
                                const std::string& table_name) {
  VLOG(4) << "RequestSendHandler:" << varname;

  // Sync
  if (varname == BATCH_BARRIER_MESSAGE) {
    VLOG(3) << "sync: recv BATCH_BARRIER_MESSAGE";
    rpc_server_->IncreaseBatchBarrier(kRequestSend);
  } else if (varname == COMPLETE_MESSAGE) {
    VLOG(3) << "sync: recv complete message";
    rpc_server_->Complete();
  } else {
    // Async
    if (!sync_mode_) {
      VLOG(3) << "async process var: " << varname;
      if (varname == BATCH_BARRIER_MESSAGE) {
        PADDLE_THROW(
            "async mode should not recv BATCH_BARRIER_MESSAGE or "
            "COMPLETE_MESSAGE");
      }
      if (AsyncSparseParamUpdateRecorder::GetInstance()->HasGrad(varname)) {
        auto& grad_slr =
            scope->FindVar(varname)->Get<framework::SelectedRows>();
        AsyncSparseParamUpdateRecorder::GetInstance()->Update(varname,
                                                              grad_slr.rows());
      }
      executor_->RunPreparedContext((*grad_to_prepared_ctx_)[varname].get(),
                                    scope);
      return true;
    } else {  // sync
      rpc_server_->WaitCond(kRequestSend);
      VLOG(3) << "sync: processing received var: " << varname;

      if (invar == nullptr) {
        LOG(FATAL) << "sync: Can not find server side var: " << varname;
        return false;
      }
    }
  }
  return true;
}

bool RequestGetHandler::Handle(const std::string& varname,
                               framework::Scope* scope,
                               framework::Variable* invar,
                               framework::Variable** outvar,
                               const int trainer_id,
                               const std::string& out_var_name,
                               const std::string& table_name) {
  VLOG(3) << "RequestGetHandler:" << varname
          << " out_var_name: " << out_var_name << " trainer_id: " << trainer_id
          << " table_name: " << table_name;

  if (sync_mode_) {
    if (varname == FETCH_BARRIER_MESSAGE) {
      VLOG(3) << "sync: recv fetch barrier message";
      rpc_server_->IncreaseBatchBarrier(kRequestGet);
    } else {
      rpc_server_->WaitCond(kRequestGet);
      *outvar = scope_->FindVar(varname);
    }
  } else {
    if (varname != FETCH_BARRIER_MESSAGE && varname != COMPLETE_MESSAGE) {
      if (enable_dc_asgd_) {
        // NOTE: the format is determined by distributed_transpiler.py
        std::string param_bak_name =
            string::Sprintf("%s.trainer_%d_bak", varname, trainer_id);
        VLOG(3) << "getting " << param_bak_name << " trainer_id " << trainer_id;
        auto var = scope_->FindVar(varname);
        auto t_orig = var->Get<framework::LoDTensor>();
        auto param_bak = scope_->Var(param_bak_name);
        auto t = param_bak->GetMutable<framework::LoDTensor>();
        t->mutable_data(dev_ctx_->GetPlace(), t_orig.type());
        VLOG(3) << "copying " << varname << " to " << param_bak_name;
        framework::TensorCopy(t_orig, dev_ctx_->GetPlace(), t);
      }
      if (AsyncSparseParamUpdateRecorder::GetInstance()->HasParam(varname) &&
          !table_name.empty()) {
        std::vector<int64_t> updated_rows;
        AsyncSparseParamUpdateRecorder::GetInstance()->GetAndClear(
            varname, trainer_id, &updated_rows);
        if (VLOG_IS_ON(3)) {
          std::ostringstream sstream;
          sstream << "[";
          for (auto& row_id : updated_rows) {
            sstream << row_id << ", ";
          }
          sstream << "]";
          VLOG(3) << "updated_rows size: " << updated_rows.size() << " "
                  << sstream.str();
        }
        auto& origin_tensor =
            scope_->FindVar(varname)->Get<framework::LoDTensor>();
        auto* origin_tensor_data = origin_tensor.data<float>();
        auto& dims = origin_tensor.dims();
        *outvar = scope->Var();
        auto* out_slr = (*outvar)->GetMutable<framework::SelectedRows>();
        out_slr->set_rows(updated_rows);
        out_slr->set_height(dims[0]);
        auto out_dims = framework::make_ddim(
            {static_cast<int64_t>(updated_rows.size()), dims[1]});
        auto* data = out_slr->mutable_value()->mutable_data<float>(
            out_dims, origin_tensor.place());
        auto width = dims[1];
        for (auto i = 0; i < updated_rows.size(); ++i) {
          PADDLE_ENFORCE_LT(updated_rows[i], dims[0]);
          memcpy(data + i * width, origin_tensor_data + updated_rows[i] * width,
                 sizeof(float) * width);
        }
      } else {
        *outvar = scope_->FindVar(varname);
      }
    }
  }
  return true;
}

bool RequestGetNoBarrierHandler::Handle(const std::string& varname,
                                        framework::Scope* scope,
                                        framework::Variable* invar,
                                        framework::Variable** outvar,
                                        const int trainer_id,
                                        const std::string& out_var_name,
                                        const std::string& table_name) {
  VLOG(4) << "RequestGetNoBarrierHandler:" << varname
          << " out_var_name: " << out_var_name;

  // get var from pserver immediately without barriers
  string::Piece without_barrier_piece(WITHOUT_BARRIER_MESSAGE);
  string::Piece var_name_piece = string::Piece(varname);

  if (string::Contains(var_name_piece, without_barrier_piece)) {
    var_name_piece = string::TrimSuffix(var_name_piece, without_barrier_piece);
    VLOG(4) << "Get var " << var_name_piece << " with "
            << WITHOUT_BARRIER_MESSAGE;
    *outvar = scope_->FindVar(var_name_piece.ToString());
    return true;
  } else {
    PADDLE_THROW("GetNoBarrier must contain %s", WITHOUT_BARRIER_MESSAGE);
  }
  return true;
}

bool RequestPrefetchHandler::Handle(const std::string& varname,
                                    framework::Scope* scope,
                                    framework::Variable* invar,
                                    framework::Variable** outvar,
                                    const int trainer_id,
                                    const std::string& out_var_name,
                                    const std::string& table_name) {
  VLOG(4) << "RequestPrefetchHandler " << varname;

  if (table_name.empty()) {
    auto var_desc = program_->Block(0).FindVar(out_var_name);
    InitializeVariable(*outvar, var_desc->GetType());
    executor_->RunPreparedContext(
        (*prefetch_var_name_to_prepared_ctx_)[varname].get(), scope);
  } else {
    (*outvar)->GetMutable<framework::LoDTensor>();
    auto lookup_table_op =
        BuildLookupTableOp(table_name, varname, out_var_name);
    paddle::platform::CPUPlace cpu_place;
    lookup_table_op->Run(*scope, cpu_place);
  }
  return true;
}

bool RequestCheckpointHandler::Handle(const std::string& varname,
                                      framework::Scope* scope,
                                      framework::Variable* invar,
                                      framework::Variable** outvar,
                                      const int trainer_id,
                                      const std::string& out_var_name,
                                      const std::string& table_name) {
  PADDLE_ENFORCE(
      checkpoint_notify_id != -1,
      "when checkpoint_notify_id = -1, there should be no RPC invoke.");

  // TODO(tangwei12): find out why scope will be error.
  auto* lt_var = scope_->FindVar(LOOKUP_TABLE_PATH)->GetMutable<std::string>();
  lt_var->clear();
  lt_var->append(out_var_name);
  VLOG(4) << "RequestCheckpointHandler update var kLookupTablePath to: "
          << out_var_name;
  executor_->RunPreparedContext(checkpoint_prepared_ctx_.get(), scope_);
  return true;
}

}  // namespace distributed
}  // namespace operators
}  // namespace paddle
