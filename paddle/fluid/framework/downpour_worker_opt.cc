/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <set>
#include <unordered_map>
#include "paddle/fluid/framework/data_type.h"
#include "paddle/fluid/framework/device_worker.h"
#include "paddle/fluid/framework/device_worker_factory.h"
#include "paddle/fluid/platform/cpu_helper.h"
#include "paddle/fluid/platform/lodtensor_printer.h"

bool DownpourWorkerOpt::HasDependentOutputVar(
        const proto::OpDesc& op_desc,
        const std::unordered_set<std::string>& dependent_vars) {
    for (auto& var : op_desc.outputs()) {
        for (auto& argu : var.arguments()) {
            if (dependent_vars.count(argu) != 0) {
                return true;
            }
        }
    }
    return false;
}

bool DownpourWorkerOpt::HasDependentInputVar(
        const proto::OpDesc& op_desc,
        const std::unordered_set<std::string>& dependent_vars) {
    for (auto& var : op_desc.inputs()) {
        for (auto& argu : var.arguments()) {
            if (dependent_vars.count(argu) != 0) {
                return true;
            }
        }
    }
    return false;
}

bool DownpourWorkerOpt::HasOutput(
        const proto::OpDesc& op_desc,
        const std::string& name) {
    for (auto& var : op_desc.outputs()) {
        for (auto& argu : var.arguments()) {
            if (argu == name) {
                return true;
            }
        }
    }
    return false;
}
void DownpourWorkerOpt::AppendOpInputVarNames(const proto::OpDesc& op_desc,
                           std::unordered_set<std::string>* vars_set) {
    for (auto& var : op_desc.inputs()) {
        for (auto& arg : var.arguments()) {
            vars_set->emplace(arg);
        }
    }
}

void DownpourWorkerOpt::AppendOpOutputVarNames(const proto::OpDesc& op_desc,
                            std::unordered_set<std::string>* vars_set) {
    for (auto& var : op_desc.outputs()) {
        for (auto& arg : var.arguments()) {
            vars_set->emplace(arg);
        }
    }
}
void DownpourWorkerOpt::CreateThreadOperatorsWithRerank(const ProgramDesc& program) {
    auto &block = program.Block(0);
    auto* ops = block.AllOps();
    // check if Independent between losses if not skip for now
    int loss_num = loss_names_.size();
    std::vector<std::string, std::unordered_set<std::string>> loss_input_vars;
    std::vector<std::string, std::unordered_set<std::string>> loss_output_vars;
    loss_vars.resize(loss_num);
    // mark forward ops by loss
    for (int i = 0; i < loss_num; i++) {
      for (auto op_iter = ops->rbegin(); op_iter != ops->rend(); ++op_iter) {
          auto &op_desc = *op_iter;
          for (int j = 0; j < loss_num-1; j++) {
              if (HasDependence(op_desc, loss_input_vars)) {
                  VLOG(3) << "losses must be independence currently";
                  return;
              }
          }
          if (HasOutput(op_desc, loss_names[i]) || HasDependentOutputVar(op_desc, loss_vars[i])) {
              AppendLossInputVarNames(op_desc, &loss_input_vars[i]);
              AppendLossOutputVarNames(op_desc, &loss_output_vars[i]);
          }
      }
    }

    //
    std::vector<std::string> loss_grad_names;
    for (int i = 0; i < loss_num; i++) {
        loss_grad_names.push_back(loss_names_[i]+"@GRAD");
    }
    std::vector<std::string, std::unordered_set<std::string>> loss_input_grad_vars;
    std::vector<std::string, std::unordered_set<std::string>> loss_output_grad_vars;
    for (int i = 0; i < loss_num; i++) {
        for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
            auto &op_desc = *op_iter;

            if (HasOutput(op_desc, loss_grad_names[i]) || HasDependentOutputVar(op_desc, loss_output_grad_vars[i])) {
                AppendLossInputVarNames(op_desc, &loss_input_grad_vars[i]);
                AppendLossOutputVarNames(op_desc, &loss_output_grad_vars[i]);
            }
        }
    }


    std::vector<std::string, std::unordered_set<std::string>> metric_input_vars;
    std::vector<std::string, std::unordered_set<std::string>> metric_output_vars;
    for (int i = 0; i < loss_num; i++) {
        for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
            auto &op_desc = *op_iter;
            if ((HasDependentInputVar(op_desc, loss_output_vars[i]) &&
                NotHasDependentOutputVar(op_desc, loss_input_vars[i])) ||
                HasDependentInputVar(op_desc, metric_output_vars[i])) {
                AppendLossInputVarNames(op_desc, &metric_input_vars[i]);
                AppendLossOutputVarNames(op_desc, &metric_input_vars[i]);
            }
        }
    }
    //
    for (int i = 0; i < loss_num; i++) {
        for (auto op_iter = ops->begin(); op_iter != ops->end(); ++op_iter) {
            auto &op_desc = *op_iter;
            if (HasDependentInputVar(op_desc, loss_input_vars[i]) ||
                HasDependentInputVar(op_desc, loss_input_grad_vars[i]) ||
                HasDependentInputVar(op_desc, metric_input_vars[i])) {
                std::unique_ptr<OperatorBase> local_op = OpRegistry::CreateOp(*op_desc);
                op_names_.push_back(op_desc->Type());
                OperatorBase *local_op_ptr = local_op.release();
                loss_ops_[i].push_back(local_op_ptr);
            }
        }
    }



}