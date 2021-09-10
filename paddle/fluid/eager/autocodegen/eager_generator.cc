// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <algorithm>
#include <fstream>
#include <iostream>
#include <string>
#include <unordered_set>

#include "paddle/fluid/framework/op_info.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/operator.h"
#include "paddle/fluid/framework/variable.h"
#include "paddle/fluid/pybind/pybind.h"
#include "paddle/fluid/string/string_helper.h"

static std::unordered_set<std::string> operators_to_skip = {
    "fused_elemwise_add_activation", // No Default Attr
    "fused_elemwise_activation", // No Default Attr
    "reverse", // Attr Error
    "flip", // Attr Error
    "cast", // Attr Error 
    "minus" // Multiple ops_

};

using namespace paddle;
using namespace framework;
using namespace imperative;

// TODO:
// 1. Number of duplicable inputs not known at code generation time
// 2. Number of duplicable outputs not known at code generation time
// 3. Ops with duplicable outputs like SumOp need to be handled manually?

void SlotNameMatching(const std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>>& grad_map,
                      const std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>>& fwd_ins,
                      const std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>>& fwd_outs,
                      std::map<std::string, std::string>& grad_fwd_slotname_map,
                      std::map<std::string, std::string>& grad_grad_slotname_map) {
    
    for(const auto& iter : grad_map) {
        const std::string& grad_slot_name = iter.first;
        const std::vector<std::shared_ptr<VariableWrapper>>& grad_vars = iter.second;
        
        // Find matching fwd_slot_name
        bool found_matching = false;
        for(const std::shared_ptr<VariableWrapper>& grad_var : grad_vars) {
            for(const auto& fwd_iter : fwd_ins) {
                const std::string& fwd_slot_name = fwd_iter.first;
                const std::vector<std::shared_ptr<VariableWrapper>>& fwd_vars = fwd_iter.second;
                for(const std::shared_ptr<VariableWrapper>& fwd_var : fwd_vars) {
                    if(grad_var == fwd_var) {
                        if(grad_fwd_slotname_map.count(grad_slot_name) && grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
                            PADDLE_THROW(platform::errors::Fatal("grad_slot_name %s matches both %s and %s fwd_slot_name", 
                                        grad_slot_name, grad_fwd_slotname_map[grad_slot_name], fwd_slot_name));
                        }
                        grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
                        found_matching = true;
                    }

                    if(fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
                        if(grad_grad_slotname_map.count(grad_slot_name) && grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
                            PADDLE_THROW(platform::errors::Fatal("grad_slot_name %s matches both %s and %s fwd_slot_name", 
                                        grad_slot_name, grad_grad_slotname_map[grad_slot_name], fwd_slot_name));
                        }
                        grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
                        found_matching = true;
                    }
                }
            }
            for(const auto& fwd_iter : fwd_outs) {
                const std::string& fwd_slot_name = fwd_iter.first;
                const std::vector<std::shared_ptr<VariableWrapper>>& fwd_vars = fwd_iter.second;
                for(const std::shared_ptr<VariableWrapper>& fwd_var : fwd_vars) {
                    if(grad_var == fwd_var) {
                        if(grad_fwd_slotname_map.count(grad_slot_name) && grad_fwd_slotname_map[grad_slot_name] != fwd_slot_name) {
                            PADDLE_THROW(platform::errors::Fatal("grad_slot_name %s matches both %s and %s fwd_slot_name", 
                                        grad_slot_name, grad_fwd_slotname_map[grad_slot_name], fwd_slot_name));
                        }
                        grad_fwd_slotname_map[grad_slot_name] = fwd_slot_name;
                        found_matching = true;
                    }

                    if(fwd_var->GetGradVar() && grad_var == fwd_var->GetGradVar()) {
                        if(grad_grad_slotname_map.count(grad_slot_name) && grad_grad_slotname_map[grad_slot_name] != fwd_slot_name) {
                            PADDLE_THROW(platform::errors::Fatal("grad_slot_name %s matches both %s and %s fwd_slot_name", 
                                        grad_slot_name, grad_grad_slotname_map[grad_slot_name], fwd_slot_name));
                        }
                        grad_grad_slotname_map[grad_slot_name] = fwd_slot_name;
                        found_matching = true;
                    }
                }
            }
        }

        if(!found_matching) {
            PADDLE_THROW(platform::errors::Fatal("Found no matching fwd_slot_name for grad_slot_name: %s", grad_slot_name));
        
        } else {
            std::string fwd_slot_name = grad_grad_slotname_map.count(grad_slot_name) ? grad_grad_slotname_map[grad_slot_name] : grad_fwd_slotname_map[grad_slot_name];
            VLOG(2) << "Found matching fwd_slot_name: " << fwd_slot_name  << " for grad_slot_name: " << grad_slot_name;
        }
    }
}

int main() {
    auto& op_info_map = paddle::framework::OpInfoMap::Instance().map();
    auto& all_kernels = paddle::framework::OperatorWithKernel::AllOpKernels();
    for (auto& pair : op_info_map) {
        /* --------------------------- */
        /* ------ Preprocessing ------ */
        /* --------------------------- */
        const OpInfo& op_info = pair.second;
        const proto::OpProto* op_proto = op_info.proto_;
        if (op_proto == nullptr) {
            continue;
        }
        const std::string& op_type = op_proto->type();

        // Skip ooerator which is not inherit form OperatorWithKernel, like while,
        // since only OperatorWithKernel can run in dygraph mode.
        if (!all_kernels.count(op_type)) {
            continue;
        }
        
        // Only handle matmul_v2 for now
        VLOG(2) << "------ Analyzing Op ------: " << op_type;
        
        if(operators_to_skip.count(op_type))
            continue;
        //if(op_type != "matmul_v2")
        //    continue;

        /* --------------------------- */
        /* --------- Forward --------- */
        /* --------------------------- */

        std::vector<int64_t> dims = {1,1,1,1};
      
        /* ------ Prepare "ins" ------ */
        std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins;
        for (const proto::OpProto::Var& input : op_proto->inputs()) {
            const std::string& in_name = input.name();
            
            // Handle dispensable input:
            // 1. At python level, dispensable input will be detected at Python-C interface and filled with an empty vector
            // 2. At C++ level, customers should always pass an empty vector for any dispensable input
            // 3. During further lowering, there will always be a placeholder VarBase in ins/outs no matter whether it's dispensable or not
            // As a result, we always create input VarBase regardless of its dispensability.

            // Handle duplicable input: list(VarBase) or VarBase
            // We dont know the exact number of inputs expected,
            // but we only need to identify the slot name order, 
            // therefore fill in 1 single input VarBase is enough in this scenario
            ins[in_name] = { std::shared_ptr<VarBase>(new VarBase("auto_" + in_name)) };
            ins[in_name][0]->SetOverridedStopGradient(false);
            ins[in_name][0]->MutableVar()->GetMutable<framework::LoDTensor>();
        }
        VLOG(2) << "Prepared Forward Ins Map, size = " << ins.size();
        
        /* ------ Prepare "outs" ------ */
        std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs;
        for (const proto::OpProto::Var& output : op_proto->outputs()) {
            const std::string& out_name = output.name();
            
            // We always create output VarBase regardless of its dispensability.
            // We dont know the exact number of outputs during code generation,
            // however, simply identifying the slot name order would be enough
            outs[out_name] = { std::shared_ptr<VarBase>(new VarBase("auto_" + out_name)) };
            outs[out_name][0]->SetOverridedStopGradient(false);
            outs[out_name][0]->MutableVar()->GetMutable<framework::LoDTensor>();
        }
        VLOG(2) << "Prepared Forward Outs Map, size = " << outs.size();
        
        /* ------ Prepare "attrs" ------ */
        framework::AttributeMap attrs;
        for (const proto::OpProto::Attr& attr : op_proto->attrs()) {
            VLOG(2) << attr.name();
            // Attrs only usable during codegen for fwd function or GradNode
            // So we use default attrs for now
        }

        auto* attr_checker = op_info.Checker();
        paddle::framework::AttributeMap default_attrs;
        if (attr_checker) {
            attr_checker->Check(&attrs, true, /*only_check_exist_value=*/true);
            default_attrs = attr_checker->GetDefaultAttrMap();
        } else {
            VLOG(2) << "Detected Null Attribute Checker, use empty default_attrs";
        }
        
        VLOG(2) << "Prepared Default Attributes Map, size = " << default_attrs.size();
        
        /* ---------------------------- */
        /* --------- Backward --------- */
        /* ---------------------------- */
        /* ------ Fwd VariableWrapper Map ------ */
        std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>> fwd_ins;
        std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>> fwd_outs;
        for(const auto& iter : ins) {
            fwd_ins[iter.first] = {};
            for(const std::shared_ptr<VarBase>& var_base : iter.second) {
                fwd_ins[iter.first].push_back(var_base->SharedVar());
            }
        }
        for(const auto& iter : outs) {
            fwd_outs[iter.first] = {};
            for(const std::shared_ptr<VarBase>& var_base : iter.second) {
                fwd_outs[iter.first].push_back(var_base->SharedVar());
            }
        }
        VLOG(2) << "Constructed Forward VariableWrapper Map";
    
        /* ------ Run GradOpMaker ------ */
        if(!op_info.dygraph_grad_op_maker_) {
            VLOG(2) << op_type << " has no GradOpMaker, skip it";
            continue;
        }

        std::shared_ptr<GradOpNode> grad_node = op_info.dygraph_grad_op_maker_(op_type, ins, outs, attrs, default_attrs, {});

        if(!grad_node) {
            VLOG(2) << "Got nullptr GradOpNode for " << op_type << " likely registered EmptyGradOpMaker, skip it";
            continue;
        }

        VLOG(2) << "Prepared GradOpNode";
        
        /* ------ Get Grad ins/outs ---- */
        // In case of multiple OpBase, stitch all the respective ins/outs into one
        std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>> grad_ins;
        std::map<std::string, std::vector<std::shared_ptr<VariableWrapper>>> grad_outs;
        
        for(auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
            OpBase& op_base = *iter;
            std::map<std::string, SavedVariableWrapperList>* g_ins = op_base.GetMutableInsMap();
            std::map<std::string, SavedVariableWrapperList>* g_outs = op_base.GetMutableOutsMap();

            for(const auto& iter : *g_ins) {
                if(!grad_ins.count(iter.first))
                    grad_ins[iter.first] = {};
                for(auto vw_iter = iter.second.begin(); vw_iter != iter.second.end(); vw_iter++) {
                    std::shared_ptr<VariableWrapper> vw = *vw_iter;
                    grad_ins[iter.first].push_back(vw);
                }
            }
            
            for(const auto& iter : *g_outs) {
                if(!grad_outs.count(iter.first))
                    grad_outs[iter.first] = {};
                for(auto vw_iter = iter.second.begin(); vw_iter != iter.second.end(); vw_iter++) {
                    std::shared_ptr<VariableWrapper> vw = *vw_iter;
                    grad_outs[iter.first].push_back(vw);
                }
            }
        }
        VLOG(2) << "Prepared Grad_Ins Map, size = " << grad_ins.size();
        VLOG(2) << "Prepared Grad_Outs Map, size = " << grad_outs.size();

        /* ------ Slot Name Matching ---- */
        // grad_ins -> fwd_ins, fwd_outs
        std::map<std::string, std::string> grad_ins_fwd_slotname_map;
        std::map<std::string, std::string> grad_ins_grad_slotname_map;
        SlotNameMatching(grad_ins, fwd_ins, fwd_outs, grad_ins_fwd_slotname_map, grad_ins_grad_slotname_map);
        VLOG(2) << "Finished Slotname Matching for Grad_Ins";

        // grad_outs -> fwd_ins, fwd_outs
        std::map<std::string, std::string> grad_outs_slotname_map;
        SlotNameMatching(grad_outs, fwd_ins, fwd_outs, grad_outs_slotname_map, grad_outs_slotname_map);
        VLOG(2) << "Finished Slotname Matching for Grad_Outs";

        /* ---------------------------------- */
        /* --------- CodeGen: Inner --------- */
        /* ---------------------------------- */

        /* ------ Maping grad slot name to fwd position ------ */
        // Example: 
        // std::tuple<vector<Tensor>, ...> kernel_function(vector<Tensor>& X, vector<Tensor>& Y, Tensor& Z);
        std::unordered_map<std::string, size_t> fwd_inputs_name_pos_map;
        std::unordered_map<std::string, size_t> fwd_outputs_name_pos_map;
        // Follow map's order
        size_t in_pos = 0;
        for(const auto& iter : ins) {
            fwd_inputs_name_pos_map[iter.first] = in_pos;
            in_pos++;
        }
        size_t out_pos = 0;
        for(const auto& iter : outs) {
            fwd_outputs_name_pos_map[iter.first] = out_pos;
            out_pos++;
        }
    }

    return 0;
}
