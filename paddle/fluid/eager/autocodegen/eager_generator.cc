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
    "sum",
    "minus" // Multiple ops_
};

static std::unordered_set<std::string> skipped_operators = {};

using namespace paddle;
using namespace framework;
using namespace imperative;

// TODO:
// 1. Number of duplicable inputs not known at code generation time
// 2. Number of duplicable outputs not known at code generation time
// 3. Ops with duplicable outputs like SumOp need to be handled manually?

std::string AttrTypeToString(const proto::AttrType& type) {
    std::string ret;
    switch(type) {
        case(proto::AttrType::INT): {
            ret = "int";
            break;
        }
        case(proto::AttrType::FLOAT): {
            ret = "float";
            break;
        }
        case(proto::AttrType::STRING): {
            ret = "std::string&";
            break;
        }
        case(proto::AttrType::INTS): {
            ret = "std::vector<int>&";
            break;
        }
        case(proto::AttrType::FLOATS): {
            ret = "std::vector<float>&";
            break;
        }
        case(proto::AttrType::STRINGS): {
            ret = "std::vector<std::string>&";
            break;
        }
        case(proto::AttrType::BOOLEAN): {
            ret = "bool";
            break;
        }
        case(proto::AttrType::BOOLEANS): {
            ret = "std::vector<bool>&";
            break;
        }
        case(proto::AttrType::LONG): {
            ret = "int64_t";
            break;
        }
        case(proto::AttrType::LONGS): {
            ret = "std::vector<int64_t>&";
            break;
        }
        case(proto::AttrType::BLOCK): {
            ret = "paddle::framework::BlockDesc*";
            break;
        }
        case(proto::AttrType::BLOCKS): {
            ret = "std::vector<paddle::framework::BlockDesc*>&";
            break;
        }
        case(proto::AttrType::FLOAT64S): {
            ret = "std::vector<double>&";
            break;
        }
        default: {
            PADDLE_THROW(platform::errors::Fatal("Unable to recognize AttrType: %d", type)); 
        }
    }
    return ret;
}

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

        //if(op_type != "matmul_v2") continue;

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
        
        framework::AttributeMap attrs;
        paddle::framework::AttributeMap default_attrs;
        auto* attr_checker = op_info.Checker();
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
            skipped_operators.insert(op_type);
            continue;
        }

        std::shared_ptr<GradOpNode> grad_node = op_info.dygraph_grad_op_maker_(op_type, ins, outs, attrs, default_attrs, {});

        if(!grad_node) {
            VLOG(2) << "Got nullptr GradOpNode for " << op_type << " likely registered EmptyGradOpMaker, skip it";
            skipped_operators.insert(op_type);
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

            for(const auto& it : *g_ins) {
                if(!grad_ins.count(it.first))
                    grad_ins[it.first] = {};
                for(auto vw_iter = it.second.begin(); vw_iter != it.second.end(); vw_iter++) {
                    std::shared_ptr<VariableWrapper> vw = *vw_iter;
                    grad_ins[it.first].push_back(vw);
                }
            }
            
            for(const auto& it : *g_outs) {
                if(!grad_outs.count(it.first))
                    grad_outs[it.first] = {};
                for(auto vw_iter = it.second.begin(); vw_iter != it.second.end(); vw_iter++) {
                    std::shared_ptr<VariableWrapper> vw = *vw_iter;
                    grad_outs[it.first].push_back(vw);
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
        
        /* ------ Get Grad Attr Map ---- */
        

        /* -------------------------------- */
        /* --------- CodeGen: All --------- */
        /* -------------------------------- */

        /* ------ Maping forward slot name to fwd position ------ */
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
        
        std::vector<std::string> fwd_outputs_names;
        for(const auto& iter : outs) {
            fwd_outputs_names.push_back(iter.first);
        }
        
        /* -------------------------------- */
        /* --------- CodeGen: Forward ----- */
        /* -------------------------------- */
        {
        
        /* 
            // Forward Function Example:
            // Function Proto
            std::tuple<vector<Tensor>, Tensor, vector<Tensor>> kernel_function(vector<vector<Tensor>>& Inputs, 
                                                                               attr0, attr1, ..., size_t Out0Num, size_t Out1Num) {
                
                const std::shared_ptr<Tracer>& tracer = imperative::GetCurrentTracer();
               
                // Forward Function Body
                // According to fwd_inputs_name_pos_map
                std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins = 
                        { {"X" , TensorsToVarBases(Inputs["fwd_inputs_name_pos_map["X"]"])}, { "Y" , TensorsToVarBases("fwd_inputs_name_pos_map["Y"]")} };

                std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs = 
                        { {"Out0" , ConstructDuplicableOutput(Out0Num)}, {"Out1" , ConstructDuplicableOutput(Out1Num)} };

                // According to op_proto->attrs()
                framework::AttributeMap attrs = { {"attr0", attr0}, {"attr1", attr1}, ... };
                tracer->TraceOp("op_type", ins, outs, attrs, tracer->ExpectedPlace(), false, {});
                
                
                // According to fwd_outputs_names
                vector<Tensor> Out0 = VarBasesToTensors(outs["Out0"]); // if duplicable
                Tensor Out1 = VarBaseToTensor(outs["Out1"])[0]; // if not duplicable
                vector<Tensor> Out2 = VarBasesToTensors(outs["Out2"]); // if duplicable
                
                // Backward Function Body
                ...
                
                return std::make_tuple(Out0, Out1, Out2);
            }
        */
        
        std::string generated_function_body = "";
        std::string dygraph_function_args_str = "std::vector<std::vector<pt::Tensor>>& Inputs";

        /* ------ Dygraph forward function generation ------ */
        // [Generation] Get Tracer
        generated_function_body += "\n";
        std::string tracer_str = "const std::shared_ptr<Tracer>& tracer = imperative::GetCurrentTracer();";
        generated_function_body += tracer_str;
        generated_function_body += "\n";

        // [Generation] Get Attrs
        generated_function_body += "\n";
        std::string attr_contents_str = "";
        for (const proto::OpProto::Attr& attr : op_proto->attrs()) {
            const std::string& attr_name = attr.name();
            
            proto::AttrType attr_type = attr.type();
            const std::string attr_type_str = AttrTypeToString(attr_type);
            
            const char* FWD_KERNEL_ARG_TEMPLATE = ", %s %s";
            std::string arg_str = paddle::string::Sprintf(FWD_KERNEL_ARG_TEMPLATE, attr_type_str, attr_name);
            dygraph_function_args_str += arg_str;

            const char* FWD_ATTR_CONTENT_TEMPLATE = "{ \"%s\", %s }, ";
            std::string attr_content_str = paddle::string::Sprintf(FWD_ATTR_CONTENT_TEMPLATE, attr_name, attr_name);
            attr_contents_str += attr_content_str;
        }

        const char* FWD_ATTR_MAP_TEMPLATE = "framework::AttributeMap attrs = { %s };";
        std::string attr_map_str = paddle::string::Sprintf(FWD_ATTR_MAP_TEMPLATE, attr_contents_str);
        generated_function_body += attr_map_str;
        generated_function_body += "\n";

        // [Generation] Get Ins Map
        generated_function_body += "\n";
        std::string ins_contents_str = "";
        for (const proto::OpProto::Var& input : op_proto->inputs()) {
            const std::string& input_name = input.name();
            const char* FWD_INS_CONTENT_TEMPLATE = "{ \"%s\", TensorsToVarBases(Inputs[%s]) },";
            ins_contents_str += paddle::string::Sprintf(FWD_INS_CONTENT_TEMPLATE, input_name, fwd_inputs_name_pos_map[input_name]);
        }
        if(ins_contents_str.size() > 0)
            ins_contents_str.pop_back(); // // Remove trailing ","
        
        const char* FWD_INS_MAP_TEMPLATE = "std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins = { %s };";
        std::string ins_map_str = paddle::string::Sprintf(FWD_INS_MAP_TEMPLATE, ins_contents_str);
        generated_function_body += ins_map_str;
        generated_function_body += "\n";
        
        // [Generation] Get Outs Map
        generated_function_body += "\n";
        std::string outs_contents_str = "";
        for (const proto::OpProto::Var& output : op_proto->outputs()) {
            const std::string& output_name = output.name();
            std::string outnum = "1";
            if(output.duplicable()) {
                outnum = output_name + "Num";

                const char* FWD_NUM_ARG_TEMPLATE = ", size_t %s";
                std::string arg_str = paddle::string::Sprintf(FWD_NUM_ARG_TEMPLATE, outnum);
                dygraph_function_args_str += arg_str;
            }
            const char* FWD_OUTS_CONTENT_TEMPLATE = "{ \"%s\", ConstructDuplicableOutput(%s) },";
            outs_contents_str += paddle::string::Sprintf(FWD_OUTS_CONTENT_TEMPLATE, output_name, outnum);
        }
        if(outs_contents_str.size() > 0)
            outs_contents_str.pop_back(); // Remove trailing ","

        const char* FWD_OUTS_MAP_TEMPLATE = "std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs = { %s };";
        std::string outs_map_str = paddle::string::Sprintf(FWD_OUTS_MAP_TEMPLATE, outs_contents_str);
        generated_function_body += outs_map_str;
        generated_function_body += "\n";
        
        // [Generation] Get TraceOp
        generated_function_body += "\n";
        const char* FWD_TRACE_OP_TEMPLATE = "tracer->TraceOp(\"%s\", ins, outs, attrs, tracer->ExpectedPlace(), false, {});";
        std::string trace_op_str = paddle::string::Sprintf(FWD_TRACE_OP_TEMPLATE, op_proto->type());
        generated_function_body += trace_op_str;
        generated_function_body += "\n";
        
        // [Generation] Convert output VarBase to Vector/Tensor
        generated_function_body += "\n";
        std::vector<std::string> return_contents(outs.size());
        std::vector<std::string> return_types(outs.size());
        for (const proto::OpProto::Var& output : op_proto->outputs()) {
            const std::string& output_name = output.name();
            std::string out_tensor_str;
            size_t return_position = fwd_outputs_name_pos_map[output_name];

            if(output.duplicable()) {
                const char* FWD_OUT_TENSORS_TEMPLATE = "std::vector<pt::Tensor> %s = VarBasesToTensors(outs[%s]);";
                out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSORS_TEMPLATE, output_name, output_name);
                return_types[return_position] = "std::vector<pt::Tensor>";
            } else {
                const char* FWD_OUT_TENSOR_TEMPLATE = "pt::Tensor %s = VarBasesToTensors(outs[%s])[0];";
                out_tensor_str = paddle::string::Sprintf(FWD_OUT_TENSOR_TEMPLATE, output_name, output_name);
                return_types[return_position] = "pt::Tensor";
            }

            return_contents[return_position] = output_name;
            generated_function_body += out_tensor_str;
            generated_function_body += "\n";
        }
        
        // [Generation] Get Return Tuple/Vector/Tensor
        generated_function_body += "\n";
        std::string return_str;
        std::string return_type_str = "";
        if(return_contents.size() > 1) {
            // Return tuple
            std::string return_content_str = "";
            for(const std::string& s : return_contents) {
                return_content_str += s + ",";
            }
            return_content_str.pop_back(); // Remove trailing ","
            
            for(const std::string& s : return_types) {
                return_type_str += s + ",";
            }
            return_type_str.pop_back(); // Remove trailing ","

            const char* FWD_TUPLE_RETURN_TEMPLATE = "return std::make_tuple<%s>(%s);";
            return_str = paddle::string::Sprintf(FWD_TUPLE_RETURN_TEMPLATE, return_type_str, return_content_str);
        } else {
            // Return vector<Tensor> or Tensor
            return_type_str = return_types[0];
            const char* FWD_TENSOR_RETURN_TEMPLATE = "return %s;";
            return_str = paddle::string::Sprintf(FWD_TENSOR_RETURN_TEMPLATE, return_contents[0]);
        }
        generated_function_body += return_str;
        generated_function_body += "\n";

        // [Generation] Get Full Function 
        std::string function_name = op_type + "_dygraph_function";

        const char* FWD_FUNCTION_TEMPLATE = "%s %s(%s) {\n %s \n}";
        std::string function_str = paddle::string::Sprintf(FWD_FUNCTION_TEMPLATE, return_type_str, function_name, dygraph_function_args_str, generated_function_body);

        VLOG(2) << function_str;
        
        }

        {
        /*
            // GradNode Example:
            vector<vector<Tensor>> GradNodeXXX::operator()(vector<vector<Tensor>>& grads) {
                
                const std::shared_ptr<Tracer>& tracer = imperative::GetCurrentTracer();
                
                // Comes from "grad_ins"
                std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins = 
                        {"X" : this->"X", "Y" : this->"Y", 
                         "Out0@Grad": TensorsToVarBases(grads["fwd_outputs_name_pos_map[grad_ins_grad_slotname_map["Out0@Grad"]]"]),
                         "Out1@Grad": TensorsToVarBases(grads["fwd_outputs_name_pos_map[grad_ins_grad_slotname_map["Out1@Grad"]]"]) 
                         };
                
                // Comes from "grad_outs"
                std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs = 
                        {"X@Grad" : ConstructDuplicableOutput(this->in_ranks["fwd_inputs_name_pos_map[grad_outs_slotname_map["X@Grad"]]"]), 
                         "Y@Grad" : ConstructDuplicableOutput(this->in_ranks["fwd_inputs_name_pos_map[grad_outs_slotname_map["Y@Grad"]]"]) };
                
                // GradNode's ins/outs/attrs are superclass to each OpBase's ins/outs/attrs
                // Visit each OpBase
                for(auto iter = "grad_node->begin()"; iter < "grad_node->end()"; iter++) {
                    framework::AttributeMap attrs;
                    for("auto& kv : iter->Attrs()") {
                        attrs[kv.first] = this->"kv.first";
                    }
                    for(auto& kv : "iter->DefaultAttrsMap()") {
                        attrs[kv.first] = this->"kv.first";
                    }
                    tracer->TraceOp("iter->Type()", ins, outs, attrs, tracer->ExpectedPlace(), false, {});
                }

                vector<vector<Tensor>> outputs(outs.size());
                for(auto& kv : outs) {
                    outputs["fwd_inputs_name_pos_map[grad_outs_slotname_map[kv.first]]"] = VarBasesToTensors(outs["kv.first"]);
                }

                return outputs;
            }
        */
        
        /* ---------------------------------- */
        /* --------- CodeGen: Backward ------ */
        /* ---------------------------------- */
        
        /* ------ Dygraph GradNode generation ------ */
        std::string generated_grad_function_body = "";
        
        // [Generation] Get Tracer
        generated_grad_function_body += "\n";
        std::string tracer_str = "const std::shared_ptr<Tracer>& tracer = imperative::GetCurrentTracer();";
        generated_grad_function_body += tracer_str;
        generated_grad_function_body += "\n";
        
        // [Generation] Get Ins Map
        generated_grad_function_body += "\n";
        std::string ins_contents_str = "";
        for (auto iter : grad_ins) {
            const std::string& grad_input_name = iter.first;
            
            if(grad_ins_fwd_slotname_map.count(grad_input_name)) {
                // Fwd Tensor
                const std::string& fwd_input_name = grad_ins_fwd_slotname_map[grad_input_name];
                const char* GRAD_INS_FWD_CONTENT_TEMPLATE = "{ \"%s\", this->%s },";
                ins_contents_str += paddle::string::Sprintf(GRAD_INS_FWD_CONTENT_TEMPLATE, grad_input_name, fwd_input_name);
            
            } else if(grad_ins_grad_slotname_map.count(grad_input_name)) {
                // Fwd Tensor's Grad
                size_t fwd_output_position = fwd_outputs_name_pos_map[grad_ins_grad_slotname_map[grad_input_name]];
                const char* GRAD_INS_GRAD_CONTENT_TEMPLATE = "{ \"%s\", TensorsToVarBases(grads[%d]) },";
                ins_contents_str += paddle::string::Sprintf(GRAD_INS_GRAD_CONTENT_TEMPLATE, grad_input_name, fwd_output_position);
            
            } else {
                PADDLE_THROW(platform::errors::Fatal("Unable to find forward slot name that matches %s", grad_input_name)); 
            }
        }
        if(ins_contents_str.size() > 0)
            ins_contents_str.pop_back(); // // Remove trailing ","
        
        const char* BWD_INS_MAP_TEMPLATE = "std::map<std::string, std::vector<std::shared_ptr<VarBase>>> ins = { %s };";
        std::string ins_map_str = paddle::string::Sprintf(BWD_INS_MAP_TEMPLATE, ins_contents_str);
        generated_grad_function_body += ins_map_str;
        generated_grad_function_body += "\n";
        
        // [Generation] Get Outs Map
        generated_grad_function_body += "\n";
        std::string outs_contents_str = "";
        for (auto iter : grad_outs) {
            const std::string& grad_output_name = iter.first;
            
            if(grad_outs_slotname_map.count(grad_output_name)) {
                // Fwd Tensor
                size_t fwd_input_position = fwd_inputs_name_pos_map[grad_outs_slotname_map[grad_output_name]];
                const char* GRAD_OUTS_CONTENT_TEMPLATE = "{ \"%s\", ConstructDuplicableOutput(this->in_ranks[%d]) },";
                outs_contents_str += paddle::string::Sprintf(GRAD_OUTS_CONTENT_TEMPLATE, grad_output_name, fwd_input_position);
            } else {
                PADDLE_THROW(platform::errors::Fatal("Unable to find forward slot name that matches %s", grad_output_name)); 
            }
        }
        if(outs_contents_str.size() > 0)
            outs_contents_str.pop_back(); // // Remove trailing ","
        
        const char* BWD_OUTS_MAP_TEMPLATE = "std::map<std::string, std::vector<std::shared_ptr<VarBase>>> outs = { %s };";
        std::string outs_map_str = paddle::string::Sprintf(BWD_OUTS_MAP_TEMPLATE, outs_contents_str);
        generated_grad_function_body += outs_map_str;
        generated_grad_function_body += "\n";

        // [Generation] Get Attrs Map
        generated_grad_function_body += "\n";
        std::string trace_opbase_str = "";
        
        for(auto iter = grad_node->begin(); iter < grad_node->end(); iter++) {
            // Each OpBase
            OpBase& op_base = *iter;

            std::string attr_contents_str = "";
            for(auto& kv: op_base.DefaultAttrsMap()) {
                const std::string& attr_name = kv.first;
                const char* ATTR_CONTENT_TEMPLATE = "{ \"%s\", this->%s},";
                attr_contents_str += paddle::string::Sprintf(ATTR_CONTENT_TEMPLATE, attr_name, attr_name);
            }
            if(attr_contents_str.size() > 0)
                attr_contents_str.pop_back();
            
            const char* ATTRS_MAP_TEMPLATE = "framework::AttributeMap attrs = { %s };";
            std::string attrs_map_str = paddle::string::Sprintf(ATTRS_MAP_TEMPLATE, attr_contents_str);
            
            const char* TRACE_OP_TEMPLATE = "tracer->TraceOp(%s, ins, outs, attrs, tracer->ExpectedPlace(), false, {});";
            std::string trace_op_str = paddle::string::Sprintf(TRACE_OP_TEMPLATE, op_base.Type());

            trace_opbase_str += "\n";
            trace_opbase_str += attrs_map_str;
            trace_opbase_str += "\n";
            trace_opbase_str += trace_op_str;
        }
        
        generated_grad_function_body += trace_opbase_str;
        generated_grad_function_body += "\n";

        // [Generation] Get Return
        std::string outputs_str = "";
        for(auto iter : grad_outs) {
            const std::string& grad_out_name = iter.first;
            size_t fwd_input_position = fwd_inputs_name_pos_map[grad_outs_slotname_map[grad_out_name]];

            const char* BWD_OUTPUT_TEMPLATE = "outputs[%d] = VarBasesToTensors(outs[\"%s\"]);\n";
            outputs_str += paddle::string::Sprintf(BWD_OUTPUT_TEMPLATE, fwd_input_position, grad_out_name);
        }

        const char* BWD_RETURN_TEMPLATE = "std::vector<std::vector<pt::Tensor>> outputs(outs.size());\n %s \n return outputs;";
        std::string return_str = paddle::string::Sprintf(BWD_RETURN_TEMPLATE, outputs_str);

        generated_grad_function_body += "\n";
        generated_grad_function_body += return_str;

        // [Generation] Get Full Grad Function
        const char* GRAD_FUNCTION_TEMPLATE = "std::vector<std::vector<pt::Tensor>> GradNode%s::operator()(std::vector<std::vector<pt::Tensor>>& grads) {\n %s \n}";
        std::string grad_function_str = paddle::string::Sprintf(GRAD_FUNCTION_TEMPLATE, op_type, generated_grad_function_body);
        
        VLOG(2) << grad_function_str;
        
        }

        /* ---------------------------------- */
        /* --------- CodeGen: GradNode ------ */
        /* ---------------------------------- */
        /*
        const char* GRAD_NODE_TEMPLATE = "              \
            class GradNode%s : public GradNodeBase {\n  \
             public:\n                                  \
                GradNode%s() : GradNodeBase() {}\n      \
                ~GradNode%s() override = default;\n     \
                                                        \
                virtual std::vector<std::vector<pt::Tensor>> operator()(const std::vector<std::vector<pt::Tensor>>& grads) override;\n  \
                                                        \
                %s // SetX, SetY, ...                   \
                                                        \
                void SetAttributes(%s) {\n              \
                    %s                                  \
                }\n                                     \
                                                        \
             private:\n                                 \
                %s // Attributes                        \
                %s // TensorWrappers                    \
            };\n                                        \
            ";
        */  

    }

    return 0;
}
