// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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
#pragma once

#include "paddle/phi/core/dense_tensor.h"
#include "paddle/fluid/framework/infershape_utils.h"
#include "paddle/utils/string/string_helper.h"
#include "paddle/phi/kernels/matmul_kernel.h"
#include "paddle/phi/kernels/transpose_kernel.h"
#include "paddle/phi/kernels/reduce_kernel.h"


namespace phi{
// check the validation of the Einsum equation.
// 1. the label must between 'a' - 'z'.
// 2. the dim of the same label must be same.
// 3. the broad cast dims in two operands is broadcastable.
// 4. there must exist '->' and the default output is complete in python.
// may be we can skip validation check in C++ and just put it in python.
inline static void ValidationCheck(const std::string&equation, /*{{{*/
    const std::vector<const DenseTensor*>&inputs) {
}/*}}}*/

enum LabelType{/*{{{*/
    Batch=1, // ABO
    Free,  // AO, BO
    Contraction, // AB
    Reduction, // A, B
};/*}}}*/

// map a label('a' - 'z') -> int, O(1) speed.
class LabelMap{/*{{{*/
constexpr static int N = 26 + 1; // 'a' - 'z' + '.', '.' is for broadcast dims
int default_value;
int map[N];
public: 
    LabelMap(int default_value=0){
        this->default_value = default_value;
        for (int i=0;i<N;++i) map[i] = default_value;
    }
    int & operator [](int label){
        int i = label - 'a';
        if (label == '.') i = N-1;
        return map[i];
    }
    int operator [](int label) const {
        int i = label - 'a';
        if (label == '.') i = N-1;
        return map[i];
    }
    // non-exist is present by is_default
    bool is_default(char label){
        return (*this)[int(label)] == default_value;
    }
};/*}}}*/

inline void print_label(const std::vector<char>&all_labels,/*{{{*/
                 const LabelMap&label2type){
    std::string str; 
    for (int a: all_labels){
      std::stringstream ss;
      ss << label2type[a];
      str += ss.str();
    }
    VLOG(5) << "Label: " + str ; 
}/*}}}*/

inline static void ReplaceEllipsis(std::string & s){/*{{{*/
    size_t pos;
    if ((pos = s.find("...", 0)) != std::string::npos){
        s.replace(pos, 3, ".");
    }
    // remove all the space in the expression
    while((pos = s.find(" ", 0)) != std::string::npos) {
        s.replace(pos, 1, "");
    }
}/*}}}*/

inline static void GlobalInfo(const std::vector<std::string>& op_labels,/*{{{*/
    const std::string& right, 
    LabelMap* label2type, 
    std::vector<char>* sorted_labels){
    // sorted_labels: ['.', <right>, <left only label>]
    
    std::vector<char>all;
    LabelMap counter(0);
    for (auto & ch: right){ // char
      int c = ch;
      (*label2type)[c] = LabelType::Free; 
    }

    for (auto & op: op_labels){
      for (auto & ch: op){ // char
        int c = ch;
        if (counter.is_default(c)) all.push_back(ch);
        counter[c] += 1;
        if ((*label2type)[c] != LabelType::Free && counter[c] == 2) (*label2type)[c] = LabelType::Contraction;
        else if (counter[c] == 2) (*label2type)[c] = LabelType::Batch;
      }
    }

    std::for_each(all.begin(), all.end(), [sorted_labels, label2type](int c){
      if ((*label2type)[c] == LabelType::Batch) sorted_labels->push_back(char(c)); 
    });
    std::for_each(all.begin(), all.end(), [sorted_labels, label2type](int c){
      if ((*label2type)[c] == LabelType::Free) sorted_labels->push_back(char(c)); 
    });
    std::for_each(all.begin(), all.end(), [sorted_labels, label2type](int c){
      if ((*label2type)[c] == LabelType::Contraction) sorted_labels->push_back(char(c)); 
    });
    std::for_each(all.begin(), all.end(), [&sorted_labels, label2type](int c){
      if ((*label2type)[c] == LabelType::Reduction) sorted_labels->push_back(char(c)); 
    });
}/*}}}*/

inline static void InferLabelShape(const std::vector<std::string>& op_labels,/*{{{*/
    const std::vector<DDim>&inputs,
    LabelMap*labelshape, std::vector<int>*broadcast_dims){
    // TODO: broad cast dim

    for (size_t i=0;i<op_labels.size();++i){
      auto & op_str = op_labels[i]; 
      auto & op_dim = inputs[i];
      PADDLE_ENFORCE_EQ(op_str.size(), op_dim.size(), phi::errors::InvalidArgument(""));
      for (size_t j=0; j<op_str.size(); ++j){
        int c = op_str[j]; 
        if (labelshape->is_default(c) || (*labelshape)[c] == -1) (*labelshape)[c] = op_dim[j];
        PADDLE_ENFORCE_EQ((*labelshape)[c], op_dim[j], phi::errors::InvalidArgument(""));
      }
    }
}
/*}}}*/

inline static void InferLabelPerm(/*{{{*/
    const std::string & op,
    int n_broadcast,
    LabelMap* label2perm){
    
    int cur = 0;
    for (int c: op){
      if (c == '.') {
        cur += n_broadcast;
        continue;
      }
      (*label2perm)[c] = cur; 
      cur += 1;
    }
}
//}}}

inline static void InferOutputDims(/*{{{*/
    const std::string& right,
    const std::vector<int>& broadcast_dims, 
    const LabelMap& labelshape,
    std::vector<int>* output_dims){
    
    for (int c: right){
      if (c == '.') 
        output_dims->insert(output_dims->end(), broadcast_dims.begin(), broadcast_dims.end());
      else
        output_dims->push_back(labelshape[c]);
    }
}
//}}}

inline static void ParseEinsumEquation(const std::string&equation, /*{{{*/
    const std::vector<DDim>&inputs,
    LabelMap*labelshape,
    LabelMap*labeltype,
    std::vector<char>* all_labels,
    std::vector<LabelMap>* label2perms, 
    std::vector<int>* broadcast_dims, 
    std::vector<int>* output_dims, 
    std::string* right) {

    auto results = paddle::string::split_string(equation, "->");
    auto left = results[0];
    ReplaceEllipsis(left);
    *right = results[1].substr(1);
    //VLOG(5) << "Einsum Infershape: right:" << right; 
    auto op_labels = paddle::string::split_string(left, ",");
    //VLOG(5) << "Einsum Infershape: op_labels:" << paddle::string::join_strings(op_labels, "\n");
    std::for_each(op_labels.begin(), op_labels.end(), ReplaceEllipsis);
    GlobalInfo(op_labels, *right, labeltype, all_labels);
    InferLabelShape(op_labels, inputs, labelshape, broadcast_dims);
    InferOutputDims(*right, *broadcast_dims, *labelshape, output_dims);
    for (size_t i=0;i < inputs.size(); ++i){
      InferLabelPerm(op_labels[i], broadcast_dims->size(), &((*label2perms)[i]));
    }
}/*}}}*/

inline void EinsumInferShape(const std::vector<MetaTensor*>&inputs,/*{{{*/
        const std::string& equation, 
        MetaTensor* out){

    // collect the following informations to prepare einsum.
    LabelMap labelshape(0); 
    LabelMap labeltype(LabelType::Reduction);
    std::vector<LabelMap>label2perms(inputs.size(), -1);
    std::vector<char>all_labels;
    std::vector<int>broadcast_dims;
    std::vector<int>output_dims; 

    std::vector<DDim>input_dims;
    for (auto & i: inputs){
      input_dims.push_back(i->dims());
    }
    std::string right;
    ParseEinsumEquation(equation, 
                        input_dims, 
                        &labelshape,
                        &labeltype,
                        &all_labels,
                        &label2perms,
                        &broadcast_dims, 
                        &output_dims, 
                        &right);

    VLOG(5) << "Einsum Infershape: input dims:" << paddle::string::join_strings(input_dims, "\n");
    VLOG(5) << "Einsum Infershape: equation:" << equation;
    //VLOG(5) << "Einsum Infershape: labeltype:" << labeltype;
    VLOG(5) << "Einsum Infershape: all_labels:" << paddle::string::join_strings(all_labels, ",");
    VLOG(5) << "Einsum Infershape: output dims:" << paddle::string::join_strings(output_dims, ",");
    print_label(all_labels, labeltype);
    out->set_dims(make_ddim(output_dims));
}/*}}}*/

inline std::vector<char> union_labels(const std::vector<char>& a, const std::vector<char>& b){/*{{{*/
    LabelMap counter(0); 
    std::vector<char>res; 
    auto f = [&](char c){
      if (counter[int(c)] == 0) res.push_back(c);
      counter[int(c)] += 1;
    };
    std::for_each(a.begin(), a.end(), f);
    std::for_each(b.begin(), b.end(), f);
    return res;
}/*}}}*/

template <typename T>
std::vector<T> GetLabelIndexByType(const std::vector<char>& all_labels,/*{{{*/
                const LabelMap& type, 
                const LabelMap& perm, 
                LabelType filter){
    std::vector<T> res;
    for (T c: all_labels){
      if (type[c] == filter && perm[c] != -1) res.push_back(perm[c]);
    } 
    return res;
}/*}}}*/

template <typename T>
std::vector<T> GetShapeByType(const std::vector<char>& all_labels,/*{{{*/
                const LabelMap& type, 
                const LabelMap& perm, 
                const LabelMap& label2shape,
                LabelType filter){
    std::vector<T> res;
    for (T c: all_labels){
      if (type[c] == filter && perm[c] != -1) res.push_back(label2shape[c]);
    }
    return res;
}/*}}}*/

template <typename T, typename Context>
DenseTensor PerformReduction(const Context& dev_ctx,/*{{{*/
                const DenseTensor& tensor, 
                const LabelMap& label2perm, 
                const std::vector<char>& all_labels,
                const LabelMap& label2type){
    auto indices = GetLabelIndexByType<int64_t>(all_labels, label2type, label2perm, LabelType::Reduction);
    VLOG(5)<<"call PerformReduction: with axis: " << paddle::string::join_strings(indices, ",");
    if (indices.size() == 0) return tensor;
    return Sum<T, Context>(dev_ctx, tensor, indices, tensor.dtype(), true);
}/*}}}*/

template <typename T, typename Context>
DenseTensor PerformTranspose(const Context& dev_ctx,/*{{{*/
                const DenseTensor& tensor, 
                const LabelMap& label2perm, 
                const std::vector<char>& all_labels,
                const LabelMap& label2type){
    std::vector<int>axis; 
    for (int c: all_labels){
      if (label2perm[c] != -1) {
        axis.push_back(label2perm[c]);
      }
    }
    auto ret = Transpose<T, Context>(dev_ctx, tensor, axis);
    VLOG(5) << "PerformTranspose: " << paddle::string::join_strings(axis, ","); 
    return ret;
}/*}}}*/

template <typename T, typename Context>
DenseTensor PerformContraction(const Context& dev_ctx,/*{{{*/
                const DenseTensor& A, 
                const DenseTensor& B, 
                const std::vector<LabelMap>& label2perm, 
                const std::vector<char>& all_labels,
                const LabelMap& label2type, const LabelMap& label2shape){
    auto batches = GetShapeByType<int>(all_labels, label2type, label2perm[0], label2shape, LabelType::Batch);
    auto recover_dim = batches;
    auto preprocess = [&](const DenseTensor& t, 
                          const LabelMap& perm) -> DenseTensor {
      auto frees = GetShapeByType<int>(all_labels, label2type, perm, label2shape, LabelType::Free);
      auto conts = GetShapeByType<int>(all_labels, label2type, perm, label2shape, LabelType::Contraction);
      auto trans_t = PerformTranspose<T, Context>(dev_ctx, t, perm, all_labels, label2type);
      std::vector<int>mul_dims = batches;
      recover_dim.insert(recover_dim.end(), frees.begin(), frees.end());
      mul_dims.push_back(std::accumulate(frees.begin(), frees.end(), 1, std::multiplies<int>()));
      mul_dims.push_back(std::accumulate(conts.begin(), conts.end(), 1, std::multiplies<int>()));
      VLOG(5) << "PerformContraction: mul_dims: " << paddle::string::join_strings(mul_dims, ",");
      trans_t.Resize(make_ddim(mul_dims));
      return trans_t;
    };
    auto trans_a = preprocess(A, label2perm[0]);
    auto trans_b = preprocess(B, label2perm[1]);
    auto after_contraction = Matmul<T, Context>(dev_ctx, trans_a, trans_b, false, true);
    VLOG(5) << "PerformContraction: recover_dim: " << paddle::string::join_strings(recover_dim, ",");
    after_contraction.Resize(make_ddim(recover_dim));
    return after_contraction;
}

//}}}

template <typename T, typename Context>
void TransposeToOutput(const Context& dev_ctx,/*{{{*/
                const DenseTensor& to_trans, 
                const std::string& right,
                const std::vector<char>& all_labels, 
                DenseTensor* output){
    std::vector<int>axis; 
    for (char c: right){
      auto it = std::find(all_labels.begin(), all_labels.end(), c);
      PADDLE_ENFORCE_NE(it, all_labels.end(), phi::errors::InvalidArgument(
          "Must in all_labels."));
      axis.push_back(it - all_labels.begin());
    }
    VLOG(5)<<"call TransposeToOutput: with axis: " << paddle::string::join_strings(axis, ",");
    return TransposeKernel<T, Context>(dev_ctx, to_trans, axis, output);
}/*}}}*/

template <typename T, typename Context>
void EinsumKernel(const Context& dev_ctx,
                const std::vector<const DenseTensor*>& inputs,
                const std::string& equation,
                DenseTensor* out){
    //ValidationCheck(inputs, equation);
    // collect the following informations to prepare einsum.
    LabelMap labelshape(0); 
    LabelMap labeltype(LabelType::Reduction);
    std::vector<LabelMap>label2perms(inputs.size(), -1);
    std::vector<char>all_labels; // order: ABO, AO, BO, AB, Reduce
    std::vector<int>broadcast_dims;
    std::vector<int>output_dims; 

    std::vector<DDim>input_dims;
    for (auto & i: inputs){
      input_dims.push_back(i->dims());
    }
    std::string right;
    ParseEinsumEquation(equation, 
                        input_dims, 
                        &labelshape,
                        &labeltype,
                        &all_labels,
                        &label2perms,
                        &broadcast_dims, 
                        &output_dims, 
                        &right);
    //VLOG(5) << "Einsum: ParseEinsumEquation done. with dims: " << broadcast_dims.size(); 
    out->Resize(make_ddim(output_dims));
    if (inputs.size() == 2) {
        auto& A = inputs[0];
        auto& B = inputs[1];
        // Reduce Procedure
        auto reduce_A = PerformReduction<T, Context>(dev_ctx, *A, label2perms[0], all_labels, labeltype);
        auto reduce_B = PerformReduction<T, Context>(dev_ctx, *B, label2perms[1], all_labels, labeltype);
        // Contract Procedure
        dev_ctx.template Alloc<T>(out);
        auto after_contraction = PerformContraction<T, Context>(dev_ctx, reduce_A, reduce_B, label2perms, all_labels, labeltype, labelshape);
        TransposeToOutput<T, Context>(dev_ctx, after_contraction, right, all_labels, out);
        // Reshape Procedure
    } else if (inputs.size() == 1){
        auto reduce_A = PerformReduction<T, Context>(dev_ctx, *inputs[0], label2perms[0], all_labels, labeltype);
        std::vector<char> right_labels; 
        for (auto c: right) right_labels.push_back(c);
        right_labels = union_labels(right_labels, all_labels);
        *out = PerformTranspose<T, Context>(dev_ctx, reduce_A, label2perms[0], right_labels, labeltype);
        out->Resize(make_ddim(output_dims));
    } else {
        PADDLE_THROW(phi::errors::InvalidArgument(
            "EinsumOp kernel only support len(operands) between (0, 2]. Use opt_einsum first to convert multi-variable to binary-variable."));
    }
}

} // namespace phi
