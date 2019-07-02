// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include <cstring>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/memory/memcpy.h"
#include "paddle/fluid/platform/assert.h"

namespace paddle {
namespace operators {
using Tensor = framework::Tensor;
using SelectedRows = framework::SelectedRows;
using LoDTensor = framework::LoDTensor;

template <typename T>
class InstagKernel : public framework::OpKernel<T> {
 public:
    void Compute(const framework::ExecutionContext& context) const override {
        // X1 is global FC output
        // Dim [batch size, embedding size]
        auto* x1 = context.Input<LoDTensor>("X1");
        // X2 is ins tag list
        // LoD [[0, Sum(ins1), Sum(ins1, ins2), ... ]]
        auto* x2 = context.Input<LoDTensor>("X2");
        // X3 is local fc tag list
        // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
        auto* x3 = context.Input<Tensor>("X3");

        std::unordered_set<int64_t> filter_tag;
        auto* x3_data = x3->data<int64_t>();
        size_t len = x3->dims()[0];
        for (size_t i = 0; i < len; i++) {
            filter_tag.insert(x3_data[i]);
        }


        // expected auto = const int64_t
        auto* x2_data = x2->data<int64_t>();
        // e.g get [0, 1, 2, 3, ...]
        auto x2_lods = x2->lod()[0];
        
        std::vector<size_t> ins_after_filter;
        for (size_t i = 0; i < x2_lods.size() -1 ; i++) {
            for(size_t j = x2_lods[i]; j < x2_lods[i+1]; j++) {
                if (filter_tag.find(x2_data[j]) != filter_tag.end()) {
                    ins_after_filter.push_back(i);
                    break;
                }
            }
        }


        // set output value
        // for those whose ins been dropout, set 0 for whole lines.
        // otherwise, copy whole line
        // Dim [local fc count, batch size, embedding size]
        LoDTensor* out = context.Output<LoDTensor>("Out");

        // expected auto = const T
        auto* x1_data = x1->data<T>();
        // expected auto = T
        auto* out_data = out->mutable_data<T>(context.GetPlace());

        size_t x1_embed_size = x1->dims()[1];
        auto x1_lods = x1->lod()[0];
        out->set_lod(x1->lod());
        memset(out_data, 0, x1->dims()[0] * x1->dims()[1] * sizeof(T));
        for (size_t i = 0; i < ins_after_filter.size(); i++) {
            for (size_t k = x1_lods[ins_after_filter[i]]
                    ; k < x1_lods[ins_after_filter[i] + 1]; k++) {
                memcpy(out_data + 
                    k * x1_embed_size, x1_data + k * x1_embed_size,
                    x1_embed_size * sizeof(T));
            }
        }
    }
};

template <typename T>
class InstagGradKernel : public framework::OpKernel<T> {
 public:
    void Compute(const framework::ExecutionContext& context) const override {
        auto *output = context.Input<Tensor>("Out");
        auto *output_grad = context.Input<Tensor>
            (framework::GradVarName("Out"));
        auto *x1_grad = context.Output<Tensor>(framework::GradVarName("X1"));

        // expected auto = T
        auto *out_data = output->data<T>();

        // expected auto = T
        auto *output_grad_data = output_grad->data<T>();
        // expected auto = T
        auto *x1_grad_data = x1_grad->mutable_data<T>(context.GetPlace());

        auto output_dims = output_grad->dims();
        for (size_t i = 0; i < output_dims[0]; i++) {
            for (size_t j = 0; j < output_dims[1]; j++) {
                x1_grad_data[i * output_dims[1] + j] = 0;
                if (out_data[i * output_dims[1] + j] != 0) {
                    x1_grad_data[i * output_dims[1] + j] +=
                     output_grad_data[i * output_dims[1] + j];
                }
            }
        }
    }
};
}  // namespace operators
}  // namespace paddle
