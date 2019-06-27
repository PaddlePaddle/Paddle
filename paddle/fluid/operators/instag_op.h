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
        auto* x1 = context.Input<Tensor>("X1");
        // X2 is ins tag list
        // LoD [[0, Sum(ins1), Sum(ins1, ins2), ... ]]
        auto* x2 = context.Input<Tensor>("X2");
        // X3 is local fc tag list
        // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
        auto* x3 = context.Input<Tensor>("X3");

        // key: tag  value: fc list
        // e.g. [ tag0: [fc0, fc1], tag 1: [fc1]]
        std::unordered_map<int64_t, std::vector<int64_t>> tag_fc_map;

        // expected auto = const int64
        auto* x3_data = x3->data<int64_t>();
        auto x3_dims = x3->dims();
        size_t fc_cnt = x3_dims[0];
        {
            size_t fc_size = x3_dims[0];
            size_t tag_size = x3_dims[1];
            for (size_t i = 0; i < fc_size; i++) {
                for (size_t j = 0; j < tag_size; j++) {
                    int64_t tag_val = x3_data[i * tag_size + j];
                    if (tag_val != -1) {
                        tag_fc_map[tag_val].push_back(i);
                    }
                }
            }
        }

        // key: ins no   value: tag list
        // e.g. [ ins0: [tag0, tag1], ins1: [tag1]]
        std::unordered_map<int64_t, std::vector<int64_t>> ins_tag_map;

        // expected auto = const int64_t
        auto* x2_data = x2->data<int64_t>();
        auto x2_dims = x2->dims();
        size_t ins_cnt = x2_dims[0];
        {
            size_t ins_size = x2_dims[0];
            size_t tag_size = x2_dims[1];
            for (size_t i = 0; i < ins_size; i++) {
                for (size_t j = 0; j < tag_size; j++) {
                    int64_t tag_val = x2_data[i * tag_size + j];
                    if (tag_val != -1) {
                        ins_tag_map[i].push_back(tag_val);
                    }
                }
            }
        }


        // Compute ins for every fc
        // key: ins   value : fc list
        std::unordered_map<int64_t, std::vector<int64_t>> ins_fc_map;
        for (size_t i = 0; i < ins_tag_map.size(); i++) {
            for (size_t j = 0; j < ins_tag_map[i].size(); j++) {
                for (size_t k = 0;
                        k < tag_fc_map[ins_tag_map[i][j]].size(); k++) {
                    ins_fc_map[i].push_back(tag_fc_map[ins_tag_map[i][j]][k]);
                }
            }
        }

        // set output value
        // for those whose ins been dropout, set 0 for whole lines.
        // otherwise, copy whole line
        // Dim [local fc count, batch size, embedding size]
        Tensor* out = context.Output<Tensor>("Out");

        // expected auto = const double
        auto* x1_data = x1->data<double>();
        // expected auto = double
        auto* out_data = out->mutable_data<double>(context.GetPlace());

        size_t x1_batch_size = x1->dims()[0];
        size_t x1_embed_size = x1->dims()[1];

        std::vector<double> zeros(x1_embed_size, 0);

        for (size_t i = 0; i < ins_cnt; i++) {
            for (size_t j = 0; j < fc_cnt; j++) {
                // if ins in this fc
                if (std::find(std::begin(ins_fc_map[i]),
                            std::end(ins_fc_map[i]), (int64_t)j) !=
                        std::end(ins_fc_map[i])) {
                    memcpy(out_data + j * x1_batch_size * x1_embed_size +
                            i * x1_embed_size, x1_data + i * x1_embed_size,
                            x1_embed_size * sizeof(double));
                } else {
                    memcpy(out_data + j * x1_batch_size * x1_embed_size +
                            i * x1_embed_size, &zeros[0],
                            x1_embed_size * sizeof(double));
                }
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

        // expected auto = double
        auto *out_data = output->data<double>();

        // expected auto = double
        auto *output_grad_data = output_grad->data<double>();
        // expected auto = double
        auto *x1_grad_data = x1_grad->mutable_data<double>(context.GetPlace());

        auto output_dims = output_grad->dims();
        for (size_t i = 0; i < output_dims[1]; i++) {
            for (size_t j = 0; j < output_dims[2]; j++) {
                x1_grad_data[i * output_dims[2] + j] = 0;
                for (size_t k = 0; k < output_dims[0]; k++) {
                    if (out_data[k * output_dims[1] * output_dims[2] +
                            i * output_dims[2] + j] != 0) {
                        x1_grad_data[i * output_dims[2] + j] +=
                         output_grad_data[k * output_dims[1] * output_dims[2]
                         + i * output_dims[2] + j];
                    }
                }
            }
        }
    }
};
}  // namespace operators
}  // namespace paddle
