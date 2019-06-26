#pragma once

#include <cstring>
#include <random>
#include <string>
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
            auto* x2 = context.Input<LoDTensor>("X2");
            // X3 is local fc tag list
            // LoD [[0, Sum(fc1), Sum(fc1, fc2) ...]]
            auto* x3 = context.Input<LoDTensor>("X3");
            
            // key: tag  value: fc list
            // e.g. [ tag0: [fc0, fc1], tag 1: [fc1]]
            std::unordered_map<int64_t, std::vector<int64_t>> tag_fc_map;
            
            // expected auto = const int64
            auto* x3_data = x3->data<int64_t>(); 
            size_t fc_cnt = 0; //count of local fc

            for (size_t i = 0; i < x3->lod()[0].size() -1; i++) {
                for (size_t j = x3->lod()[0][i]; j < x3->lod()[0][i+1]; j++) {
                    int64_t tag_val = x3_data[j];
                    tag_fc_map[tag_val].push_back(fc_cnt);
                }
                ++fc_cnt;
            }
        
            // key: ins no   value: tag list
            // e.g. [ ins0: [tag0, tag1], ins1: [tag1]]
            std::unordered_map<int64_t, std::vector<int64_t>> ins_tag_map;
            
            // expected auto = const int64_t
            auto* x2_data = x2->data<int64_t>();
            size_t ins_cnt = 0;

            for (size_t i = 0; i < x2->lod()[0].size()-1; i++) {
                for (size_t j = x2->lod()[0][i]; j < x2->lod()[0][i+1]; j++) {
                    int64_t tag_val = x2_data[j];
                    ins_tag_map[ins_cnt].push_back(tag_val);
                }
                ++ins_cnt;
            }

            // Compute ins for every fc
            // key: ins   value : fc list
            std::unordered_map<int64_t, std::vector<int64_t>> ins_fc_map;
            for (size_t i = 0; i < ins_cnt; i++) {
                for (size_t j = 0; j < ins_tag_map[j].size(); j++) {
                    for (size_t k = 0; k < tag_fc_map[ins_tag_map[j][i]].size(); k++) {
                       ins_fc_map[i].push_back(tag_fc_map[ins_tag_map[j][i]][k]); 
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
                    if (std::find(std::begin(ins_fc_map[i]), std::end(ins_fc_map[i]), (int64_t)j) != std::end(ins_fc_map[i])) {
                        memcpy(out_data + j * x1_batch_size * x1_embed_size + i * x1_embed_size, 
                                x1_data + i * x1_embed_size, x1_embed_size * sizeof(double));
                    } 
                    // else copy all 0
                    else {
                        memcpy(out_data + j * x1_batch_size * x1_embed_size + i * x1_embed_size,
                                &zeros[0], x1_embed_size * sizeof(double));
                    }
                }
            }

        }
    };

    template <typename T>
    class InstagGradKernel : public framework::OpKernel<T> {
    public:
        void Compute(const framework::ExecutionContext& context) const override {
            auto *output_grad = context.Input<Tensor>(framework::GradVarName("Out"));
            auto *x1_grad = context.Output<Tensor>(framework::GradVarName("X1"));

	    // expected auto = double	
            auto *output_grad_data = output_grad->data<double>();
	    // expected auto = double
            auto *x1_grad_data = x1_grad->mutable_data<double>(context.GetPlace());


            auto output_dims = output_grad->dims();
            for (size_t i = 0; i < output_dims[1]; i++) {
                for (size_t j = 0; j < output_dims[2]; j++) {
                    for (size_t k = 0; k < output_dims[0]; k++) {
                        x1_grad_data[i * output_dims[2] + j] +=
                            output_grad_data[k * output_dims[1] * output_dims[2] + i * output_dims[2] + j];
                    }
                }
            }
        }
    };
  } // namespace operators
} // namespace paddle
