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

#pragma once

#include <random>
#include <string>
#include <vector>

#include "paddle/fluid/framework/eigen.h"
#include "paddle/fluid/framework/lod_tensor.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/selected_rows.h"
#include "paddle/fluid/operators/jit/kernels.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/math/bloomfilter.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using LoDTensor = framework::LoDTensor;
using SelectedRows = framework::SelectedRows;
using DDim = framework::DDim;

template <typename T>
struct RandomEmbeddingFunctor {
  void operator()(const framework::ExecutionContext &context,
                  const LoDTensor *table_t, const LoDTensor *ids_t, const int rand_len,
                  LoDTensor *output_t) {
    auto *table = table_t->data<T>();
    int64_t row_number = table_t->dims()[0];
    int64_t row_width = rand_len;
    int64_t last_dim = output_t->dims()[1];
    const int64_t *ids = ids_t->data<int64_t>();
    auto ids_lod = ids_t->lod()[0];
    auto *output = output_t->mutable_data<T>(context.GetPlace());
    auto& dev_ctx =
            context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> set_zero;
    set_zero(dev_ctx, reinterpret_cast<Tensor *>(output_t), static_cast<T>(0));
    if (ids_lod.back() == 0 || ids_t->numel() == 0 || ids_t->lod().size() == 0) {
        return;
    }
    int64_t ids_count = ids_t->numel() / ids_lod.back();

    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
    for (int64_t i = 0; i != ids_lod.back(); ++i) {
      size_t begin = i * ids_count;
      for (int64_t j = 0; j != ids_count; ++j) {
        PADDLE_ENFORCE_LT(ids[begin], row_number);
        PADDLE_ENFORCE_GE(ids[begin], 0, "ids %d", i);
        blas.VCOPY(row_width, table + ids[begin + j],
                   output + i * last_dim + j * row_width);
      }
    }
  }
};

bool ShouldUseSeq(const int64_t* word_repr, const int len, const Tensor * filter, const Tensor* black_filter){
    if ((filter && 0 == bloomfilter_get(reinterpret_cast<const math::bloomfilter*>(filter), word_repr, len * sizeof(int64_t))) ||
           (black_filter && 1 == bloomfilter_get(reinterpret_cast<const math::bloomfilter*>(black_filter), word_repr, len * sizeof(int64_t)))) {
      return false;
    }
    return true;
}

template <typename T>
class SequencePyramidEmbeddingKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    const LoDTensor *ids_t = context.Input<LoDTensor>("Ids");  // int tensor
    LoDTensor *output_t = context.Output<LoDTensor>("Out");    // float tensor
    LoDTensor *hash_ids_t = context.Output<LoDTensor>("HashIds");    // int tensor
    const LoDTensor *table_var = context.Input<LoDTensor>("W");
    const int rand_len = context.Attr<int>("rand_len");
    const int num_hash = context.Attr<int>("num_hash");
    const int mod_by = context.Attr<int>("mod_by");
    const int dropout_rate = context.Attr<float>("dropout_rate");

    const Tensor *filter = context.Input<Tensor>("Filter");  // int tensor
    const Tensor *black_filter = context.Input<Tensor>("BlackFilter");  // int tensor
    const int white_list_len = context.Attr<int>("white_list_len");
    const int black_list_len = context.Attr<int>("black_list_len");

    if (white_list_len > 0) {
      PADDLE_ENFORCE_NOT_NULL(filter, "Filter connot be null");
    }
    if (black_list_len > 0) {
      PADDLE_ENFORCE_NOT_NULL(black_filter, "BlackFilter connot be null");
    }
    
    const auto &ids_lod = ids_t->lod();
    // in run time, the LoD of ids must be 1
    PADDLE_ENFORCE(ids_lod.size(), 1UL,
                   "The LoD level of Input(Ids) must be 1");
    PADDLE_ENFORCE(table_var->dims()[1], 1UL,
                   "The last dim of Parameter W must be 1");
    // in run time, the shape from Ids -> output
    int min_win_size = context.Attr<int>("min_win_size");
    int max_win_size = context.Attr<int>("max_win_size");
    
    // NOTE: fixed seed should only be used in unittest or for debug.
    // Guarantee to use random seed in training. 
    std::random_device rnd;
    std::minstd_rand engine;
    int seed =
        context.Attr<bool>("fix_seed") ? context.Attr<int>("seed") : rnd();
    engine.seed(seed);
    std::uniform_real_distribution<float> dist(0, 1);
 
    // Generate enumerate sequence set
    auto lod0 = ids_lod[0];
    auto ids_data = ids_t->data<int64_t>();
    std::vector<std::vector<std::vector<int64_t>>> seq_enums;
    for (size_t i = 0; i < lod0.size() - 1; ++i) {
      size_t inst_start = lod0[i];
      std::vector<std::vector<int64_t>> seq_enum;
      for (int win_size = min_win_size; win_size <= max_win_size; win_size++) {
        int enumerate_shift = win_size - 1;
        if (lod0[i + 1] - lod0[i] > enumerate_shift) {
          size_t len = lod0[i + 1] - lod0[i] - enumerate_shift;
          for (int j = 0; j < len;j++) {
            size_t win_start = inst_start + j; //window start
            std::vector<int64_t> seq;
            for (int word_idx = 0; word_idx < win_size; ++word_idx) {
              size_t word_pos = win_start + word_idx; //sub sequence
              seq.push_back(ids_data[word_pos]);
            }
            if (dist(engine) < 1.0f - dropout_rate) {
              if (ShouldUseSeq(&seq[0], seq.size(), filter, black_filter)) {
                seq_enum.push_back(seq);
              }
            }
          }
        }
      }
      seq_enums.push_back(seq_enum);
    }
    framework::LoD new_lod;
    new_lod.emplace_back(1, 0);  // size = 1, value = 0;
    int offset = 0;
    auto new_lod0 = new_lod[0];
    for (size_t i = 1; i < lod0.size(); ++i) {
      offset = offset + seq_enums[i-1].size();
      new_lod0.push_back(offset);
    }
    new_lod[0] = new_lod0;
    output_t->Resize({offset, num_hash * rand_len});
    output_t->set_lod(new_lod);
    hash_ids_t->Resize({offset, num_hash});
    auto hash_ids_data = hash_ids_t->mutable_data<int64_t>(context.GetPlace());
    hash_ids_t->set_lod(new_lod);

    //hash sub sequence
    float *buffer = new float[max_win_size]; //to align with lego, bottom use float data
    for (int i = 0;i < seq_enums.size();i++) {
      for (int j = 0;j < seq_enums[i].size();j++) {
        memset(buffer, 0, max_win_size * sizeof(float));
        int win_size = seq_enums[i][j].size();
        PADDLE_ENFORCE_GE(win_size, 0, "sub sequence [%d][%d] cannot be empty ", i, j);
        for (int k = 0;k < win_size;++k) {
          buffer[k] = static_cast<float>(seq_enums[i][j][k]);
          //VLOG(1) << "qxz buffer[" << k << "]=" << buffer[k];
        }
        int idx = new_lod0[i] + j; 
        for (int ihash = 0; ihash != num_hash; ++ihash) {
          hash_ids_data[idx * num_hash + ihash] = 
              XXH32(buffer, sizeof(float) * win_size, ihash * rand_len) % mod_by;
              //VLOG(1) << "qxz hash [" << ihash << "]=" << hash_ids_data[idx * num_hash + ihash];
        }
      }
    }

    //random lookup embedding table
    RandomEmbeddingFunctor<T> functor;
    functor(context, table_var, hash_ids_t, rand_len, output_t);

  }
};

template <typename T>
class SequencePyramidEmbeddingGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &context) const override {
    auto *table_var = context.InputVar("W");
    DDim table_dim;
    if (table_var->IsType<LoDTensor>()) {
      table_dim = context.Input<LoDTensor>("W")->dims();
    } else {
      PADDLE_THROW(
          "The parameter W of a LookupTable "
          "must be LoDTensor");
    }
    PADDLE_ENFORCE(table_dim[1], 1UL,
                   "The last dim of Parameter W must be 1");

    bool is_sparse = context.Attr<bool>("is_sparse");
    // Since paddings are not trainable and fixed in forward, the gradient of
    // paddings makes no sense and we don't deal with it in backward.
    if (is_sparse) {
      auto *ids = context.Input<LoDTensor>("HashIds");
      auto *d_output = context.Input<LoDTensor>(framework::GradVarName("Out"));
      auto *d_table = context.Output<SelectedRows>(framework::GradVarName("W"));
      const int rand_len = context.Attr<int>("rand_len");

      auto *ids_data = ids->data<int64_t>();
      int64_t ids_num = ids->numel();
      auto lod = ids->lod()[0];
     
      int64_t new_ids_num = ids_num * rand_len;
      framework::Vector<int64_t> *new_rows = d_table->mutable_rows();
      int64_t row_number = table_dim[0];
      new_rows->resize(new_ids_num);
      int64_t new_idx = 0;
      for (int i = 0; i < new_ids_num; ++i) {
         new_idx = ids_data[i / rand_len] + i % rand_len; 
         PADDLE_ENFORCE_LT(new_idx, row_number);
         if (new_idx == 188574148) {
           VLOG(1) << "qxz new_idx == 188574148, i = " << i << ", id = " << ids_data[i / rand_len];
         }
         (*new_rows)[i] = new_idx; 
      }

      auto *d_table_value = d_table->mutable_value();
      d_table_value->Resize({new_ids_num, 1});
      T *d_table_data = d_table_value->mutable_data<T>(context.GetPlace());
      const T *d_output_data = d_output->data<T>();
     
      auto blas = math::GetBlas<platform::CPUDeviceContext, T>(context);
      blas.VCOPY(new_ids_num, d_output_data, d_table_data);
    } else {
      LOG(ERROR) << "Dense is not supported in sequence_pyramid_embedding_op now";
    }
  }
};

}  // namespace operators
}  // namespace paddle
