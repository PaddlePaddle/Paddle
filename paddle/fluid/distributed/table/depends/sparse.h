// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <math.h>  // for sqrt in CPU and CUDA
#include <functional>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>
#include "gflags/gflags.h"

#include "paddle/fluid/distributed/common/utils.h"
#include "paddle/fluid/distributed/table/depends/large_scale_kv.h"

namespace paddle {
namespace distributed {

enum SaveMode { all, base, delta };

class SparseOptimizer {
 public:
  explicit SparseOptimizer(
      const std::vector<std::string>& value_names,
      const std::vector<int>& value_dims, const std::vector<int>& value_offsets,
      const std::unordered_map<std::string, int>& value_idx)
      : value_names_(value_names),
        value_dims_(value_dims),
        value_offsets_(value_offsets),
        value_idx_(value_idx) {}

  virtual void update(const uint64_t* keys, const float* update_values,
                      size_t num, const std::vector<uint64_t>& offsets,
                      ValueBlock* block) = 0;

  virtual void set_global_lr(float* lr) { global_learning_rate_ = lr; }

  virtual int64_t save(const int mode,
                    const int shard_idx,
                    const std::string prefix,
                    const std::vector<std::shared_ptr<ValueBlock>>& shard_values) = 0;

  int64_t SaveToBin(const int mode,
                     const std::vector<std::string>& var_names,
                     const std::vector<int>& offsets,
                     const std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    auto place = platform::CPUPlace();
    platform::DeviceContextPool &pool = platform::DeviceContextPool::Instance();
    auto &dev_ctx = *pool.Get(place);

    int64_t ids_num = 0;
    for (auto &block : shard_values) {
      auto count = block->GetEntryCount();
      ids_num += count;
    }

    std::vector<std::shared_ptr<framework::Variable>> vars;
    std::vector<float *> tensors;
    std::vector<int64_t> ids;

    for(auto i = 0; i < var_names.size(); ++i) {
      auto var = std::make_shared<framework::Variable>();
      auto *slr = var->GetMutable<framework::SelectedRows>();
      auto *src_t = slr->mutable_value();

      src_t->Resize({ids_num, update_numel});
      auto *value = src_t->mutable_data<float>(place);

      vars.push_back(var);
      tensors.push_back(value);
    }

    int64_t not_save_num = 0;
    auto param_offset = 0;
    for(auto& block : shard_values) {
      for(auto value: block->values_ ) {
        if (mode == SaveMode::delta && !value.second->need_save_) {
          not_save_num++;
          continue;
        }
        if (!value.second->is_entry_) {
          continue;
        }
        ids.emplace_back(value.first);
        for(auto i = 0; i < var_names.size(); ++i) {
          auto offset = offsets[i];
          std::copy_n(value.second->data_.data() + offset, update_numel,
                      tensors[i] + param_offset * update_numel);
        }
        param_offset += 1;
      }
    }
    for(auto &var : vars) {
      auto *slr = var->GetMutable<framework::SelectedRows>();
      slr->set_rows(ids);
      slr->set_height(ids.size());
    }
    for (int i = 0; i < static_cast<int>(var_names.size()); ++i) {
      auto &filename = var_names[i];
      auto &selectedRows = vars[i]->Get<framework::SelectedRows>();

      std::ofstream fout(filename, std::ios::binary);
      PADDLE_ENFORCE_EQ(static_cast<bool>(fout), true,
                        platform::errors::Unavailable(
                            "Cannot open %s to save variables.", filename));

      framework::SerializeToStream(fout, selectedRows, dev_ctx);
      fout.close();
    } 
    return ids_num - not_save_num;
  }
  
  virtual void load(const int shard_idx,
                    const std::string prefix,
                    std::vector<std::shared_ptr<ValueBlock>>& shard_values) = 0;  

  void LoadFromBin(const std::vector<std::string>& filenames,
                   const std::vector<int>& offsets,
                   std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::shared_ptr<framework::Variable>> variables;
    auto place = platform::CPUPlace();

    for (int i = 0; i < static_cast<int>(filenames.size()); ++i) {
      auto var = std::make_shared<framework::Variable>();
      variables.push_back(var);
      auto &filename = filenames[i];
      std::ifstream fin(filename, std::ios::binary);
      auto *selectedRows = var->GetMutable<framework::SelectedRows>();

      platform::DeviceContextPool &pool =
          platform::DeviceContextPool::Instance();
      auto &dev_ctx = *pool.Get(place);

      framework::DeserializeFromStream(fin, selectedRows, dev_ctx);
    }

    std::vector<const float *> tensors;

    for (int i = 0; i < static_cast<int>(filenames.size()); ++i) {
      auto &slr = variables[i]->Get<framework::SelectedRows>();
      auto src_t = slr.value();
      const auto *value = src_t.data<float>();
      tensors.push_back(value);
    }

    auto rows = variables[0]->Get<framework::SelectedRows>().rows();
    if(variables.size() > 1) {
      for(auto i = 1; i < static_cast<int64_t>(variables.size()); ++i) {
        auto num = variables[i]->Get<framework::SelectedRows>().rows().size();
          PADDLE_ENFORCE_EQ(num, rows.size(),
                            platform::errors::InvalidArgument("rows' num me be equal to %s", rows.size()));
      }
    }

    auto shard_num = shard_values.size();
    std::vector<uint64_t> shard_counts(shard_num, 0);
    auto param_offset = 0;
    for (auto i = 0; i < static_cast<int64_t>(rows.size()); ++i) {
      auto id = rows[i];
      auto shard_id = id % shard_num;
      shard_counts[shard_id]++;
      auto block = shard_values[shard_id];
      auto value = block->Init(id);
      value->initialize();

      // copy param and optimizer param.
      for(auto j = 0; j < static_cast<int64_t>(offsets.size()); ++j) {
        auto offset = offsets[j];
        std::copy_n(tensors[j] + param_offset * update_numel, update_numel, value->data_.data() + offset);
      }
      param_offset++;
    }
    for(auto i = 0; i < shard_values.size(); ++i) {
      shard_values[i]->SetEntryCount(shard_counts[i]);
    }
  }

  const std::vector<std::string>& value_names_;
  const std::vector<int>& value_dims_;
  const std::vector<int>& value_offsets_;
  const std::unordered_map<std::string, int>& value_idx_;
  int param_offset = 0;
  int update_numel = 0;

 protected:
  float* global_learning_rate_;
};

// sum calc for sparse tensor
class SSUM : public SparseOptimizer {
 public:
  explicit SSUM(const std::vector<std::string>& value_names,
                const std::vector<int>& value_dims,
                const std::vector<int>& value_offsets,
                const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* value = block->Get(id);
      float* param = value + param_offset;
      blas.VADD(update_numel, update_values + x * update_numel, param, param);
    }
  }

  int64_t save(const int mode,
               const int shard_idx,
               const std::string prefix,
               const std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset};
    return SaveToBin(mode, var_names, offsets, shard_values);
  }

  void load(const int shard_idx,
            const std::string prefix,
            std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset};
    LoadFromBin(var_names, offsets, shard_values);
  }
};

// sgd optimzer for sparse tensor
class SSGD : public SparseOptimizer {
 public:
  explicit SSGD(const std::vector<std::string>& value_names,
                const std::vector<int>& value_dims,
                const std::vector<int>& value_offsets,
                const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    idx = value_idx.at("LearningRate");
    lr_offset = value_offsets.at(idx);
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* value = block->Get(id);

      float learning_rate = *(global_learning_rate_) * (value + lr_offset)[0];
      VLOG(4) << "SSGD LearningRate: " << learning_rate;
      float* param = value + param_offset;

      std::vector<float> grads;
      grads.resize(update_numel);
      blas.VCOPY(update_numel, update_values + x * update_numel, grads.data());
      blas.SCAL(update_numel, learning_rate, grads.data());
      blas.VSUB(update_numel, param, grads.data(), param);
    }
  }

  int64_t save(const int mode,
               const int shard_idx,
               const std::string prefix,
               const std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset};
    return SaveToBin(mode, var_names, offsets, shard_values);
  }

  void load(const int shard_idx,
            const std::string prefix,
            std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset};
    LoadFromBin(var_names, offsets, shard_values);
  }

  int lr_offset;
};

// adam optimzer for sparse tensor
class SAdam : public SparseOptimizer {
 public:
  explicit SAdam(const std::vector<std::string>& value_names,
                 const std::vector<int>& value_dims,
                 const std::vector<int>& value_offsets,
                 const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    idx = value_idx.at("LearningRate");
    lr_offset = value_offsets.at(idx);

    idx = value_idx.at("Moment1");
    m1_offset = value_offsets.at(idx);

    idx = value_idx.at("Moment2");
    m2_offset = value_offsets.at(idx);

    idx = value_idx.at("Beta1Pow");
    beta1_pow_offset = value_offsets.at(idx);

    idx = value_idx.at("Beta2Pow");
    beta2_pow_offset = value_offsets.at(idx);

    // add attr later
    beta1 = 0.9;
    beta2 = 0.999;
    beta1_pow = 1.0;
    beta2_pow = 1.0;
    epsilon = 1.0e-8;
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    beta1_pow *= beta1;
    beta2_pow *= beta2;
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* values = block->Get(id);
      float lr_ = *(global_learning_rate_) * (values + lr_offset)[0];
      VLOG(4) << "SAdam LearningRate: " << lr_;
      float* param = values + param_offset;
      float* moment1 = values + m1_offset;
      float* moment2 = values + m2_offset;

      lr_ *= sqrt(1 - beta2_pow) / (1 - beta1_pow); 

      std::vector<float> grad, grad2, tmp;
      grad.resize(update_numel);
      grad2.resize(update_numel);
      tmp.resize(update_numel);

      blas.VCOPY(update_numel, update_values + x * update_numel, grad.data());
      blas.VCOPY(update_numel, update_values + x * update_numel, grad2.data());

      blas.SCAL(update_numel, 1 - beta1, grad.data());
      blas.VSQUARE(update_numel, grad2.data(), grad2.data());
      blas.SCAL(update_numel, 1 - beta2, grad2.data());

      blas.SCAL(update_numel, beta1, moment1);
      blas.VADD(update_numel, moment1, grad.data(), moment1);
      blas.SCAL(update_numel, beta2, moment2);
      blas.VADD(update_numel, moment2, grad2.data(), moment2);

      float* tmp_ = tmp.data();
      // float eps_ = epsilon * sqrt(1 - beta2_pow);
      float eps_ = epsilon;

      SQRT<float>(update_numel, moment2, tmp_);
      ADD<float>(update_numel, tmp_, eps_, tmp_);

      blas.VDIV(update_numel, moment1, tmp_, tmp_);
      blas.SCAL(update_numel, lr_, tmp_);
      blas.VSUB(update_numel, param, tmp_, param);
    }
  }

  int64_t save(const int mode,
            const int shard_idx,
            const std::string prefix,
            const std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    // save beta1_pow & beta2_pow;
    std::ofstream beta1_out(prefix + "." + "beta1_pow.block" + std::to_string(shard_idx), std::ios::binary);
    beta1_out.write(reinterpret_cast<const char *>(&beta1_pow), sizeof(float)); 
    beta1_out.close();
    
    std::ofstream beta2_out(prefix + "." + "beta2_pow.block" + std::to_string(shard_idx), std::ios::binary);
    beta2_out.write(reinterpret_cast<const char *>(&beta2_pow), sizeof(float));    
    beta2_out.close();

    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx), 
                                       prefix + "." + "moment1.block" + std::to_string(shard_idx), 
                                       prefix + "." + "moment2.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset, m1_offset, m2_offset};
    return SaveToBin(mode, var_names, offsets, shard_values);
  }

  void load(const int shard_idx,
            const std::string prefix,
            std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    // load beta1_pow && beta2_pow;
    std::ifstream beta1_fin(prefix + "." + "beta1_pow.block" + std::to_string(shard_idx), std::ios::binary);
    beta1_fin.read(reinterpret_cast<char *>(&beta1_pow), sizeof(float));

    std::ifstream beta2_fin(prefix + "." + "beta2_pow.block" + std::to_string(shard_idx), std::ios::binary);
    beta2_fin.read(reinterpret_cast<char *>(&beta2_pow), sizeof(float));

    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx),
                                       prefix + "." + "moment1.block" + std::to_string(shard_idx),
                                       prefix + "." + "moment2.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset, m1_offset, m2_offset};

    LoadFromBin(var_names, offsets, shard_values);
  }

  int lr_offset;
  int m1_offset;
  int m2_offset;
  int beta1_pow_offset;
  int beta2_pow_offset;

  float beta1;
  float beta2;
  float beta1_pow;
  float beta2_pow;
  float epsilon;
};

class SAdagrad : public SparseOptimizer {
 public:
  explicit SAdagrad(const std::vector<std::string>& value_names,
                 const std::vector<int>& value_dims,
                 const std::vector<int>& value_offsets,
                 const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    idx = value_idx.at("LearningRate");
    lr_offset = value_offsets.at(idx);

    idx = value_idx.at("Moment");
    m_offset = value_offsets.at(idx);

    // add attr later
    epsilon = 1.0e-6;
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* values = block->Get(id);
      float lr_ = (*global_learning_rate_) * (*(values + lr_offset));
      float* param = values + param_offset;
      float* moment = values + m_offset;

      std::vector<float> grad, grad2, tmp;
      grad.resize(update_numel);
      grad2.resize(update_numel);
      tmp.resize(update_numel);

      blas.VCOPY(update_numel, update_values + x * update_numel, grad.data());
      blas.VCOPY(update_numel, update_values + x * update_numel, grad2.data());
      blas.VSQUARE(update_numel, grad2.data(), grad2.data());

      blas.VADD(update_numel, moment, grad2.data(), moment);
 
      float* tmp_ = tmp.data();
      SQRT<float>(update_numel, moment, tmp_);
      ADD<float>(update_numel, tmp_, epsilon, tmp_);

      blas.VDIV(update_numel, grad.data(), tmp_, tmp_);
      blas.SCAL(update_numel, lr_, tmp_);
      blas.VSUB(update_numel, param, tmp_, param);
    }
  }

  int64_t save(const int mode,
               const int shard_idx,
               const std::string prefix,
               const std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx), 
                                       prefix + "." + "moment.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset, m_offset};
    return SaveToBin(mode, var_names, offsets, shard_values);
  }

  void load(const int shard_idx,
            const std::string prefix,
            std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx),
                                       prefix + "." + "moment.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset, m_offset};
    LoadFromBin(var_names, offsets, shard_values);
  }

  int lr_offset;
  int m_offset;
  float epsilon;
};

class SDecayedAdagrad : public SparseOptimizer {
 public:
  explicit SDecayedAdagrad(const std::vector<std::string>& value_names,
                           const std::vector<int>& value_dims,
                           const std::vector<int>& value_offsets,
                           const std::unordered_map<std::string, int>& value_idx)
      : SparseOptimizer(value_names, value_dims, value_offsets, value_idx) {
    auto idx = value_idx.at("Param");
    param_offset = value_offsets.at(idx);
    update_numel = value_dims.at(idx);

    idx = value_idx.at("LearningRate");
    lr_offset = value_offsets.at(idx);

    idx = value_idx.at("Moment");
    m_offset = value_offsets.at(idx);

    // add attr later
    epsilon = 1.0e-6;
    decay = 1.0;
  }

  void update(const uint64_t* keys, const float* update_values, size_t num,
              const std::vector<uint64_t>& offsets,
              ValueBlock* block) override {
    auto blas = GetBlas<float>();
    for (auto x : offsets) {
      auto id = keys[x];
      if (!block->GetEntry(id)) continue;
      auto* values = block->Get(id);
      float lr_ = (*global_learning_rate_) * (*(values + lr_offset));
      float* param = values + param_offset;
      float* moment = values + m_offset;

      std::vector<float> grad, grad2, tmp;
      grad.resize(update_numel);
      grad2.resize(update_numel);
      tmp.resize(update_numel);

      blas.VCOPY(update_numel, update_values + x * update_numel, grad.data());
      blas.VCOPY(update_numel, update_values + x * update_numel, grad2.data());
      blas.VSQUARE(update_numel, grad2.data(), grad2.data());
      blas.SCAL(update_numel, decay, moment);
 
      blas.VADD(update_numel, moment, grad2.data(), moment);
      
      float* tmp_ = tmp.data();
      SQRT<float>(update_numel, moment, tmp_);
      ADD<float>(update_numel, tmp_, epsilon, tmp_);

      blas.VDIV(update_numel, grad.data(), tmp_, tmp_);
      blas.SCAL(update_numel, lr_, tmp_);
      blas.VSUB(update_numel, param, tmp_, param);
    }
  }

  int64_t save(const int mode,
               const int shard_idx,
               const std::string prefix,
               const std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx), 
                                       prefix + "." + "moment.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset, m_offset};
    return SaveToBin(mode, var_names, offsets, shard_values);
  }

  void load(const int shard_idx,
            const std::string prefix,
            std::vector<std::shared_ptr<ValueBlock>>& shard_values) {
    std::vector<std::string> var_names{prefix + "." + "param.block" + std::to_string(shard_idx),
                                       prefix + "." + "moment.block" + std::to_string(shard_idx)};
    std::vector<int> offsets {param_offset, m_offset};
    LoadFromBin(var_names, offsets, shard_values);
  }

  int lr_offset;
  int m_offset;
  float epsilon;
  float decay;
};

}  // namespace distributed
}  // namespace paddle
