//   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
#include <memory>
#include <vector>
#include "paddle/fluid/framework/fleet/fleet_wrapper.h"
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/tensor.h"

namespace paddle {
namespace operators {

template <typename T>
void PullSparseFunctor(
    const framework::ExecutionContext& ctx,
    const std::vector<uint64_t>* fea_keys_const,
    const std::vector<T*>* pull_result_ptr_const,
    const std::vector<::std::future<int32_t>>* pull_sparse_status_const,
    uint32_t max_feasign_num,
    uint32_t sleep_seconds_before_fail_exit) {
  const auto& inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  const auto& outputs = ctx.MultiOutput<framework::LoDTensor>("Out");
  uint32_t fea_dim = static_cast<uint32_t>(ctx.Attr<int>("EmbeddingDim"));
#ifdef PADDLE_WITH_PSLIB
  std::vector<uint64_t>* fea_keys =
      const_cast<std::vector<uint64_t>*>(fea_keys_const);
  std::vector<T*>* pull_result_ptr =
      const_cast<std::vector<T*>*>(pull_result_ptr_const);
  std::vector<::std::future<int32_t>>* pull_sparse_status =
      const_cast<std::vector<::std::future<int32_t>>*>(
          pull_sparse_status_const);
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  fea_keys->clear();
  fea_keys->reserve(max_feasign_num);
  
  uint64_t padding_id =static_cast<uint64_t>(ctx.Attr<int>("PaddingId"));
  std::vector<T> init_value(fea_dim, 0);
  pull_result_ptr->clear();
  framework::LoDTensor* output = nullptr;
  T* output_data = nullptr;
  size_t output_index = -1;
  size_t output_len = 0;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const framework::LoDTensor* tensor = inputs[index];
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (size_t i = 0; i < len; ++i, output_len+=fea_dim) {
      if (!output || output_len == output->numel()) {
        ++output_index;
        PADDLE_ENFORCE_LT(output_index, outputs.size(),
                          "output_index should < outputs size");
        output = outputs[output_index];
        output_data = output->mutable_data<T>(ctx.GetPlace());
        output_len = 0;
        PADDLE_ENFORCE_EQ(output->numel() % fea_dim, 0,
                          "output->numel \% fea_dim should be 0");
        PADDLE_ENFORCE_NE(output_data, nullptr,
                          "output_data should not be null");
      }
      if (ids[i] == padding_id) {
        memcpy(output_data + output_len, init_value.data(),
               sizeof(T) * fea_dim);
        continue;
      }
      fea_keys->push_back(static_cast<uint64_t>(ids[i]));
      pull_result_ptr->push_back(output_data + output_len);
    }
  }
  PADDLE_ENFORCE_NE(framework::FleetWrapper::pslib_ptr_, nullptr,
                    "pslib_ptr_ should not be null");
  auto status = framework::FleetWrapper::pslib_ptr_->_worker_ptr->pull_sparse(
      pull_result_ptr->data(), table_id, fea_keys->data(), fea_keys->size());
  pull_sparse_status->clear();
  pull_sparse_status->push_back(std::move(status));
  for (auto& t : *pull_sparse_status) {
    t.wait();
    auto status = t.get();
    if (status != 0) {
      LOG(ERROR) << "fleet pull sparse failed, status[" << status << "]";
      sleep(sleep_seconds_before_fail_exit);
    }
  }

#else
  for (size_t index = 0; index < inputs.size(); ++index) {
    const auto* tensor = inputs[index];
    size_t len = tensor->numel();
    std::vector<T> init_data(fea_dim, 0);
    for (size_t i = 0; i < len; ++i) {
      memcpy(outputs[index]->mutable_data<T>(ctx.GetPlace()),
             init_data.data(), fea_dim);
    }
  }
#endif
}

size_t get_absolute_sum(size_t start, size_t end, size_t level,
                        const framework::LoD& lod) {
  if (level >= lod.size() - 1) {
    return end - start;
  }
  size_t ret = 0;
  for (size_t i = start; i < end - 1; ++i) {
    size_t pos1 = lod[level][i];
    size_t pos2 = lod[level][i + 1];
    ret += get_absolute_sum(pos1, pos2, level + 1, lod);
  }
  return ret;
}

template <typename T>
void PushSparseFunctor(
    const framework::ExecutionContext& ctx,
    const std::vector<uint64_t>* push_keys_const,
    const std::vector<std::vector<T>>* push_values_const,
    const std::vector<T>* fea_labels_const,
    uint32_t max_feasign_num) {
#ifdef PADDLE_WITH_PSLIB
  auto inputs = ctx.MultiInput<framework::LoDTensor>("Ids");
  auto outputs =
      ctx.MultiInput<framework::LoDTensor>(framework::GradVarName("Out"));
  uint32_t fea_dim = static_cast<uint32_t>(ctx.Attr<int>("EmbeddingDim"));
  std::string accesor = ctx.Attr<std::string>("AccessorClass");
  int show_index = 0;
  int click_index = 1;
  // these default values can not be used, it must be set.
  bool dump_slot = false;
  int slot_offset = 0;
  int grad_dim = 0;
  // don't worry, user do not have to care about all these flags
  if (accesor == "DownpourCtrAccessor") {
    dump_slot = true;
    slot_offset = 1;
    grad_dim = fea_dim - 2;
    show_index = 1;
    click_index = 2;
  } else if (accesor == "DownpourFeatureValueAccessor") {
    dump_slot = false;
    slot_offset = 0;
    grad_dim = fea_dim - 2;
  } else if (accesor == "DownpourSparseValueAccessor") {
    dump_slot = false;
    slot_offset = 0;
    grad_dim = fea_dim;
  }

  PADDLE_ENFORCE_GE(grad_dim, 0, "grad_dim should >= 0");

  int batch_size = -1;
  for (auto* input : inputs) {    
    int cur_batch_size =
        input->lod().size() ? input->lod()[0].size() - 1 : input->dims()[0];
    if (batch_size == -1) {
      batch_size = cur_batch_size;
    } else {
      PADDLE_ENFORCE_EQ(batch_size, cur_batch_size,
                        "inputs batch_size must be the same");
    }
  }
  PADDLE_ENFORCE_GT(batch_size, 0, "batch_size should > 0");

  bool scale_sparse = ctx.Attr<bool>("ScaleSparseGrad");
  if (scale_sparse && grad_dim > 0) {
    size_t dim = static_cast<size_t>(grad_dim);
    for (const framework::LoDTensor* g_tensor_const : outputs) {
      framework::LoDTensor* g_tensor =
          const_cast<framework::LoDTensor*>(g_tensor_const);
      T* g = g_tensor->mutable_data<T>(ctx.GetPlace());
      Eigen::Map<
          Eigen::Matrix<T, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>
          g_mat(g, g_tensor->numel() / dim, dim);
      g_mat.rightCols(grad_dim) *= batch_size;
    }
  }

  uint64_t padding_id =static_cast<uint64_t>(ctx.Attr<int>("PaddingId"));
  std::vector<T>* fea_labels = const_cast<std::vector<T>*>(fea_labels_const);
  fea_labels->clear();
  const std::string& label_name = ctx.Attr<std::string>("CtrLabelName");
  const framework::Scope& scope = ctx.scope();
  framework::Variable* var = scope.FindVar(label_name);
  size_t global_idx = 0;
  if (label_name != "") {
    PADDLE_ENFORCE_NE(var, nullptr, "label var should not be null");
    framework::LoDTensor* label_tensor = var->GetMutable<framework::LoDTensor>();
    PADDLE_ENFORCE_NE(label_tensor, nullptr, "label tensor should not be null");
    int64_t* label_ptr = label_tensor->data<int64_t>();

    for (auto* tensor : inputs) {
      VLOG(0) << "tensor";
      const int64_t* ids = tensor->data<int64_t>();
      size_t fea_idx = 0;
      for (size_t lod_idx = 1; lod_idx < tensor->lod()[0].size(); ++lod_idx) {
        size_t cur = get_absolute_sum(tensor->lod()[0][lod_idx - 1],
            tensor->lod()[0][lod_idx], 0, tensor->lod());
        for (size_t i = 0; i < cur; ++i, ++fea_idx) {
          if (ids[fea_idx] == padding_id) {
            continue;
          }
          fea_labels->push_back(static_cast<T>(label_ptr[lod_idx - 1]));
          ++global_idx;
        }
      }
    }
  }

  std::vector<uint64_t>* push_keys =
      const_cast<std::vector<uint64_t>*>(push_keys_const);
  push_keys->clear();
  push_keys->reserve(max_feasign_num);
  std::vector<std::vector<T>>* push_values =
      const_cast<std::vector<std::vector<T>>*>(push_values_const);
  push_values->clear();
  push_values->reserve(max_feasign_num);
  push_values->clear();

  auto input_names = ctx.Attr<std::vector<std::string>>("InputNames");
  framework::LoDTensor* output = nullptr;
  T* output_data = nullptr;
  size_t output_index = -1;
  size_t output_len = 0;
  size_t input_idx = 0;
  for (size_t index = 0; index < inputs.size(); ++index) {
    const framework::LoDTensor* tensor = inputs[index];
    const int64_t* ids = tensor->data<int64_t>();
    size_t len = tensor->numel();
    for (size_t i = 0; i < len; ++i, output_len+=fea_dim) {
      if (!output || output_len == output->numel()) {
        ++output_index;
        PADDLE_ENFORCE_LT(output_index, outputs.size(),
                          "output_index should < outputs size");
        output = const_cast<framework::LoDTensor*>(outputs[output_index]);
        output_data = output->mutable_data<T>(ctx.GetPlace());
        output_len = 0;
        PADDLE_ENFORCE_EQ(output->numel() % fea_dim, 0,
                          "output->numel \% fea_dim should == 0");
        PADDLE_ENFORCE_NE(output_data, nullptr,
                          "output_data should not be null");
      }
      if (ids[i] == padding_id) {
        continue;
      }
      push_keys->emplace_back(ids[i]);
      push_values->emplace_back(fea_dim + slot_offset);
      T* data = push_values->back().data();
      if (!var) {
        memcpy(data + slot_offset, output_data + output_len, sizeof(T) * fea_dim);
      } else {
        memcpy(data + slot_offset, output_data + output_len, sizeof(T) * grad_dim);
        data[show_index] = 1.0f;
        data[click_index] = static_cast<T>(fea_labels->at(input_idx));
      }
      if (dump_slot) {
        int slot = boost::lexical_cast<int>(input_names[index]);
        data[0] = static_cast<T>(slot);
      }
      ++input_idx;
    }
  }

  if (label_name != "") {
    PADDLE_ENFORCE_EQ(input_idx, global_idx,
                      "input_idx should == global_idx");
  }

  std::vector<T*> push_g_vec(input_idx, nullptr);
  for (auto i = 0u; i < push_keys->size(); ++i) {
    push_g_vec[i] = push_values->at(i).data();
  }
  PADDLE_ENFORCE_NE(framework::FleetWrapper::pslib_ptr_, nullptr,
                    "pslib_ptr_ should not be null");
  auto table_id = static_cast<uint32_t>(ctx.Attr<int>("TableId"));
  auto status = framework::FleetWrapper::pslib_ptr_->_worker_ptr->push_sparse(
      table_id, push_keys->data(), (const T**)push_g_vec.data(),
      push_keys->size());
#endif
}

template <typename T>
class PullSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    PullSparseFunctor<T>(ctx, &fea_keys_, &pull_result_ptr_,
                         &pull_sparse_status_, max_feasign_num_,
                         sleep_seconds_before_fail_exit_);
  }
 protected:
   std::vector<uint64_t> fea_keys_;
   std::vector<T*> pull_result_ptr_;
   std::vector<::std::future<int32_t>> pull_sparse_status_;
   uint32_t max_feasign_num_ = 10240000;
   uint32_t sleep_seconds_before_fail_exit_ = 300;
};

template <typename T>
class PushSparseCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext &ctx) const override {
    PushSparseFunctor<T>(ctx, &push_keys_, &push_values_,
                        &fea_labels_, max_feasign_num_);
  }

 protected:
  std::vector<uint64_t> push_keys_;
  std::vector<std::vector<T>> push_values_;
  std::vector<T> fea_labels_;
  uint32_t max_feasign_num_ = 10240000;
};
}  // namespace operators
}  // namespace paddle
