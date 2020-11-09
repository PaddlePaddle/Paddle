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
#include <algorithm>
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dropout_op.h"
#include "paddle/fluid/operators/math/blas.h"
#include "paddle/fluid/operators/math/detail/activation_functions.h"
#include "paddle/fluid/operators/math/fc.h"
#include "paddle/fluid/operators/math/lstm_compute.h"
#include "paddle/fluid/operators/math/math_function.h"
#include "paddle/fluid/operators/unique_op.h"
#include "paddle/fluid/operators/utils.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;
using TensorList = std::vector<framework::Tensor>;

void SwapPoniter(Tensor** a, Tensor** b) {
  Tensor* c = *a;
  *a = *b;
  *b = c;
}

template <typename T>
void Print3DTensor(const Tensor* a, std::string name) {
  const int& row = a->dims()[0];
  const int& heigth = a->dims()[1];
  const int& width = a->dims()[2];
  std::string message = "print value, name is " + name + "\n";
  for (int r = 0; r < row; r++) {
    for (int i = 0; i < heigth; i++) {
      for (int j = 0; j < width; j++) {
        message +=
            std::to_string(a->data<T>()[r * heigth * width + i * width + j]) +
            " ";
      }
      message += "\n";
    }
    message += "*******\n";
  }

  VLOG(0) << message;
  VLOG(0) << "----------------------------";
}

template <typename T>
void Print2DTensor(const Tensor* a, std::string name) {
  const int& heigth = a->dims()[0];
  const int& width = a->dims()[1];
  std::string message = "print value, name is " + name + "\n";
  for (int i = 0; i < heigth; i++) {
    for (int j = 0; j < width; j++) {
      message += std::to_string(a->data<T>()[i * width + j]) + " ";
    }
    message += "\n";
  }
  VLOG(0) << message;
  VLOG(0) << "----------------------------";
}
template <typename T>
void create_mask_matrix(const framework::ExecutionContext& context,
                        const Tensor* sequence_length, Tensor* mask_matrix,
                        const bool& is_reverse, int* min_seq_len) {
  const auto& seq_len_vec = GetDataFromTensor<int>(sequence_length);
  const int& table_width = mask_matrix->dims()[0];
  Tensor temp;
  temp.Resize(
      framework::make_ddim({mask_matrix->dims()[1], mask_matrix->dims()[0]}));
  T* data_temp = temp.mutable_data<T>(context.GetPlace());
  std::fill(data_temp, data_temp + mask_matrix->numel(), static_cast<T>(1.0));
  *min_seq_len = table_width;
  for (unsigned int i = 0; i < seq_len_vec.size(); i++) {
    // reset the mask matrix
    *min_seq_len = std::min(seq_len_vec[i], *min_seq_len);
    if (seq_len_vec[i] == table_width) {
      continue;
    }
    if (is_reverse) {
      std::fill(data_temp + i * table_width,
                data_temp + (i + 1) * table_width - seq_len_vec[i],
                static_cast<T>(0));
    } else {
      std::fill(data_temp + i * table_width + seq_len_vec[i],
                data_temp + (i + 1) * table_width, static_cast<T>(0));
    }
  }
  mask_matrix->mutable_data<T>(context.GetPlace());
  std::vector<int> trans_vec;
  trans_vec.emplace_back(1);
  trans_vec.emplace_back(0);
  auto& dev_ctx = context.template device_context<platform::CPUDeviceContext>();
  TransCompute<platform::CPUDeviceContext, T>(2, dev_ctx, temp, mask_matrix,
                                              trans_vec);
}

template <typename T>
struct Cell {
  virtual ~Cell() {}
  virtual void operator()(const platform::CPUDeviceContext* device_ctx,
                          Tensor* input, const Tensor* weight_hh,
                          const Tensor* init_h, const Tensor* init_c,
                          Tensor* last_h, Tensor* last_c, Tensor* last_c_act,
                          Tensor* output) {}
};

template <typename T>
struct LSTMCell : Cell<T> {
  void operator()(const platform::CPUDeviceContext* device_ctx, Tensor* input,
                  const Tensor* weight_hh, const Tensor* init_h,
                  const Tensor* init_c, Tensor* last_h, Tensor* last_c,
                  Tensor* last_c_act, Tensor* output) override {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(*device_ctx);
    auto mat_dim_a = math::CreateMatrixDescriptor(init_h->dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(weight_hh->dims(), 0, true);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    // convert the batch matmul to matmul, this operator could be speed faster
    blas.MatMul(*init_h, mat_dim_a, *weight_hh, mat_dim_b, static_cast<T>(1.0),
                input, static_cast<T>(1.0));

    math::LstmMetaValue<T> lstm_value;
    lstm_value.check_ig = nullptr;
    lstm_value.check_fg = nullptr;
    lstm_value.check_og = nullptr;

    auto gate_act = math::detail::GetActivationType("sigmoid_v2");
    auto cell_act = math::detail::GetActivationType("tanh_v2");
    auto cand_act = math::detail::GetActivationType("tanh_v2");

    size_t frame_size = init_h->dims()[2];
    size_t batch_size = init_h->dims()[1];

    Tensor cell_pre_act;
    if (last_c_act == nullptr) { /* is test */
      cell_pre_act.mutable_data<T>(init_h->dims(), device_ctx->GetPlace());
      last_c_act = &cell_pre_act;
    }

    lstm_value.prev_state_value = init_c->data<T>();
    lstm_value.gate_value = input->data<T>();
    lstm_value.output_value = output->data<T>();
    lstm_value.state_value = last_c->data<T>();
    lstm_value.state_active_value = last_c_act->data<T>();
    T cell_clip = 0.0;
    math::LstmUnitFunctor<platform::CPUDeviceContext, T>::compute(
        *device_ctx, lstm_value, frame_size, batch_size, cell_clip, gate_act,
        cell_act, cand_act, false);
  }
};
template <typename T>
void dropout_cpu_function_inplace(const framework::ExecutionContext& context,
                                  Tensor* x, Tensor* mask,
                                  const float& dropout_prob,
                                  const int& seed_number, const bool& is_test) {
  auto* x_data = x->data<T>();
  if (!is_test) {
    size_t size = framework::product(x->dims());

    if (!mask->IsInitialized()) {
      auto mask_data =
          mask->mutable_data<uint8_t>(x->dims(), context.GetPlace());
      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        std::fill(x_data, x_data + size, static_cast<T>(0));
        std::fill(mask_data, mask_data + size, static_cast<T>(0));
        return;
      }
      auto engine = framework::GetCPURandomEngine(seed_number);
      std::uniform_real_distribution<float> dist(0, 1);
      for (size_t i = 0; i < size; ++i) {
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
          x_data[i] = static_cast<T>(0);
        } else {
          mask_data[i] = 1;
          x_data[i] /= static_cast<T>(1.0f - dropout_prob);
        }
      }
      return;
    }
    auto mask_data = mask->data<uint8_t>();
    if (dropout_prob == 1.0f) {
      std::fill(x_data, x_data + size, static_cast<T>(0));
      return;
    }
    for (size_t i = 0; i < size; ++i) {
      if (mask_data[i] == 1) {
        x_data[i] /= static_cast<T>(1.0f - dropout_prob);
      } else {
        x_data[i] = static_cast<T>(0);
      }
    }
  }
}

template <typename T, typename CellType>
struct Layer {
  explicit Layer(const CellType& cell) : cell_(cell) {}
  virtual ~Layer() {}
  void preprocess(const framework::ExecutionContext& context,
                  const Tensor* input, const Tensor& weight,
                  const Tensor& bias_ih, const Tensor& bias_hh,
                  Tensor* cache_input, bool is_test) {
    // crate the temp input for the X * W_ih^T + Bias_ih
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    const int& hidden_size = weight.dims()[0];
    cache_input->Resize(framework::make_ddim(
        {input->dims()[0], input->dims()[1], hidden_size}));
    if (is_test) {
      cache_input->mutable_data<T>(context.GetPlace());
    }
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(dev_ctx);
    auto mat_dim_a = math::CreateMatrixDescriptor(input->dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(weight.dims(), 0, true);
    // convert the batch matmul to matmul, this operator could be speed faster
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    blas.MatMul(*input, mat_dim_a, weight, mat_dim_b, static_cast<T>(1.0),
                cache_input, static_cast<T>(0));

    auto eigen_in = framework::EigenMatrix<T>::Reshape(
        *cache_input, cache_input->dims().size() - 1);
    auto eigen_bias_ih = framework::EigenMatrix<T>::From(
        bias_ih, framework::make_ddim({1, bias_ih.dims()[0]}));
    auto eigen_bias_hh = framework::EigenMatrix<T>::From(
        bias_hh, framework::make_ddim({1, bias_hh.dims()[0]}));
    const int& row_num =
        framework::product(cache_input->dims()) / cache_input->dims()[2];
    eigen_in = eigen_in +
               eigen_bias_ih.broadcast(Eigen::DSizes<int, 2>(row_num, 1)) +
               eigen_bias_hh.broadcast(Eigen::DSizes<int, 2>(row_num, 1));
  }

  void postprocess(const framework::ExecutionContext& context, Tensor* output,
                   const Tensor* init_h, const Tensor* init_c, Tensor* last_h,
                   Tensor* last_c, const Tensor& mask_tensor, bool is_lstm) {
    // in the output, if mask flag is 0, we will retun the zero data
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto eigen_output =
        framework::EigenMatrix<T>::Reshape(*output, output->dims().size() - 1);
    auto eigen_mask = framework::EigenMatrix<T>::From(
        mask_tensor, framework::make_ddim({mask_tensor.dims()[1], 1}));
    auto eigen_init_h =
        framework::EigenMatrix<T>::Reshape(*init_h, init_h->dims().size() - 1);
    auto eigen_last_h =
        framework::EigenMatrix<T>::Reshape(*last_h, last_h->dims().size() - 1);
    auto eigen_mask_broadcast =
        eigen_mask.broadcast(Eigen::DSizes<int, 2>(1, output->dims()[1]));
    eigen_last_h.device(place) = eigen_output * eigen_mask_broadcast +
                                 eigen_init_h * (1 - eigen_mask_broadcast);
    eigen_output.device(place) = eigen_output * eigen_mask_broadcast;

    if (is_lstm) {
      auto eigen_init_c = framework::EigenMatrix<T>::Reshape(
          *init_c, init_c->dims().size() - 1);
      auto eigen_last_c = framework::EigenMatrix<T>::Reshape(
          *last_c, last_c->dims().size() - 1);
      eigen_last_c.device(place) = eigen_last_c * eigen_mask_broadcast +
                                   eigen_init_c * (1 - eigen_mask_broadcast);
    }
  }

  virtual void operator()(const framework::ExecutionContext& context,
                          const Tensor* input, const TensorList& vec,
                          const TensorList& init_h, const TensorList& init_c,
                          const Tensor* sequence_length, TensorList last_h,
                          TensorList last_c, Tensor* output,
                          const int& layer_idx, const int& gate_num,
                          Tensor* gate_value, Tensor* cell_value,
                          Tensor* cell_act_value, bool is_test) {}

  void RunTestIter(const framework::ExecutionContext& context,
                   const Tensor* input, const TensorList& vec,
                   const TensorList& init_h, const TensorList& init_c,
                   const Tensor* sequence_length, TensorList* last_h_ptr,
                   TensorList* last_c_ptr, Tensor* output, int layer_idx,
                   Tensor* gate_value, Tensor* cell_value,
                   Tensor* cell_act_value, bool is_bidirect, int offset) {
    bool is_reverse = false;
    if (is_bidirect) {
      layer_idx = 2 * layer_idx + offset;
      if (offset > 0) {
        is_reverse = true;
      }
    }
    bool is_lstm = false;
    if (init_c.size() > 0 && last_c_ptr->size() > 0) {
      is_lstm = true;
    }
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    const int& time_step = input->dims()[0];
    this->preprocess(context, input, vec[0 + offset * 4], vec[2 + offset * 4],
                     vec[3 + offset * 4], gate_value, true);
    auto input_tensors = Unbind(*gate_value);
    auto output_tensors = Unbind(*output);
    if (is_reverse) {
      std::reverse(input_tensors.begin(), input_tensors.end());
      std::reverse(output_tensors.begin(), output_tensors.end());
    }
    TensorList mask_tensor_list;
    // construct the mask matrix for the mask
    bool has_sequence_length = false;
    if (sequence_length != nullptr) {
      has_sequence_length = true;
    }
    Tensor mask_matrix;
    int mask_min_length = time_step;
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));

      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse,
                            &mask_min_length);
      mask_tensor_list = Unbind(mask_matrix);
    }

    if (is_reverse) {
      mask_min_length = mask_min_length - time_step + 1;
    }
    bool has_allocate_mem_c = false;
    bool has_use_last_h_holder = false;
    const int& reverse_flag = is_reverse ? -1 : 1;

    // define the init_h holder for the swap
    Tensor init_h_temp;
    framework::TensorCopy(*&init_h[layer_idx], context.GetPlace(), dev_ctx,
                          &init_h_temp);
    Tensor* init_h_holder = &init_h_temp;
    Tensor* last_h_holder = nullptr;
    if (0 < mask_min_length) {
      last_h_holder = &(output_tensors[0]);
    } else {
      last_h_holder = &(*last_h_ptr)[layer_idx];
      has_use_last_h_holder = true;
    }

    Tensor* init_c_holder = nullptr;
    const Tensor* init_c_temp_holder = nullptr;
    Tensor init_c_temp;
    Tensor* last_c_holder = nullptr;

    if (is_lstm) {
      last_c_holder = &(*last_c_ptr)[layer_idx];
      init_c_temp_holder = &init_c[layer_idx];
    }
    for (int i = 0; i < time_step; i++) {
      bool in_mask = (reverse_flag * i) >= mask_min_length;
      if (i > 0) {
        if (!has_allocate_mem_c) {
          if (is_lstm) {
            init_c_temp.Resize(init_c[layer_idx].dims());
            init_c_temp.mutable_data<T>(context.GetPlace());
            init_c_holder = &init_c_temp;
          }
          has_allocate_mem_c = true;
        }
        SwapPoniter(&init_c_holder, &last_c_holder);
        init_c_temp_holder = init_c_holder;
      }
      cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4], init_h_holder,
            init_c_temp_holder, last_h_holder, last_c_holder, nullptr,
            &output_tensors[i]);
      if (in_mask) {
        this->postprocess(context, &output_tensors[i], init_h_holder,
                          init_c_temp_holder, last_h_holder, last_c_holder,
                          mask_tensor_list[i], is_lstm);
      }
      // prepare next step
      if (i + 1 < time_step) {
        bool next_step_mask = (reverse_flag * (i + 1)) >= mask_min_length;
        if (next_step_mask) {
          if (!has_use_last_h_holder) {
            init_h_holder = &(*last_h_ptr)[layer_idx];
          }
        } else {
          init_h_holder = &(output_tensors[i + 1]);
        }
        SwapPoniter(&init_h_holder, &last_h_holder);
      }
    }
    if (has_sequence_length) {
      if (last_h_holder != &(*last_h_ptr)[layer_idx]) {
        framework::TensorCopy(*last_h_holder, context.GetPlace(), dev_ctx,
                              &(*last_h_ptr)[layer_idx]);
      }
    } else {
      framework::TensorCopy(output_tensors[time_step - 1], context.GetPlace(),
                            dev_ctx, &(*last_h_ptr)[layer_idx]);
    }

    if (time_step % 2 == 0) {
      if (is_lstm) {
        framework::TensorCopy(*last_c_holder, context.GetPlace(), dev_ctx,
                              &(*last_c_ptr)[layer_idx]);
      }
    }
  }

  void RunIter(const framework::ExecutionContext& context, const Tensor* input,
               const TensorList& vec, const TensorList& init_h,
               const TensorList& init_c, const Tensor* sequence_length,
               TensorList* last_h_ptr, TensorList* last_c_ptr, Tensor* output,
               int layer_idx, Tensor* gate_value, Tensor* cell_value,
               Tensor* cell_act_value, bool is_bidirect, int offset,
               bool is_test) {
    if (is_test) {
      RunTestIter(context, input, vec, init_h, init_c, sequence_length,
                  last_h_ptr, last_c_ptr, output, layer_idx, gate_value,
                  cell_value, cell_act_value, is_bidirect, offset);
      return;
    }
    bool is_reverse = false;
    if (is_bidirect) {
      layer_idx = 2 * layer_idx + offset;
      if (offset > 0) {
        is_reverse = true;
      }
    }
    bool is_lstm = false;
    if (init_c.size() > 0 && last_c_ptr->size() > 0) {
      is_lstm = true;
    }
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    const int& time_step = input->dims()[0];
    this->preprocess(context, input, vec[0 + offset * 4], vec[2 + offset * 4],
                     vec[3 + offset * 4], gate_value, is_test);
    auto input_tensors = Unbind(*gate_value);
    auto output_tensors = Unbind(*output);
    if (is_reverse) {
      std::reverse(input_tensors.begin(), input_tensors.end());
      std::reverse(output_tensors.begin(), output_tensors.end());
    }
    TensorList mask_tensor_list;
    // construct the mask matrix for the mask
    bool has_sequence_length = false;
    if (sequence_length != nullptr) {
      has_sequence_length = true;
    }
    Tensor mask_matrix;
    int mask_min_length = time_step;
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));
      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse,
                            &mask_min_length);
      mask_tensor_list = Unbind(mask_matrix);
    }
    if (is_reverse) {
      mask_min_length = mask_min_length - time_step + 1;
    }

    // define the init_h holder for the swap
    bool has_use_last_h_holder = false;
    const int& reverse_flag = is_reverse ? -1 : 1;

    TensorList cell_value_tensors;
    TensorList cell_act_value_tensors;

    Tensor init_h_temp;
    framework::TensorCopy(*&init_h[layer_idx], context.GetPlace(), dev_ctx,
                          &init_h_temp);
    Tensor* init_h_holder = &init_h_temp;
    Tensor* last_h_holder = nullptr;
    if (0 < mask_min_length) {
      last_h_holder = &(output_tensors[0]);
    } else {
      last_h_holder = &(*last_h_ptr)[layer_idx];
      has_use_last_h_holder = true;
    }

    const Tensor* init_c_holder = nullptr;
    Tensor* last_c_holder = nullptr;
    Tensor* last_c_act_holder = nullptr;
    if (is_lstm) {
      cell_value->Resize({time_step, cell_value->numel() / time_step});
      cell_value_tensors = Unbind(*cell_value);
      cell_act_value->Resize({time_step, cell_value->numel() / time_step});
      cell_act_value_tensors = Unbind(*cell_act_value);
    }
    for (int i = 0; i < time_step; i++) {
      bool in_mask = (reverse_flag * i) >= mask_min_length;
      if (is_lstm) {
        if (i == 0) {
          init_c_holder = &init_c[layer_idx];
        } else {
          init_c_holder = &cell_value_tensors[i - 1];
        }
        cell_value_tensors[i].Resize(init_c[layer_idx].dims());
        cell_act_value_tensors[i].Resize(init_c[layer_idx].dims());
        last_c_holder = &cell_value_tensors[i];
        last_c_act_holder = &cell_act_value_tensors[i];
      }

      cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4], init_h_holder,
            init_c_holder, last_h_holder, last_c_holder, last_c_act_holder,
            &output_tensors[i]);
      if (in_mask) {
        this->postprocess(context, &output_tensors[i], init_h_holder,
                          init_c_holder, last_h_holder, last_c_holder,
                          mask_tensor_list[i], is_lstm);
      }
      // prepare next step
      if (i + 1 < time_step) {
        bool next_step_mask = (reverse_flag * (i + 1)) >= mask_min_length;
        if (next_step_mask) {
          if (!has_use_last_h_holder) {
            init_h_holder = &(*last_h_ptr)[layer_idx];
          }
        } else {
          init_h_holder = &(output_tensors[i + 1]);
        }
        SwapPoniter(&init_h_holder, &last_h_holder);
      }
    }
    if (has_sequence_length) {
      if (last_h_holder != &(*last_h_ptr)[layer_idx]) {
        framework::TensorCopy(*last_h_holder, context.GetPlace(), dev_ctx,
                              &(*last_h_ptr)[layer_idx]);
      }
    } else {
      framework::TensorCopy(output_tensors[time_step - 1], context.GetPlace(),
                            dev_ctx, &(*last_h_ptr)[layer_idx]);
    }
    if (is_lstm) {
      framework::TensorCopy(cell_value_tensors[time_step - 1],
                            context.GetPlace(), dev_ctx,
                            &(*last_c_ptr)[layer_idx]);
    }
  }
  // Cell for the rnn module
  CellType cell_;
};

template <typename T, typename CellType>
struct SingleLayer : public Layer<T, CellType> {
  explicit SingleLayer(const CellType& cell) : Layer<T, CellType>(cell) {}
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const TensorList& init_h, const TensorList& init_c,
                  const Tensor* sequence_length, TensorList last_h,
                  TensorList last_c, Tensor* output, const int& layer_idx,
                  const int& gate_num, Tensor* gate_value, Tensor* cell_value,
                  Tensor* cell_act_value, bool is_test) {
    this->RunIter(context, input, vec, init_h, init_c, sequence_length, &last_h,
                  &last_c, output, layer_idx, gate_value, cell_value,
                  cell_act_value, false, 0, is_test);
  }
};

template <typename T, typename CellType>
struct BidirLayer : public Layer<T, CellType> {
  explicit BidirLayer(const CellType& cell) : Layer<T, CellType>(cell) {}
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const TensorList& init_h, const TensorList& init_c,
                  const Tensor* sequence_length, TensorList last_h,
                  TensorList last_c, Tensor* output, const int& layer_idx,
                  const int& gate_num, Tensor* gate_value, Tensor* cell_value,
                  Tensor* cell_act_value, bool is_test) {
    TensorList output_vec(2);
    Tensor forward_input_w, forward_cell_value, forward_cell_act_value;
    Tensor backward_input_w, backward_cell_value, backward_cell_act_value;
    int time_step = input->dims()[0];
    int batch_size = input->dims()[1];
    int hidden_size = output->dims()[2];
    for (int i = 0; i < 2; ++i) {
      output_vec[i].Resize({time_step, batch_size, hidden_size / 2});
      output_vec[i].mutable_data<T>(context.GetPlace());
    }
    if (!is_test) {
      gate_value->Resize({2, gate_value->numel() / 2});
      cell_value->Resize({2, cell_value->numel() / 2});
      cell_act_value->Resize({2, cell_act_value->numel() / 2});

      forward_input_w = gate_value->Slice(0, 1);
      backward_input_w = gate_value->Slice(1, 2);

      if (cell_value->numel() > 0) /* for lstm and gru */ {
        forward_cell_value = cell_value->Slice(0, 1);
        backward_cell_value = cell_value->Slice(1, 2);
        forward_cell_act_value = cell_act_value->Slice(0, 1);
        backward_cell_act_value = cell_act_value->Slice(1, 2);
      }
    }

    this->RunIter(context, input, vec, init_h, init_c, sequence_length, &last_h,
                  &last_c, &output_vec[0], layer_idx, &forward_input_w,
                  &forward_cell_value, &forward_cell_act_value, true, 0,
                  is_test);

    this->RunIter(context, input, vec, init_h, init_c, sequence_length, &last_h,
                  &last_c, &output_vec[1], layer_idx, &backward_input_w,
                  &backward_cell_value, &backward_cell_act_value, true, 1,
                  is_test);

    // concat the the output result
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    paddle::operators::math::ConcatFunctor<platform::CPUDeviceContext, T>
        concat_functor;
    concat_functor(dev_ctx, output_vec, static_cast<int>(2), output);
  }
};
template <typename TensorType>
void SplitReserveData(TensorType* reserve_data, Tensor* gate_data,
                      Tensor* cell_data, Tensor* cell_act_data,
                      Tensor* hidden_data, int direction_num,
                      const int& time_step, const int& batch_size,
                      const int& hidden_size, const int& gate_num,
                      const int& num_layers) {
  int gate_num_tmp = gate_num;
  if (gate_num_tmp > 0) {
    gate_num_tmp += 1;
  }
  const int& gate_data_idx = (gate_num_tmp - 1) * num_layers;
  const int& cell_data_idx = gate_num_tmp * num_layers;
  const int& cell_act_data_idx = (gate_num_tmp + 1) * num_layers;
  const int& hidden_data_idx =
      (gate_num_tmp + 1) * num_layers + (num_layers - 1);
  if (gate_data_idx > 0) /** for lstm, gru **/ {
    *gate_data = reserve_data->Slice(0, gate_data_idx);
    *cell_data = reserve_data->Slice(gate_data_idx, cell_data_idx);
    *cell_act_data = reserve_data->Slice(cell_data_idx, cell_act_data_idx);
  } else /** for simple rnn **/ {
    *gate_data = reserve_data->Slice(0, cell_act_data_idx);
  }

  if (num_layers > 1) {
    *hidden_data = reserve_data->Slice(cell_act_data_idx, hidden_data_idx);
  }
}

template <typename TensorType>
void reset_parameter_vector(const std::vector<TensorType>& raw_params_vec,
                            const int& num_layers, const int& gate_num,
                            const bool& is_bidirec,
                            std::vector<TensorList>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    TensorList tensor_list;
    tensor_list.reserve(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      Tensor tensor_holder;
      tensor_list.emplace_back(tensor_holder);
    }
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      tensor_list[j].ShareDataWith(*raw_params_vec[tensor_idx]);
    }
    params_vec->emplace_back(tensor_list);
  }
}

template <typename CellType, typename T>
void AllocateReserveData(const framework::ExecutionContext& ctx,
                         Tensor* reserve_data, Tensor* gate_data,
                         Tensor* cell_data, Tensor* cell_act_data,
                         Tensor* hidden_data, const Tensor* input,
                         bool is_bidirec, int num_layers, int gate_num,
                         int hidden_size) {
  int gate_num_tmp = gate_num;
  if (gate_num > 0) {
    gate_num_tmp += 1;
  }
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& time_step = input->dims()[0];
  const int& batch_size = input->dims()[1];
  const int& block_size = direction_num * time_step * batch_size * hidden_size;
  const int& hidden_data_idx =
      (gate_num_tmp + 1) * num_layers + (num_layers - 1);
  reserve_data->Resize({hidden_data_idx, block_size});
  reserve_data->mutable_data<T>(ctx.GetPlace());
  SplitReserveData(reserve_data, gate_data, cell_data, cell_act_data,
                   hidden_data, direction_num, time_step, batch_size,
                   hidden_size, gate_num, num_layers);
}

template <typename CellType, template <typename, typename> class LayerT,
          template <typename, typename> class SingleLayerT,
          template <typename, typename> class BidirLayerT, typename T>
void RnnFunc(const framework::ExecutionContext& ctx, const Tensor* input,
             const std::vector<const Tensor*> weight_list, const Tensor* init_h,
             const Tensor* init_c, const Tensor* sequence_length,
             Tensor* last_h, Tensor* last_c, Tensor* output,
             Tensor* dropout_mask, const int& num_layers, const int& gate_num,
             const int& input_size, const int& hidden_size,
             const bool& is_bidirec, const std::string& cell_type,
             const float& dropout_prob, const bool& is_test, const int& seed,
             Tensor* reserve_data) {
  bool is_lstm = true;
  if (last_c == nullptr && init_c == nullptr) {
    is_lstm = false;
  }
  const int& direction_num = is_bidirec ? 2 : 1;
  const auto& init_h_dims = init_h->dims();
  PADDLE_ENFORCE_EQ(init_h_dims[0], num_layers * direction_num,
                    platform::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of init hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers, init_h_dims[0]));
  if (is_lstm) {
    const auto& init_c_dims = init_c->dims();
    PADDLE_ENFORCE_EQ(init_c_dims[0], num_layers * direction_num,
                      platform::errors::InvalidArgument(
                          "The num_layers of in RNN layer must be the same as "
                          "first dim of cell state hidden, but received"
                          " num_layers:%d, dim:%d",
                          num_layers, init_h_dims[0]));
  }
  CellType cell;

  std::vector<TensorList> parameter_lists;
  parameter_lists.reserve(num_layers);
  reset_parameter_vector(weight_list, num_layers, gate_num, is_bidirec,
                         &parameter_lists);

  Tensor gate_data, cell_data, cell_act_data, hidden_data;

  if (!is_test) {
    AllocateReserveData<CellType, T>(
        ctx, reserve_data, &gate_data, &cell_data, &cell_act_data, &hidden_data,
        input, is_bidirec, num_layers, gate_num, hidden_size);
    gate_data.Resize({num_layers, gate_data.numel() / num_layers});
    cell_data.Resize({num_layers, cell_data.numel() / num_layers});
    cell_act_data.Resize({num_layers, cell_act_data.numel() / num_layers});

    if (num_layers > 1) {
      hidden_data.Resize(
          {num_layers - 1, hidden_data.numel() / (num_layers - 1)});
    }
  }
  Tensor* input_holder;
  Tensor* output_holder = output;
  Tensor temp;
  bool has_allocate_mem = false;

  auto init_h_unbind = Unbind(*init_h);
  auto last_h_unbind = Unbind(*last_h);
  TensorList init_c_unbind, last_c_unbind;
  if (is_lstm) {
    init_c_unbind = Unbind(*init_c);
    last_c_unbind = Unbind(*last_c);
  }

  Tensor curr_gate_data, curr_cell_data, curr_cell_act_data;
  Tensor curr_hidden_data, prev_hidden_data;
  for (int i = 0; i < num_layers; i++) {
    if (!is_test) {
      if (cell_data.numel() > 0) /** for lstm, gru **/ {
        curr_cell_data = cell_data.Slice(i, i + 1);
        curr_cell_act_data = cell_act_data.Slice(i, i + 1);
      }
      curr_gate_data = gate_data.Slice(i, i + 1);
      output_holder = output;
      if (i < num_layers - 1 && num_layers > 1) {
        curr_hidden_data = hidden_data.Slice(i, i + 1);
        curr_hidden_data.Resize(output->dims());
        output_holder = &curr_hidden_data;
      }
    }
    if (i > 0) {
      if (!has_allocate_mem) {
        temp.Resize(output->dims());
        temp.mutable_data<T>(ctx.GetPlace());
        input_holder = &temp;
        has_allocate_mem = true;
      }
      if (!is_test) {
        prev_hidden_data = hidden_data.Slice(i - 1, i);
        input_holder = &prev_hidden_data;
        input_holder->Resize(output->dims());
      } else {
        SwapPoniter(&output_holder, &input_holder);
      }
      if (dropout_prob != 0 && (!is_test)) {
        dropout_cpu_function_inplace<T>(ctx, input_holder, dropout_mask,
                                        dropout_prob, seed, is_test);
      }
    }
    const Tensor* input_temp_holder = input;
    if (i > 0) {
      input_temp_holder = input_holder;
    }
    LayerT<T, CellType>* layer;
    SingleLayerT<T, CellType> slayer(cell);
    BidirLayerT<T, CellType> blayer(cell);
    if (is_bidirec) {
      layer = &blayer;
    } else {
      layer = &slayer;
    }
    (*layer)(ctx, input_temp_holder, parameter_lists[i], init_h_unbind,
             init_c_unbind, sequence_length, last_h_unbind, last_c_unbind,
             output_holder, i, gate_num, &curr_gate_data, &curr_cell_data,
             &curr_cell_act_data, is_test);
  }
  if (num_layers % 2 == 0) {
    framework::TensorCopy(
        *output_holder, ctx.GetPlace(),
        ctx.template device_context<platform::CPUDeviceContext>(), output);
  }
}

template <typename DeviceContext, typename T>
class RNNCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    auto state = ctx.MultiOutput<Tensor>("State");
    auto* output = ctx.Output<Tensor>("Out");
    auto* dropout_mask = ctx.Output<Tensor>("DropoutState");
    auto* reserve_data = ctx.Output<Tensor>("Reserve");

    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& input_size = ctx.Attr<int>("input_size");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const float& dropout_prob = ctx.Attr<float>("dropout_prob");
    const std::string& mode = ctx.Attr<std::string>("mode");
    const bool& is_test = ctx.Attr<bool>("is_test");
    const int& seed = ctx.Attr<int>("seed");

    bool has_seq_length = ctx.HasInput("SequenceLength");
    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());
    int gate_num = 4;
    if (mode == "LSTM") {
      state[0]->mutable_data<T>(ctx.GetPlace());
      state[1]->mutable_data<T>(ctx.GetPlace());
      RnnFunc<LSTMCell<T>, Layer, SingleLayer, BidirLayer, T>(
          ctx, input, weight_list, pre_state[0], pre_state[1], sequence_length,
          state[0], state[1], output, dropout_mask, num_layers, gate_num,
          input_size, hidden_size, is_bidirec, mode, dropout_prob, is_test,
          seed, reserve_data);
    }
  }
};

}  // namespace operators
}  // namespace paddle
