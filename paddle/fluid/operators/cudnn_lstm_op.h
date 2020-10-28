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
#include <memory>
#include <string>
#include <type_traits>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/activation_op.h"
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

void SwapPoniter(Tensor** a, Tensor** b) {
  Tensor* c = *a;
  *a = *b;
  *b = c;
}

template <typename T>
void create_mask_matrix(const framework::ExecutionContext& context,
                        const Tensor* sequence_length, Tensor* mask_matrix,
                        const bool& is_reverse) {
  const auto& seq_len_vec = GetDataFromTensor<int>(sequence_length);
  const int& table_width = mask_matrix->dims()[0];
  VLOG(2) << "INPUT MASK TENSOR SHAPE:" << mask_matrix->dims();
  Tensor temp;
  temp.Resize(
      framework::make_ddim({mask_matrix->dims()[1], mask_matrix->dims()[0]}));
  T* data_temp = temp.mutable_data<T>(context.GetPlace());
  std::fill(data_temp, data_temp + mask_matrix->numel(), static_cast<T>(1.0));
  for (unsigned int i = 0; i < seq_len_vec.size(); i++) {
    // reset the mask matrix
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
  // Print2DTensor<T>(&temp, "Original mask Tensor");
  // transpose the result for the mask
  mask_matrix->mutable_data<T>(context.GetPlace());
  std::vector<int> trans_vec;
  trans_vec.emplace_back(1);
  trans_vec.emplace_back(0);
  auto& dev_ctx = context.template device_context<platform::CPUDeviceContext>();
  TransCompute<platform::CPUDeviceContext, T>(2, dev_ctx, temp, mask_matrix,
                                              trans_vec);
}

template <typename T>
void dropout_cpu_grad_function_inplace(
    const framework::ExecutionContext& context, Tensor* grad_x,
    const Tensor* mask, const float& dropout_prob) {
  auto& place = *context.template device_context<platform::CPUDeviceContext>()
                     .eigen_device();
  auto M = EigenVector<uint8_t>::Flatten(*mask);
  auto dX = EigenVector<T>::Flatten(*grad_x);
  if (dropout_prob == 1.0f) {
    dX.device(place) = static_cast<T>(0) * dX;
  } else {
    dX.device(place) = dX * M.cast<T>() / static_cast<T>(1.0f - dropout_prob);
  }
}

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
template <typename T>
void parameter_split(const Tensor* weight, const int& gate_num,
                     const int& layers_num, const int& input_size,
                     const int& hidden_size, const int& is_bidirec,
                     std::vector<TensorList>* params_vec) {
  // if the weight of RNN is flatten, we need to convert the
  // the flattened weight to single split tensor
  const auto& weight_numel = weight->numel();
  // resize the weight tensor, could slice tensor directly
  const auto& mem_block_size = gate_num * hidden_size;
  Tensor weight_shared;
  weight_shared.ShareDataWith(*weight);
  weight_shared.Resize(framework::make_ddim(
      {static_cast<int64_t>(weight_numel / mem_block_size), mem_block_size}));

  // the calcluate the offset of tensor
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& first_input_w_stride = input_size;
  const int& other_input_w_stride = hidden_size * direction_num;
  const int& hidden_w_stride = hidden_size;
  const int& bias_offset =
      direction_num *
      (first_input_w_stride + hidden_w_stride +
       (layers_num - 1) * (other_input_w_stride + hidden_w_stride));

  for (int i = 0; i < layers_num; ++i) {
    TensorList tensor_list;
    // parameter for the w_hi, w_hh, bias_hi, bias_hh
    const int& weight_size = 4 * direction_num;
    tensor_list.reserve(weight_size);
    for (int j = 0; j < weight_size; ++j) {
      const int& section = j / 4;
      int k = j % 4;
      if (k < 2) {
        int start_idx = 0;
        int end_idx = 0;
        if (i == 0) {
          start_idx = section * (hidden_w_stride + first_input_w_stride) +
                      (k % 2) * first_input_w_stride;
          end_idx =
              start_idx + (k == 0 ? first_input_w_stride : hidden_w_stride);
        } else {
          start_idx = direction_num * (hidden_w_stride + first_input_w_stride) +
                      (i - 1) * direction_num *
                          (hidden_w_stride + other_input_w_stride) +
                      section * (hidden_w_stride + other_input_w_stride) +
                      (k % 2) * other_input_w_stride;
          end_idx =
              start_idx + (k == 0 ? other_input_w_stride : hidden_w_stride);
        }
        auto tmp_tensor = weight_shared.Slice(start_idx, end_idx);
        tmp_tensor.Resize(
            framework::make_ddim({tmp_tensor.dims()[1], tmp_tensor.dims()[0]}));
        tensor_list.emplace_back(tmp_tensor);
      } else {
        const auto& start_idx =
            bias_offset + i * 2 * direction_num + section * 2 + k % 2;
        auto tmp_tensor = weight_shared.Slice(start_idx, start_idx + 1);
        tmp_tensor.Resize(
            framework::make_ddim({tmp_tensor.dims()[1], tmp_tensor.dims()[0]}));
        tensor_list.emplace_back(tmp_tensor);
      }
    }
    params_vec->emplace_back(tensor_list);
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

template <typename T>
struct Cell {
  virtual ~Cell() {}
  virtual void operator()(const platform::CPUDeviceContext* device_ctx,
                          Tensor* input, const Tensor* weight_hh,
                          const Tensor* init_h, const Tensor* init_c,
                          Tensor* last_h, Tensor* last_c, Tensor* last_c_act,
                          Tensor* output) {
    VLOG(2) << "Calling Base Cell !!!!!";
  }
};

template <typename T, template <typename> class ActivationFunctor>
struct SimpleRNNCell : Cell<T> {
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

    // activate
    auto hidden = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(input, "Input", "hidden", "Activation"));
    auto* place = device_ctx->eigen_device();
    ActivationFunctor<T> functor;
    functor(*place, hidden, hidden);
  }
};

template <typename T>
struct LSTMCell : Cell<T> {
  void operator()(const platform::CPUDeviceContext* device_ctx, Tensor* input,
                  const Tensor* weight_hh, const Tensor* init_h,
                  const Tensor* init_c, Tensor* last_h, Tensor* last_c,
                  Tensor* last_c_act, Tensor* output) override {
    VLOG(2) << "Calling LSTM Cell !!!!!";
    VLOG(2) << "input shape: " << input->dims();
    VLOG(2) << "w_hh shape: " << weight_hh->dims();
    VLOG(2) << "init_h shape: " << init_h->dims();
    VLOG(2) << "init_c shape: " << init_c->dims();
    VLOG(2) << "last_h shape: " << last_h->dims();
    VLOG(2) << "last_c shape: " << last_c->dims();
    VLOG(2) << "output shape: " << output->dims();
    // Print3DTensor<T>(input, "Cell Input");
    // Print3DTensor<T>(init_h, "init_h");
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(*device_ctx);
    auto mat_dim_a = math::CreateMatrixDescriptor(init_h->dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(weight_hh->dims(), 0, true);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    // convert the batch matmul to matmul, this operator could be speed faster
    blas.MatMul(*init_h, mat_dim_a, *weight_hh, mat_dim_b, static_cast<T>(1.0),
                input, static_cast<T>(1.0));

    // Print3DTensor<T>(input, "Cell Input after");

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
    framework::TensorCopy(*output, device_ctx->GetPlace(), *device_ctx, last_h);
    // Print3DTensor<T>(last_h, "last_h");
  }
};

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
                   Tensor* last_c, const Tensor& mask_tensor) {
    // in the output, if mask flag is 0, we will retun the zero data
    auto eigen_output =
        framework::EigenMatrix<T>::Reshape(*output, output->dims().size() - 1);
    auto eigen_mask = framework::EigenMatrix<T>::From(
        mask_tensor, framework::make_ddim({mask_tensor.dims()[1], 1}));
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto eigen_mask_broadcast =
        eigen_mask.broadcast(Eigen::DSizes<int, 2>(1, output->dims()[1]));

    eigen_output.device(place) = eigen_output * eigen_mask_broadcast;

    auto eigen_init_h =
        framework::EigenMatrix<T>::Reshape(*init_h, init_h->dims().size() - 1);
    auto eigen_last_h =
        framework::EigenMatrix<T>::Reshape(*last_h, last_h->dims().size() - 1);
    eigen_last_h.device(place) = eigen_last_h * eigen_mask_broadcast +
                                 eigen_init_h * (1 - eigen_mask_broadcast);

    auto eigen_init_c =
        framework::EigenMatrix<T>::Reshape(*init_c, init_c->dims().size() - 1);
    auto eigen_last_c =
        framework::EigenMatrix<T>::Reshape(*last_c, last_c->dims().size() - 1);
    eigen_last_c.device(place) = eigen_last_c * eigen_mask_broadcast +
                                 eigen_init_c * (1 - eigen_mask_broadcast);
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
    if (init_c.size() > 0 && last_c->size() > 0) {
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
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));

      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse);
      // Print2DTensor<T>(&mask_matrix, "Mask Matrix");
      mask_tensor_list = Unbind(mask_matrix);
    }

    // define the init_h holder for the swap
    bool has_allocate_mem = false;
    TensorList last_h = *last_h_ptr;
    TensorList last_c = *last_c_ptr;
    TensorList cell_value_tensors;
    TensorList cell_act_value_tensors;

    Tensor* init_h_holder = nullptr;
    Tensor init_h_temp;
    Tensor* last_h_holder = &last_h[layer_idx];
    Tensor* init_c_holder = nullptr;
    Tensor init_c_temp;
    Tensor* last_c_holder = &last_c[layer_idx];

    for (int i = 0; i < time_step; i++) {
      if (i > 0) {
        if (!has_allocate_mem) {
          init_h_temp.Resize(init_h[layer_idx].dims());
          init_h_temp.mutable_data<T>(context.GetPlace());
          init_h_holder = &init_h_temp;
          init_c_temp.Resize(init_c[layer_idx].dims());
          init_c_temp.mutable_data<T>(context.GetPlace());
          init_c_holder = &init_c_temp;
          has_allocate_mem = true;
        }
        SwapPoniter(&init_c_holder, &last_c_holder);
        SwapPoniter(&init_h_holder, &last_h_holder);
      }
      if (i == 0) {
        cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4],
              &init_h[layer_idx], &init_c[layer_idx], last_h_holder,
              last_c_holder, nullptr, &output_tensors[i]);
      } else {
        cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4], init_h_holder,
              init_c_holder, last_h_holder, last_c_holder, nullptr,
              &output_tensors[i]);
      }
      if (has_sequence_length) {
        if (i == 0) {
          this->postprocess(context, &output_tensors[i], &init_h[layer_idx],
                            &init_c[layer_idx], last_h_holder, last_c_holder,
                            mask_tensor_list[i]);
        } else {
          this->postprocess(context, &output_tensors[i], init_h_holder,
                            init_c_holder, last_h_holder, last_c_holder,
                            mask_tensor_list[i]);
        }
      }
    }
    if (time_step % 2 == 0) {
      framework::TensorCopy(*last_h_holder, context.GetPlace(), dev_ctx,
                            &last_h[layer_idx]);
      framework::TensorCopy(*last_c_holder, context.GetPlace(), dev_ctx,
                            &last_c[layer_idx]);
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
    }
    bool is_reverse = false;
    if (is_bidirect) {
      layer_idx = 2 * layer_idx + offset;
      if (offset > 0) {
        is_reverse = true;
      }
    }
    bool is_lstm = false;
    if (init_c.size() > 0 && last_c->size() > 0) {
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
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));

      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse);
      // Print2DTensor<T>(&mask_matrix, "Mask Matrix");
      mask_tensor_list = Unbind(mask_matrix);
    }

    // define the init_h holder for the swap
    bool has_allocate_mem = false;
    TensorList last_h = *last_h_ptr;
    TensorList last_c = *last_c_ptr;
    TensorList cell_value_tensors;
    TensorList cell_act_value_tensors;

    Tensor* init_h_holder = nullptr;
    Tensor init_h_temp;
    Tensor* last_h_holder = &last_h[layer_idx];
    Tensor* init_c_holder = nullptr;
    Tensor init_c_temp;
    Tensor* last_c_holder = &last_c[layer_idx];

    cell_value->Resize({time_step, cell_value->numel() / time_step});
    cell_value_tensors = Unbind(*cell_value);
    cell_act_value->Resize({time_step, cell_value->numel() / time_step});
    cell_act_value_tensors = Unbind(*cell_act_value);
    for (int i = 0; i < time_step; i++) {
      cell_value_tensors[i].Resize(init_c[layer_idx].dims());
      cell_act_value_tensors[i].Resize(init_c[layer_idx].dims());
      if (i > 0) {
        if (!has_allocate_mem) {
          init_h_temp.Resize(init_h[layer_idx].dims());
          init_h_temp.mutable_data<T>(context.GetPlace());
          init_h_holder = &init_h_temp;
          has_allocate_mem = true;
        }
        SwapPoniter(&init_h_holder, &last_h_holder);
      }
      if (i == 0) {
        cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4],
              &init_h[layer_idx], &init_c[layer_idx], last_h_holder,
              &cell_value_tensors[i], &cell_act_value_tensors[i],
              &output_tensors[i]);
      } else {
        cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4], init_h_holder,
              &cell_value_tensors[i - 1], last_h_holder, &cell_value_tensors[i],
              &cell_act_value_tensors[i], &output_tensors[i]);
      }
      if (has_sequence_length) {
        if (i == 0) {
          this->postprocess(context, &output_tensors[i], &init_h[layer_idx],
                            &init_c[layer_idx], last_h_holder,
                            &cell_value_tensors[i], mask_tensor_list[i]);
        } else {
          this->postprocess(context, &output_tensors[i], init_h_holder,
                            &cell_value_tensors[i - 1], last_h_holder,
                            &cell_value_tensors[i], mask_tensor_list[i]);
        }
      }
    }
    if (time_step % 2 == 0) {
      framework::TensorCopy(*last_h_holder, context.GetPlace(), dev_ctx,
                            &last_h[layer_idx]);
    }
    framework::TensorCopy(cell_value_tensors[time_step - 1], context.GetPlace(),
                          dev_ctx, &last_c[layer_idx]);
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

// TODO(zhoushunjie)
inline void SplitReserveData(Tensor* reserve_data, Tensor* gate_data,
                             Tensor* cell_data, Tensor* cell_act_data,
                             Tensor* hidden_data, int direction_num,
                             const int& time_step, const int& batch_size,
                             const int& hidden_size, const int& gate_num,
                             const int& num_layers) {
  const int& block_size = direction_num * time_step * batch_size * hidden_size;
  const int& gate_data_idx = (gate_num - 1) * num_layers;
  const int& cell_data_idx = gate_num * num_layers;
  const int& cell_act_data_idx = (gate_num + 1) * num_layers;
  const int& hidden_data_idx = (gate_num + 1) * num_layers + (num_layers - 1);
  reserve_data->Resize({hidden_data_idx, block_size});
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

template <typename CellType, typename T>
void AllocateReserveData(const framework::ExecutionContext& ctx,
                         Tensor* reserve_data, Tensor* gate_data,
                         Tensor* cell_data, Tensor* cell_act_data,
                         Tensor* hidden_data, const Tensor* input,
                         bool is_bidirec, int num_layers, int gate_num,
                         int hidden_size) {
  if (std::is_same<CellType, LSTMCell<T>>::value) {
    // need to store cell value
    gate_num += 1;
  }
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& time_step = input->dims()[0];
  const int& batch_size = input->dims()[1];
  // gate_data: 4 * num_layers * block_size
  // cell_data: num_layers * block_size
  // hidden_data: (num_layers - 1) * block_size
  VLOG(0) << "===========================";
  VLOG(0) << "gate_num: " << gate_num << ", num_layers: " << num_layers
          << ", direction_num:" << direction_num << ", time_step: " << time_step
          << ", input_size: " << batch_size << ", hidden_size:" << hidden_size;

  const int& block_size = direction_num * time_step * batch_size * hidden_size;
  const int& hidden_data_idx = (gate_num + 1) * num_layers + (num_layers - 1);
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
  // check the dim message of init state
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
        framework::TensorCopy(
            prev_hidden_data, ctx.GetPlace(),
            ctx.template device_context<platform::CPUDeviceContext>(),
            input_holder);
        input_holder->Resize(output->dims());

      } else {
        SwapPoniter(&output_holder, &input_holder);
      }
      if (dropout_prob != 0 && (!is_test)) {
        // only train mode just need dropout
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
    // the final result is in output_holder, must copy the data to output
    framework::TensorCopy(
        *output_holder, ctx.GetPlace(),
        ctx.template device_context<platform::CPUDeviceContext>(), output);
  }
}

template <typename DeviceContext, typename T>
class CudnnLSTMCPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string& cell_type = ctx.Attr<std::string>("cell_type");
    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& input_size = ctx.Attr<int>("input_size");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const float& dropout_prob = ctx.Attr<float>("dropout_prob");
    const bool& is_test = ctx.Attr<bool>("is_test");
    const int& seed = ctx.Attr<int>("seed");
    // get the input and weight tensor for the cacluate rnn cell
    auto* input = ctx.Input<Tensor>("Input");
    auto* init_h = ctx.Input<Tensor>("InitH");
    auto* init_c = ctx.Input<Tensor>("InitC");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");

    bool has_seq_length = ctx.HasInput("SequenceLength");
    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }
    // auto* sequence_length = ctx.Input<Tensor>("SequenceLength");
    auto* last_h = ctx.Output<Tensor>("LastH");
    auto* last_c = ctx.Output<Tensor>("LastC");
    auto* output = ctx.Output<Tensor>("Out");
    auto* dropout_mask = ctx.Output<Tensor>("StateOut");

    // store gate value for backward computing
    auto* reserve_data = ctx.Output<Tensor>("Reserve");

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());
    last_h->mutable_data<T>(ctx.GetPlace());

    int gate_num = 4;
    if (cell_type == "lstm") {
      last_c->mutable_data<T>(ctx.GetPlace());
      RnnFunc<LSTMCell<T>, Layer, SingleLayer, BidirLayer, T>(
          ctx, input, weight_list, init_h, init_c, sequence_length, last_h,
          last_c, output, dropout_mask, num_layers, gate_num, input_size,
          hidden_size, is_bidirec, cell_type, dropout_prob, is_test, seed,
          reserve_data);
    } else if (cell_type == "gru") {
      gate_num = 3;
      // run gru
    } else if (cell_type == "rnn_relu") {
      gate_num = 0;
      // run rnn
      last_c = nullptr;
      init_c = nullptr;
      RnnFunc<SimpleRNNCell<T, ReluFunctor>, Layer, SingleLayer, BidirLayer, T>(
          ctx, input, weight_list, init_h, init_c, sequence_length, last_h,
          last_c, output, dropout_mask, num_layers, gate_num, input_size,
          hidden_size, is_bidirec, cell_type, dropout_prob, is_test, seed,
          reserve_data);
    } else if (cell_type == "rnn_tanh") {
      gate_num = 0;
      last_c = nullptr;
      init_c = nullptr;
      RnnFunc<SimpleRNNCell<T, TanhFunctor>, Layer, SingleLayer, BidirLayer, T>(
          ctx, input, weight_list, init_h, init_c, sequence_length, last_h,
          last_c, output, dropout_mask, num_layers, gate_num, input_size,
          hidden_size, is_bidirec, cell_type, dropout_prob, is_test, seed,
          reserve_data);
    }
  }
};

template <typename T>
void create_lstm_value(math::LstmMetaValue<T>* lstm_value) {
  lstm_value->check_ig = nullptr;
  lstm_value->check_fg = nullptr;
  lstm_value->check_og = nullptr;
}

template <typename T>
void create_lstm_grad(math::LstmMetaGrad<T>* lstm_grad) {
  lstm_grad->check_ig_grad = nullptr;
  lstm_grad->check_fg_grad = nullptr;
  lstm_grad->check_og_grad = nullptr;
}

template <typename T>
struct GradLayer {
  virtual ~GradLayer() {}
  virtual void operator()(
      const framework::ExecutionContext& context, const Tensor* input,
      const Tensor* output, const TensorList& init_h_unbind,
      const TensorList& init_c_unbind, const TensorList& last_h_grad_unbind,
      const TensorList& last_c_grad_unbind,
      const TensorList& gate_tensor_unbind,
      const TensorList& state_tensor_unbind,
      const TensorList& act_state_tensor_unbind, const Tensor* output_grad,
      const std::vector<TensorList>& parameter_lists,
      const Tensor* sequence_length, Tensor* input_grad,
      TensorList* init_h_grad_unbind, TensorList* init_c_grad_unbind,
      const std::vector<TensorList>& weight_list_grad, const int& layer_idx,
      const int& gate_num) {}
  void preprocess(const framework::ExecutionContext& context,
                  const Tensor* grad_output, Tensor* grad_last_h) {
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto eigen_grad_output = framework::EigenMatrix<T>::Reshape(
        *grad_output, grad_output->dims().size() - 1);
    auto eigen_grad_last_h = framework::EigenMatrix<T>::Reshape(
        *grad_last_h, grad_last_h->dims().size() - 1);
    // the output gradient contribute the gradient to last_h
    eigen_grad_last_h.device(place) = eigen_grad_last_h + eigen_grad_output;
  }
  void mask_preprocess(const framework::ExecutionContext& context,
                       const Tensor* grad_output, Tensor* grad_last_h,
                       Tensor* grad_last_c, Tensor* grad_pre_h,
                       Tensor* grad_pre_c, const Tensor& mask_tensor) {
    auto& place = *context.template device_context<platform::CPUDeviceContext>()
                       .eigen_device();
    auto eigen_mask = framework::EigenMatrix<T>::From(
        mask_tensor, framework::make_ddim({mask_tensor.dims()[1], 1}));
    auto eigen_mask_broadcast =
        eigen_mask.broadcast(Eigen::DSizes<int, 2>(1, grad_output->dims()[2]));

    auto eigen_grad_last_h = framework::EigenMatrix<T>::Reshape(
        *grad_last_h, grad_last_h->dims().size() - 1);
    auto eigen_grad_last_c = framework::EigenMatrix<T>::Reshape(
        *grad_last_c, grad_last_c->dims().size() - 1);
    auto eigen_grad_pre_h = framework::EigenMatrix<T>::Reshape(
        *grad_pre_h, grad_pre_h->dims().size() - 1);
    auto eigen_grad_pre_c = framework::EigenMatrix<T>::Reshape(
        *grad_pre_c, grad_pre_c->dims().size() - 1);
    auto eigen_grad_output = framework::EigenMatrix<T>::Reshape(
        *grad_output, grad_output->dims().size() - 1);

    eigen_grad_pre_h.device(place) =
        (1 - eigen_mask_broadcast) * eigen_grad_last_h;
    eigen_grad_pre_c.device(place) =
        (1 - eigen_mask_broadcast) * eigen_grad_last_c;
    eigen_grad_last_h.device(place) = eigen_mask_broadcast * eigen_grad_last_h;
    eigen_grad_last_c.device(place) = eigen_mask_broadcast * eigen_grad_last_c;

    // the output gradient contribute the gradient to last_h
    eigen_grad_last_h.device(place) =
        eigen_grad_last_h + eigen_mask_broadcast * eigen_grad_output;
  }

  void postprocess(const framework::ExecutionContext& context,
                   const Tensor& grad_gate, const Tensor& input,
                   Tensor* input_grad, const TensorList& parameters,
                   const TensorList& grad_parameters) {
    // we get the grad_gate step by step, and need to bradocast the grad to the
    // grad_Whi
    // grad_Bias_hi, grad_Bias_hh
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(device_ctx);

    // calc the gradient for the W_hi
    auto mat_dim_out_grad =
        math::CreateMatrixDescriptor(grad_gate.dims(), 0, true);
    auto mat_dim_input = math::CreateMatrixDescriptor(input.dims(), 0, false);
    mat_dim_out_grad.height_ *= mat_dim_out_grad.batch_size_;
    mat_dim_input.height_ *= mat_dim_input.batch_size_;
    blas.MatMul(grad_gate, mat_dim_out_grad, input, mat_dim_input,
                static_cast<T>(1.0), &grad_parameters[0], T(0));

    // calc the gradient for the X
    mat_dim_out_grad = math::CreateMatrixDescriptor(grad_gate.dims(), 0, false);
    mat_dim_out_grad.height_ *= mat_dim_out_grad.batch_size_;
    auto mat_dim_parameter =
        math::CreateMatrixDescriptor(parameters[0].dims(), 0, false);
    blas.MatMul(grad_gate, mat_dim_out_grad, parameters[0], mat_dim_parameter,
                static_cast<T>(1.0), input_grad, T(0));

    // calc the gradient of Bias_hi, Bias_hh
    math::ColwiseSum<platform::CPUDeviceContext, T> col_sum;
    col_sum(device_ctx, grad_gate, &grad_parameters[2]);
    col_sum(device_ctx, grad_gate, &grad_parameters[3]);
  }
};

template <typename T, typename GradCellType>
struct SingleGradLayer : GradLayer<T> {
  explicit SingleGradLayer(GradCellType& cell) : cell_(cell) {}
  virtual ~SingleGradLayer() {}
  void operator()(
      const framework::ExecutionContext& context, const Tensor* input,
      const Tensor* output, const TensorList& init_h_unbind,
      const TensorList& init_c_unbind, const TensorList& last_h_grad_unbind,
      const TensorList& last_c_grad_unbind,
      const TensorList& gate_tensor_unbind,
      const TensorList& state_tensor_unbind,
      const TensorList& act_state_tensor_unbind, const Tensor* output_grad,
      const std::vector<TensorList>& parameter_lists,
      const Tensor* sequence_length, Tensor* input_grad,
      TensorList* init_h_grad_unbind, TensorList* init_c_grad_unbind,
      const std::vector<TensorList>& weight_list_grad, const int& layer_idx,
      const int& gate_num) {
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    const bool& is_bidirec = context.Attr<bool>("is_bidirec");
    const int& time_step = input->dims()[0];
    const int& batch_size = input->dims()[1];
    const int& input_size = input->dims()[2];
    const int& direction_num = is_bidirec ? 2 : 1;
    const int& hidden_size = context.Attr<int>("hidden_size");

    math::SetConstant<platform::CPUDeviceContext, T> zero;

    const bool& has_sequence_length = sequence_length == nullptr ? false : true;
    Tensor mask_matrix;
    TensorList mask_tensor_list;
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));
      create_mask_matrix<T>(context, sequence_length, &mask_matrix, false);
      mask_tensor_list = Unbind(mask_matrix);
    }

    // create lstm_value and lstm_grad
    math::LstmMetaValue<T> lstm_value;
    math::LstmMetaGrad<T> lstm_grad;
    create_lstm_value(&lstm_value);
    create_lstm_grad(&lstm_grad);

    // copy the last_h, last_c for swaping pointer
    Tensor dynamic_grad_last_h;
    Tensor dynamic_grad_last_c;
    dynamic_grad_last_h.Resize(last_h_grad_unbind[layer_idx].dims());
    dynamic_grad_last_h.mutable_data<T>(context.GetPlace());
    dynamic_grad_last_c.Resize(last_c_grad_unbind[layer_idx].dims());
    dynamic_grad_last_c.mutable_data<T>(context.GetPlace());
    framework::TensorCopy(last_h_grad_unbind[layer_idx], context.GetPlace(),
                          &dynamic_grad_last_h);
    framework::TensorCopy(last_c_grad_unbind[layer_idx], context.GetPlace(),
                          &dynamic_grad_last_c);

    // if the init_c init_h grad is nullptr, we will create the tensor
    Tensor dynamic_grad_pre_h;
    Tensor dynamic_grad_pre_c;
    if (init_h_grad_unbind->size() > 0) {
      dynamic_grad_pre_h.ShareDataWith((*init_h_grad_unbind)[layer_idx]);
    } else {
      dynamic_grad_pre_h.Resize(dynamic_grad_last_h.dims());
      dynamic_grad_pre_h.mutable_data<T>(context.GetPlace());
    }
    if (init_c_grad_unbind->size() > 0) {
      dynamic_grad_pre_c.ShareDataWith((*init_h_grad_unbind)[layer_idx]);
    } else {
      dynamic_grad_pre_c.Resize(dynamic_grad_last_c.dims());
      dynamic_grad_pre_c.mutable_data<T>(context.GetPlace());
    }

    Tensor grad_gate_tensor;
    grad_gate_tensor.Resize(gate_tensor_unbind[layer_idx].dims());
    grad_gate_tensor.mutable_data<T>(context.GetPlace());

    Tensor grad_cell_tensor;
    grad_cell_tensor.Resize(output->dims());
    grad_cell_tensor.mutable_data(context.GetPlace());
    zero(device_ctx, &grad_cell_tensor, static_cast<T>(0.0));

    Tensor weight_grad = weight_list_grad[layer_idx][1];
    weight_grad.mutable_data<T>(context.GetPlace());
    zero(device_ctx, &weight_grad, static_cast<T>(0.0));

    // in this section, create the gate_state_grad for the postprocess calculate
    // ubind the output, the output from [time_step, batch_size, hidden_size]
    auto output_tensor_unbind = Unbind(*output);

    auto layer_gate_tensor = gate_tensor_unbind[layer_idx];
    layer_gate_tensor.Resize(
        {time_step * direction_num, batch_size, hidden_size * gate_num});
    auto layer_gate_tensor_unbind = Unbind(layer_gate_tensor);

    // TODO(wawltor)  this is bug for the gru and rnn
    auto layer_state_tensor = state_tensor_unbind[layer_idx];
    layer_state_tensor.Resize(
        {time_step * direction_num, batch_size, hidden_size * gate_num});
    auto layer_state_tensor_unbind = Unbind(layer_gate_tensor);
    layer_state_tensor_unbind.emplace_back(*output);

    auto layer_act_state_tensor = act_state_tensor_unbind[layer_idx];
    layer_act_state_tensor.Resize(
        {time_step * direction_num, batch_size, hidden_size * gate_num});
    auto layer_act_state_tensor_unbind = Unbind(layer_act_state_tensor);

    Tensor* pre_hidden;
    Tensor* pre_state;
    for (int i = time_step - 1; i >= 0; --i) {
      if (has_sequence_length) {
        this->mask_preprocess(context, output_tensor_unbind[i],
                              &dynamic_grad_last_h, &dynamic_grad_last_c,
                              &dynamic_grad_pre_h, &dynamic_grad_pre_c,
                              mask_tensor_list[i]);
      } else {
        this->preprocess(context, output_tensor_unbind[i],
                         &dynamic_grad_last_h);
      }
      if (i == 0) {
        pre_hidden = &init_h_unbind[layer_idx];
        pre_state = &init_c_unbind[layer_idx];
      } else {
        pre_hidden = &output_tensor_unbind[i - 1];
        pre_state = &layer_state_tensor_unbind[i - 1];
      }
      // TODO(wawltor) add the rnn cell
      cell_(context, &layer_gate_tensor_unbind[i],
            &layer_state_tensor_unbind[i], &layer_act_state_tensor_unbind[i],
            &(parameter_lists[layer_idx][1]), pre_hidden, pre_state,
            &dynamic_grad_last_h, &dynamic_grad_last_c, &grad_gate_tensor,
            &(weight_list_grad[layer_idx][1]), &dynamic_grad_pre_h,
            &dynamic_grad_pre_c, &lstm_value, &lstm_grad);
      SwapPoniter(&&dynamic_grad_last_h, &&dynamic_grad_pre_h);
      SwapPoniter(&&dynamic_grad_last_c, &&dynamic_grad_pre_c);
    }

    // postproces for gradient for w_hi, X, bias_hi, bias_hh
    this->postprocess(context, grad_gate_tensor, *input, input_grad,
                      parameter_lists[layer_idx], weight_list_grad[layer_idx]);

    // copy the gradient to init_c init_h
    if ((*init_h_grad_unbind).size() > 0 && time_step % 2 == 0) {
      framework::TensorCopy(dynamic_grad_last_h, context.GetPlace(),
                            &((*init_h_grad_unbind)[layer_idx]));
    }
    if ((*init_c_grad_unbind).size() > 0 && time_step % 2 == 0) {
      framework::TensorCopy(dynamic_grad_last_c, context.GetPlace(),
                            &((*init_c_grad_unbind)[layer_idx]));
    }
  }
  GradCellType cell_;
};

template <typename T, typename GradCellType>
struct BidirGradLayer : GradLayer<T> {
  explicit BidirGradLayer(GradCellType& cell) : cell_(cell) {}
  virtual ~BidirGradLayer() {}
  virtual void operator()(
      const framework::ExecutionContext& context, const Tensor* input,
      const Tensor* output, const TensorList& init_h_unbind,
      const TensorList& init_c_unbind, const TensorList& last_h_grad_unbind,
      const TensorList& last_c_grad_unbind,
      const TensorList& gate_tensor_unbind,
      const TensorList& state_tensor_unbind,
      const TensorList& act_state_tensor_unbind, const Tensor* output_grad,
      const std::vector<TensorList>& parameter_lists,
      const Tensor* sequence_length, Tensor* input_grad,
      TensorList* init_h_grad_unbind, TensorList* init_c_grad_unbind,
      const std::vector<TensorList>& weight_list_grad, const int& layer_idx,
      const int& gate_num) {}
  GradCellType cell_;
};

template <typename T>
struct GradCell {
  virtual ~GradCell() {}
  virtual void operator()(const framework::ExecutionContext& context,
                          const Tensor* gate_tensor, const Tensor* state_tensor,
                          const Tensor* act_state_tensor,
                          const Tensor* weight_hh, const Tensor* pre_hidden,
                          const Tensor* pre_state, const Tensor* grad_hidden,
                          const Tensor* grad_state, Tensor* grad_gate,
                          Tensor* grad_weight_hh, Tensor* grad_pre_hidden,
                          Tensor* grad_pre_state,
                          math::LstmMetaValue<T>* lstm_value,
                          math::LstmMetaGrad<T>* lstm_grad) {}
};

template <typename T>
struct LSTMGradCell : GradCell<T> {
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* gate_tensor, const Tensor* state_tensor,
                  const Tensor* act_state_tensor, const Tensor* weight_hh,
                  const Tensor* pre_hidden, const Tensor* pre_state,
                  const Tensor* grad_hidden, const Tensor* grad_state,
                  Tensor* grad_gate, Tensor* grad_weight_hh,
                  Tensor* grad_pre_hidden, Tensor* grad_pre_state,
                  math::LstmMetaValue<T>* lstm_value,
                  math::LstmMetaGrad<T>* lstm_grad) {
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(device_ctx);
    size_t frame_size = gate_tensor->dims()[2];
    size_t batch_size = gate_tensor->dims()[1];

    lstm_value.gate_value = gate_tensor->data<T>();
    lstm_value.state_value = state_tensor->data<T>();
    lstm_value.state_active_value = act_state_tensor->data<T>();
    lstm_value.prev_state_value = pre_state->data<T>();

    lstm_grad.state_grad = grad_state->data<T>();
    lstm_grad.gate_grad = grad_gate->data<T>();
    lstm_grad.output_grad = grad_hidden->data<T>();
    lstm_grad.prev_state_grad = grad_pre_state->data<T>();

    lstm_value.output_value = nullptr;
    lstm_grad.state_active_grad = nullptr;

    auto gate_act = math::detail::GetActivationType("sigmoid_v2");
    auto state_act = math::detail::GetActivationType("tanh_v2");
    auto cand_act = math::detail::GetActivationType("tanh_v2");

    T cell_clip = 0.0;
    math::LstmUnitGradFunctor<platform::CPUDeviceContext, T>::compute(
        &device_ctx, lstm_value, lstm_grad, frame_size, batch_size, cell_clip,
        gate_act, state_act, cand_act);

    blas.MatMul(*grad_gate, false, *weight_hh, true, static_cast<T>(1.0),
                grad_pre_hidden, static_cast<T>(1.0));

    blas.MatMul(*pre_hidden, true, *grad_gate, false, static_cast<T>(1.0),
                grad_weight_hh, static_cast<T>(1.0));
    // trans the gradient to Whh, pre_hidden
  }
};

template <typename GradCellType,
          template <typename, typename> class SingleGradLayerT,
          template <typename, typename> class BidirGradLayerT, typename T>
void RnnGradFunc(const framework::ExecutionContext& ctx, const int& gate_num,
                 const int& cell_num) {
  // get the tensor pointer for the input
  auto* input = ctx.Input<Tensor>("Input");
  auto weight_list = ctx.MultiInput<Tensor>("WeightList");
  auto* init_h = ctx.Input<Tensor>("InitH");
  auto* init_c = ctx.Input<Tensor>("InitC");
  auto* reserve_state = ctx.Input<Tensor>("Reserve");
  auto* state_out = ctx.Input<Tensor>("StateOut");
  auto* output = ctx.Input<Tensor>("Out");
  auto* output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
  auto* last_h_grad = ctx.Input<Tensor>(framework::GradVarName("LastH"));
  auto* last_c_grad = ctx.Input<Tensor>(framework::GradVarName("LastC"));

  bool has_seq_length = ctx.HasInput("SequenceLength");
  const Tensor* sequence_length = nullptr;
  if (has_seq_length) {
    sequence_length = ctx.Input<Tensor>("SequenceLength");
  }

  // get the tensor pointer for the output
  auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
  auto weight_grad_list =
      ctx.MultiOutput<framework::Tensor>(framework::GradVarName("WeightList"));
  auto* init_h_grad = ctx.Output<Tensor>(framework::GradVarName("InitH"));
  auto* init_c_grad = ctx.Output<Tensor>(framework::GradVarName("InitC"));

  // get the attributes for the calcluate
  const int& num_layers = ctx.Attr<int>("num_layers");
  const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
  const float& dropout_prob = ctx.Attr<float>("dropout_prob");
  const bool& is_test = ctx.Attr<bool>("is_test");

  // get the input_size, batch_size, time_step, hidden_size
  const int& time_step = input->dims()[0];
  const int& batch_size = input->dims()[1];
  const int& input_size = input->dims()[2];
  const int& hidden_size = ctx.Attr<int>("hidden_size");
  const int& direction_num = is_bidirec ? 2 : 1;

  // allocate the memory
  input_grad->mutable_data<T>(input->dims(), ctx.GetPlace());
  if (init_h_grad) init_h_grad->mutable_data<T>(init_h->dims(), ctx.GetPlace());
  if (init_c_grad) init_c_grad->mutable_data<T>(init_c->dims(), ctx.GetPlace());

  // reset the parameter to sorted
  std::vector<TensorList> parameter_lists;
  parameter_lists.reserve(num_layers);
  reset_parameter_vector(weight_list, num_layers, gate_num, is_bidirec,
                         &parameter_lists);
  std::vector<TensorList> parameter_lists_grad;
  parameter_lists_grad.reserve(num_layers);
  reset_parameter_vector(weight_grad_list, num_layers, gate_num, is_bidirec,
                         &parameter_lists_grad);

  // resolve the state of reverse_state
  const int& block_size = time_step * batch_size * hidden_size * direction_num;
  // NOTICE *******
  // reserve_state->Resize(framework::make_ddim({reserve_state->numel()/block_size,
  //    block_size}));
  Tensor gate_tensor;
  Tensor state_tensor;
  Tensor act_state_tensor;
  Tensor hidden_tensor;
  gate_tensor = reserve_state->Slice(0, gate_num * num_layers);
  gate_tensor.Resize({num_layers, time_step * direction_num, batch_size,
                      hidden_size * gate_num});
  if (cell_num >= 1) {
    state_tensor = state_tensor.Slice(gate_num * num_layers,
                                      (gate_num + cell_num) * num_layers);
    act_state_tensor = state_tensor.Slice(
        gate_num * num_layers, (gate_num + 2 * cell_num) * num_layers);
    state_tensor = state_tensor.Resize({num_layers, time_step * direction_num,
                                        batch_size, hidden_size * gate_num});
    act_state_tensor =
        state_tensor.Resize({num_layers, time_step * direction_num, batch_size,
                             hidden_size * gate_num});
  }
  if (num_layers > 1) {
    hidden_tensor = reserve_state->Slice(
        (gate_num + 2 * cell_num) * num_layers,
        (gate_num + 2 * cell_num) * num_layers + num_layers - 1);
    hidden_tensor.Resize({num_layers - 1, time_step * direction_num, batch_size,
                          hidden_size * gate_num});
  }
  // unbind
  auto last_h_grad_unbind = Unbind(*last_h_grad);
  auto last_c_grad_unbind = Unbind(*last_c_grad);
  auto gate_tensor_unbind = Unbind(gate_tensor);

  std::vector<Tensor> init_h_unbind;
  std::vector<Tensor> init_c_unbind;
  std::vector<Tensor> init_h_grad_unbind;
  std::vector<Tensor> init_c_grad_unbind;
  std::vector<Tensor> state_tensor_unbind;
  std::vector<Tensor> act_state_tensor_unbind;
  std::vector<Tensor> hidden_tensor_unbind;

  init_h_unbind = Unbind(*init_h);
  init_c_unbind = Unbind(*init_c);

  if (init_h_grad != nullptr) {
    init_h_grad_unbind = Unbind(*init_h_grad);
  }
  if (init_c_grad != nullptr) {
    init_c_grad_unbind = Unbind(*init_c_grad);
  }
  if (cell_num > 1) {
    state_tensor_unbind = Unbind(state_tensor);
    act_state_tensor_unbind = Unbind(state_tensor);
  }
  if (num_layers > 1) {
    hidden_tensor_unbind = Unbind(hidden_tensor);
  }
  // add the output tensor to the hidden vector
  Tensor tmp;
  hidden_tensor_unbind.emplace_back(tmp);
  hidden_tensor_unbind[num_layers - 1].ShareDataWith(*output);

  GradCellType cell;
  Tensor* layer_input;
  Tensor* layer_output;
  Tensor* layer_input_grad_holder = nullptr;
  Tensor* layer_output_grad_holder = output_grad;
  Tensor input_grad_temp;
  Tensor output_grad_temp;

  bool has_allocate_mem = false;
  for (int i = num_layers - 1; i >= 0; --i) {
    // the layer input output had saved, just use the data
    if (i > 0) {
      layer_input->ShareDataWith(hidden_tensor_unbind[i - 1]);
    } else {
      layer_input->ShareDataWith(*input);
    }
    layer_output->ShareDataWith(hidden_tensor_unbind[i]);
    if (num_layers == 1) {
      layer_input_grad_holder = input_grad;
    } else {
      if (i == num_layers - 1) {
        input_grad_temp.Resize(layer_input->dims());
        input_grad_temp.mutable_data<T>(ctx.GetPlace());
        layer_input_grad_holder = &input_grad_temp;
      }
    }
    if (is_bidirec) {
      BidirGradLayerT<T, GradCellType> layer(cell);
    } else {
      SingleGradLayerT<T, GradCellType> layer(cell);
      layer(ctx, layer_input, layer_output, init_h_unbind, init_c_unbind,
            last_h_grad_unbind, last_c_grad_unbind, gate_tensor_unbind,
            state_tensor_unbind, layer_output_grad_holder, parameter_lists,
            sequence_length, layer_input_grad_holder, &init_h_grad_unbind,
            &init_c_grad_unbind, parameter_lists_grad, i, gate_num);
    }

    // calcluate the dropout gradient for the layer_input_grad_holder
    // state_out save in the forward process
    if (i > 0) {
      if ((!is_test) && (dropout_prob != 0)) {
        dropout_cpu_grad_function_inplace<T>(ctx, layer_input_grad_holder,
                                             state_out, dropout_prob);
      }
    }

    if (i - 1 == 0) {
      layer_output_grad_holder = input_grad;
    } else {
      if (!has_allocate_mem) {
        output_grad_temp.Resize(layer_input_grad_holder->dims());
        output_grad_temp.mutable_data<T>(ctx.GetPlace());
        layer_output_grad_holder = output_grad_temp;
      }
    }
    SwapPoniter(&layer_input_grad_holder, &layer_output_grad_holder);
  }
}

template <typename DeviceContext, typename T>
class CudnnLSTMCPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    const std::string& cell_type = ctx.Attr<std::string>("cell_type");
    int gate_num = 4;
    int cell_num = 1;
    if (cell_type == "lstm") {
      RnnGradFunc<LSTMGradCell<T>, SingleGradLayer, BidirGradLayer, T>(
          ctx, gate_num, cell_num);
    }
  }
};
}  // namespace operators
}  // namespace paddle
