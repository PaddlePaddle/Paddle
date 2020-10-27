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

using Tensor = framework::Tensor;
using TensorList = std::vector<framework::Tensor>;

void SwapPoniter(Tensor** a, Tensor** b) {
  Tensor* c = *a;
  *a = *b;
  *b = c;
}

template <typename T>
void create_mask_matrix(const framework::ExecutionContext& context,
                        const Tensor* sequence_length, Tensor* mask_matrix,
                        const bool& is_reverse, int* min_seq_len) {
  const auto& seq_len_vec = GetDataFromTensor<int>(sequence_length);
  const int& table_width = mask_matrix->dims()[0];
  // VLOG(2) << "INPUT MASK TENSOR SHAPE:" << mask_matrix->dims();
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
void dropout_cpu_function_inplace(const framework::ExecutionContext& context,
                                  Tensor* x, Tensor* mask,
                                  const float& dropout_prob,
                                  const int& seed_number,
                                  const bool& upscale_in_train,
                                  const bool& is_test) {
  // VLOG(2) << "dropout_cpu_function_inplace function";
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
          if (upscale_in_train) {
            x_data[i] /= static_cast<T>(1.0f - dropout_prob);
          }
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
        if (upscale_in_train) {
          x_data[i] /= static_cast<T>(1.0f - dropout_prob);
        }
      } else {
        x_data[i] = static_cast<T>(0);
      }
    }
  } else {
    if (!upscale_in_train) {
      auto X = EigenMatrix<T>::Reshape(*x, 1);
      auto& place =
          *context.template device_context<platform::CPUDeviceContext>()
               .eigen_device();
      X.device(place) = X * static_cast<T>(1.0f - dropout_prob);
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

void reset_parameter_vector(const std::vector<const Tensor*>& raw_params_vec,
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
                          Tensor* last_h, Tensor* last_c, Tensor* output) {}
};

template <typename T>
struct LSTMCell : Cell<T> {
  void operator()(const platform::CPUDeviceContext* device_ctx, Tensor* input,
                  const Tensor* weight_hh, const Tensor* init_h,
                  const Tensor* init_c, Tensor* last_c, Tensor* output) {
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(*device_ctx);
    auto mat_dim_a = math::CreateMatrixDescriptor(init_h->dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(weight_hh->dims(), 0, true);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    // convert the batch matmul to matmul, this operator could be speed faster
    blas.MatMul(*init_h, mat_dim_a, *weight_hh, mat_dim_b, static_cast<T>(1.0),
                input, T(1.0));

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
    cell_pre_act.mutable_data<T>(init_h->dims(), device_ctx->GetPlace());

    lstm_value.prev_state_value = init_c->data<T>();
    lstm_value.gate_value = input->data<T>();
    lstm_value.output_value = output->data<T>();
    lstm_value.state_value = last_c->data<T>();
    lstm_value.state_active_value = cell_pre_act.data<T>();
    T cell_clip = 0.0;
    math::LstmUnitFunctor<platform::CPUDeviceContext, T>::compute(
        *device_ctx, lstm_value, frame_size, batch_size, cell_clip, gate_act,
        cell_act, cand_act, false);
  }
};

template <typename T>
struct Layer {
  virtual ~Layer() {}
  void preprocess(const framework::ExecutionContext& context,
                  const Tensor* input, const Tensor& weight,
                  const Tensor& bias_ih, const Tensor& bias_hh,
                  Tensor* cache_input) {
    // crate the temp input for the X * W_ih^T + Bias_ih
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    const int& hidden_size = weight.dims()[0];
    cache_input->Resize(framework::make_ddim(
        {input->dims()[0], input->dims()[1], hidden_size}));
    cache_input->mutable_data<T>(context.GetPlace());
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(dev_ctx);
    auto mat_dim_a = math::CreateMatrixDescriptor(input->dims(), 0, false);
    auto mat_dim_b = math::CreateMatrixDescriptor(weight.dims(), 0, true);
    // convert the batch matmul to matmul, this operator could be speed faster
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    blas.MatMul(*input, mat_dim_a, weight, mat_dim_b, static_cast<T>(1.0),
                cache_input, T(0));

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
                   const Tensor* init_c, Tensor* last_h, Tensor* last_c,
                   const Tensor& mask_tensor) {
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

    auto eigen_last_h =
        framework::EigenMatrix<T>::Reshape(*last_h, last_h->dims().size() - 1);
    eigen_last_h.device(place) =
        eigen_output + eigen_last_h * (1 - eigen_mask_broadcast);

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
                          const int& layer_idx, const int& gate_num) {}
};

template <typename T, typename CellType>
struct SingleLayer : Layer<T> {
  explicit SingleLayer(CellType& cell) : cell_(cell) {}
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const TensorList& init_h, const TensorList& init_c,
                  const Tensor* sequence_length, TensorList last_h,
                  TensorList last_c, Tensor* output, const int& layer_idx,
                  const int& gate_num) {
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    // first step, we could calcalute the W_ih * input + Bias_ih to more faster
    const int& time_step = input->dims()[0];
    // vec[0] is parameter of w_hi, vec[2] is bias of b_hi
    Tensor input_w;
    this->preprocess(context, input, vec[0], vec[2], vec[3], &input_w);
    // VLOG(2) << "output shape: " << output->dims();

    auto input_tensors = Unbind(input_w);
    auto output_tensors = Unbind(*output);
    TensorList mask_tensor_list;
    // construct the mask matrix for the mask
    bool has_sequence_length = false;
    if (sequence_length != nullptr) {
      has_sequence_length = true;
    }
    Tensor mask_matrix;
    int min_seq_len = 0;
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));
      create_mask_matrix<T>(context, sequence_length, &mask_matrix, false,
                            &min_seq_len);
      mask_tensor_list = Unbind(mask_matrix);
    }

    // define the init_h holder for the swap
    bool has_allocate_mem = false;
    Tensor* init_c_holder = nullptr;
    Tensor init_c_temp;
    Tensor* last_c_holder = &last_c[layer_idx];
    for (int i = 0; i < time_step; i++) {
      if (i > 0) {
        if (!has_allocate_mem) {
          init_c_temp.Resize(init_c[layer_idx].dims());
          init_c_temp.mutable_data<T>(context.GetPlace());
          init_c_holder = &init_c_temp;
          has_allocate_mem = true;
        }
        SwapPoniter(&init_c_holder, &last_c_holder);
      }

      if (i == 0) {
        cell_(&dev_ctx, &input_tensors[i], &vec[1], &init_h[layer_idx],
              &init_c[layer_idx], last_c_holder, &output_tensors[i]);
      } else {
        cell_(&dev_ctx, &input_tensors[i], &vec[1], &output_tensors[i - 1],
              init_c_holder, last_c_holder, &output_tensors[i]);
      }
      if (has_sequence_length && i >= min_seq_len) {
        // copy data to last_h
        if (i == min_seq_len) {
          if (i == 0) {
            framework::TensorCopy(init_h[layer_idx], dev_ctx.GetPlace(),
                                  dev_ctx, &last_h[layer_idx]);
          } else {
            framework::TensorCopy(output_tensors[i - 1], dev_ctx.GetPlace(),
                                  dev_ctx, &last_h[layer_idx]);
          }
        }
        if (i == 0) {
          this->postprocess(context, &output_tensors[i], &init_c[layer_idx],
                            &last_h[layer_idx], last_c_holder,
                            mask_tensor_list[i]);
        } else {
          this->postprocess(context, &output_tensors[i], init_c_holder,
                            &last_h[layer_idx], last_c_holder,
                            mask_tensor_list[i]);
        }
      }
    }
    if (!has_sequence_length) {
      framework::TensorCopy(output_tensors[time_step - 1], dev_ctx.GetPlace(),
                            dev_ctx, &last_h[layer_idx]);
    }
    if (time_step % 2 == 0) {
      framework::TensorCopy(*last_c_holder, context.GetPlace(), dev_ctx,
                            &last_c[layer_idx]);
    }
  }

  // Cell for the rnn module
  CellType cell_;
};

template <typename T, typename CellType>
struct BidirLayer : Layer<T> {
  explicit BidirLayer(CellType& cell) : cell_(cell) {}
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const TensorList& init_h, const TensorList& init_c,
                  const Tensor* sequence_length, TensorList last_h,
                  TensorList last_c, Tensor* output, const int& layer_idx,
                  const int& gate_num) {
    TensorList output_vec;
    Tensor temp_forward, temp_backward;
    output_vec.reserve(2);
    output_vec.emplace_back(temp_forward);
    output_vec.emplace_back(temp_backward);
    auto& dev_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    // first step, calculate the fw_ih * input + fb_ih to more faster
    const int& time_step = input->dims()[0];
    const int& batch_size = input->dims()[1];
    const int& hidden_size = output->dims()[2];
    Tensor forward_input_w;
    this->preprocess(context, input, vec[0], vec[2], vec[3], &forward_input_w);
    auto forward_input_tensors = Unbind(forward_input_w);

    output_vec[0].Resize(
        framework::make_ddim({time_step, batch_size, hidden_size / 2}));
    output_vec[0].mutable_data<T>(context.GetPlace());
    auto forward_output_tensors = Unbind(output_vec[0]);

    // if has the sequence, build mask tensor list
    bool has_sequence_length = false;
    if (sequence_length != nullptr) {
      has_sequence_length = true;
    }
    Tensor forward_mask_matrix;
    TensorList forward_mask_tensor_list;
    int min_seq_len = 0;
    if (has_sequence_length) {
      forward_mask_matrix.Resize(
          framework::make_ddim({time_step, input->dims()[1]}));
      create_mask_matrix<T>(context, sequence_length, &forward_mask_matrix,
                            false, &min_seq_len);
      forward_mask_tensor_list = Unbind(forward_mask_matrix);
    }
    bool has_forward_allocate_mem = false;
    Tensor* forward_init_c_holder = nullptr;
    Tensor forward_init_c_temp;
    Tensor* forward_last_c_holder = &last_c[2 * layer_idx];
    for (int i = 0; i < time_step; i++) {
      if (i > 0) {
        if (!has_forward_allocate_mem) {
          forward_init_c_temp.Resize(init_c[2 * layer_idx].dims());
          forward_init_c_temp.mutable_data<T>(context.GetPlace());
          forward_init_c_holder = &forward_init_c_temp;
          has_forward_allocate_mem = true;
        }
        SwapPoniter(&forward_init_c_holder, &forward_last_c_holder);
      }
      if (i == 0) {
        cell_(&dev_ctx, &forward_input_tensors[i], &vec[1],
              &init_h[2 * layer_idx], &init_c[2 * layer_idx],
              &last_c[2 * layer_idx], &forward_output_tensors[i]);
      } else {
        cell_(&dev_ctx, &forward_input_tensors[i], &vec[1],
              &forward_output_tensors[i - 1], forward_init_c_holder,
              forward_last_c_holder, &forward_output_tensors[i]);
      }
      if (has_sequence_length && i >= min_seq_len) {
        if (i == min_seq_len) {
          if (i == 0) {
            framework::TensorCopy(init_h[2 * layer_idx], dev_ctx.GetPlace(),
                                  dev_ctx, &last_h[2 * layer_idx]);
          } else {
            framework::TensorCopy(forward_output_tensors[i - 1],
                                  dev_ctx.GetPlace(), dev_ctx,
                                  &last_h[2 * layer_idx]);
          }
        }

        if (i == 0) {
          this->postprocess(context, &forward_output_tensors[i],
                            &init_c[2 * layer_idx], &last_h[2 * layer_idx],
                            forward_last_c_holder, forward_mask_tensor_list[i]);
        } else {
          this->postprocess(context, &forward_output_tensors[i],
                            forward_init_c_holder, &last_h[2 * layer_idx],
                            forward_last_c_holder, forward_mask_tensor_list[i]);
        }
      }
    }
    // create the temp output for the forward
    output_vec[1].Resize(
        framework::make_ddim({time_step, batch_size, hidden_size / 2}));
    output_vec[1].mutable_data<T>(context.GetPlace());
    auto backward_output_tensors = Unbind(output_vec[1]);
    std::reverse(backward_output_tensors.begin(),
                 backward_output_tensors.end());

    // second step, we calcluate the bw_ih * reverse_input + bw_ih
    Tensor backward_input_w;
    this->preprocess(context, input, vec[4], vec[6], vec[7], &backward_input_w);
    auto backward_input_tensors = Unbind(backward_input_w);
    std::reverse(backward_input_tensors.begin(), backward_input_tensors.end());
    Tensor backward_mask_matrix;
    backward_mask_matrix.Resize(
        framework::make_ddim({time_step, input->dims()[1]}));
    TensorList backward_mask_tensor_list;
    if (has_sequence_length) {
      create_mask_matrix<T>(context, sequence_length, &backward_mask_matrix,
                            true, &min_seq_len);
      backward_mask_tensor_list = Unbind(backward_mask_matrix);
    }
    bool has_backward_allocate_mem = false;
    Tensor* backward_init_c_holder = nullptr;
    Tensor backward_init_c_temp;
    Tensor* backward_last_c_holder = &last_c[2 * layer_idx + 1];
    for (int i = 0; i < time_step; i++) {
      if (i > 0) {
        if (!has_backward_allocate_mem) {
          backward_init_c_temp.Resize(init_c[2 * layer_idx + 1].dims());
          backward_init_c_temp.mutable_data<T>(context.GetPlace());
          backward_init_c_holder = &backward_init_c_temp;
          has_backward_allocate_mem = true;
        }
        SwapPoniter(&backward_init_c_holder, &backward_last_c_holder);
      }
      if (i == 0) {
        cell_(&dev_ctx, &backward_input_tensors[i], &vec[5],
              &init_h[2 * layer_idx + 1], &init_c[2 * layer_idx + 1],
              &last_c[2 * layer_idx + 1], &backward_output_tensors[i]);
      } else {
        if (has_sequence_length && i < time_step - min_seq_len) {
          cell_(&dev_ctx, &backward_input_tensors[i], &vec[5],
                &last_h[2 * layer_idx + 1], backward_init_c_holder,
                backward_last_c_holder, &backward_output_tensors[i]);
        } else {
          cell_(&dev_ctx, &backward_input_tensors[i], &vec[5],
                &backward_output_tensors[i - 1], backward_init_c_holder,
                backward_last_c_holder, &backward_output_tensors[i]);
        }
      }
      if (has_sequence_length && i < time_step - min_seq_len) {
        if (i == 0) {
          framework::TensorCopy(init_h[2 * layer_idx + 1], dev_ctx.GetPlace(),
                                dev_ctx, &last_h[2 * layer_idx + 1]);
          this->postprocess(context, &backward_output_tensors[i],
                            &init_c[2 * layer_idx + 1],
                            &last_h[2 * layer_idx + 1], backward_last_c_holder,
                            backward_mask_tensor_list[i]);
        } else {
          this->postprocess(context, &backward_output_tensors[i],
                            backward_init_c_holder, &last_h[2 * layer_idx + 1],
                            backward_last_c_holder,
                            backward_mask_tensor_list[i]);
        }
      }
    }
    if (!has_sequence_length) {
      framework::TensorCopy(forward_output_tensors[time_step - 1],
                            dev_ctx.GetPlace(), dev_ctx,
                            &last_h[2 * layer_idx]);
    }
    framework::TensorCopy(backward_output_tensors[time_step - 1],
                          dev_ctx.GetPlace(), dev_ctx,
                          &last_h[2 * layer_idx + 1]);
    // concat the the output result
    paddle::operators::math::ConcatFunctor<platform::CPUDeviceContext, T>
        concat_functor;
    concat_functor(dev_ctx, output_vec, static_cast<int>(2), output);

    if (time_step % 2 == 0) {
      framework::TensorCopy(
          *forward_last_c_holder, context.GetPlace(),
          context.template device_context<platform::CPUDeviceContext>(),
          &last_c[2 * layer_idx]);
      framework::TensorCopy(
          *backward_last_c_holder, context.GetPlace(),
          context.template device_context<platform::CPUDeviceContext>(),
          &last_c[2 * layer_idx + 1]);
    }
  }

  CellType cell_;
};

template <typename CellType, template <typename, typename> class SingleLayerT,
          template <typename, typename> class BidirLayerT, typename T>
void RnnFunc(const framework::ExecutionContext& ctx, const Tensor* input,
             const std::vector<const Tensor*> weight_list, const Tensor* init_h,
             const Tensor* init_c, const Tensor* sequence_length,
             Tensor* last_h, Tensor* last_c, Tensor* output,
             Tensor* dropout_mask, const int& num_layers, const int& gate_num,
             const int& input_size, const int& hidden_size,
             const bool& is_bidirec, const std::string& cell_type,
             const float& dropout_prob, const bool& is_test, const int& seed) {
  // check the dim message of init state
  const int& direction_num = is_bidirec ? 2 : 1;
  const auto& init_h_dims = init_h->dims();
  const auto& init_c_dims = init_c->dims();
  PADDLE_ENFORCE_EQ(init_h_dims[0], num_layers * direction_num,
                    platform::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of init hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers, init_h_dims[0]));
  PADDLE_ENFORCE_EQ(init_c_dims[0], num_layers * direction_num,
                    platform::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of cell state hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers, init_h_dims[0]));
  CellType cell;
  std::vector<TensorList> parameter_lists;
  parameter_lists.reserve(num_layers);
  reset_parameter_vector(weight_list, num_layers, gate_num, is_bidirec,
                         &parameter_lists);

  Tensor* input_holder;
  Tensor* output_holder = output;
  Tensor temp;
  bool has_allocate_mem = false;

  auto init_h_unbind = Unbind(*init_h);
  auto init_c_unbind = Unbind(*init_c);
  auto last_h_unbind = Unbind(*last_h);
  auto last_c_unbind = Unbind(*last_c);

  for (int i = 0; i < num_layers; i++) {
    if (i > 0) {
      if (!has_allocate_mem) {
        temp.Resize(output->dims());
        temp.mutable_data<T>(ctx.GetPlace());
        input_holder = &temp;
        has_allocate_mem = true;
      }
      SwapPoniter(&output_holder, &input_holder);
      if (dropout_prob != 0 && (!is_test)) {
        // only train mode just need dropout
        dropout_cpu_function_inplace<T>(ctx, input_holder, dropout_mask,
                                        dropout_prob, seed,
                                        true /*upscale_in_train*/, is_test);
      }
    }
    if (is_bidirec) {
      BidirLayerT<T, CellType> layer(cell);
      if (i == 0) {
        layer(ctx, input, parameter_lists[i], init_h_unbind, init_c_unbind,
              sequence_length, last_h_unbind, last_c_unbind, output_holder, i,
              gate_num);
      } else {
        layer(ctx, input_holder, parameter_lists[i], init_h_unbind,
              init_c_unbind, sequence_length, last_h_unbind, last_c_unbind,
              output_holder, i, gate_num);
      }
    } else {
      SingleLayerT<T, CellType> layer(cell);
      if (i == 0) {
        layer(ctx, input, parameter_lists[i], init_h_unbind, init_c_unbind,
              sequence_length, last_h_unbind, last_c_unbind, output_holder, i,
              gate_num);
      } else {
        layer(ctx, input_holder, parameter_lists[i], init_h_unbind,
              init_c_unbind, sequence_length, last_h_unbind, last_c_unbind,
              output_holder, i, gate_num);
      }
    }
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
    auto* last_h = ctx.Output<Tensor>("LastH");
    auto* last_c = ctx.Output<Tensor>("LastC");
    auto* output = ctx.Output<Tensor>("Out");
    auto* dropout_mask = ctx.Output<Tensor>("StateOut");

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());
    last_h->mutable_data<T>(ctx.GetPlace());
    last_c->mutable_data<T>(ctx.GetPlace());
    if (cell_type == "lstm") {
      RnnFunc<LSTMCell<T>, SingleLayer, BidirLayer, T>(
          ctx, input, weight_list, init_h, init_c, sequence_length, last_h,
          last_c, output, dropout_mask, num_layers, 4, input_size, hidden_size,
          is_bidirec, cell_type, dropout_prob, is_test, seed);
    }
  }
};

}  // namespace operators
}  // namespace paddle
