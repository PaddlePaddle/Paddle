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
#include "paddle/fluid/operators/math/concat_and_split.h"
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

double total_time = 0.0;

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
                          Tensor* output) {}
};

template <typename T, template <typename> class EigenActivationFunctor,
          math::detail::ActivationType act_type>
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
//    size_t frame_size = init_h->dims()[2];

#ifdef __AVX__
//    if (!(frame_size & (8 - 1)) && (std::is_same<T, float>::value)) {
//      // run avx
//      VLOG(0) << "run avx";
//      auto start = system_clock::now();
//      __m256* z = reinterpret_cast<__m256*>(input->data<T>());
//      __m256* hidden = reinterpret_cast<__m256*>(last_h->data<T>());
//      int num = input->numel();
//      for(int i = 0; i < num / 8; ++i) {
//        hidden[i] = math::detail::forward::activation(z[i], act_type);
//      }
//      auto end = system_clock::now();
//      auto duration = duration_cast<microseconds>(end - start);
//      total_time += double(duration.count()) * microseconds::period::num /
//      microseconds::period::den;
//      framework::TensorCopy(*last_h, device_ctx->GetPlace(), *device_ctx,
//      output);
//      return;
//    }
#endif
    //    VLOG(0) << "run eigen";
    // activate
    //    auto start = system_clock::now();
    auto z = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(input, "Input", "z", "Activation"));
    auto hidden = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(last_h, "Input", "hidden", "Activation"));

    auto* place = device_ctx->eigen_device();
    EigenActivationFunctor<T> functor;
    functor(*place, z, hidden);
    // VLOG(0) << "hidden_data: ";
    // Print3DTensor<T>(last_h, "last_h");
    //    auto end = system_clock::now();
    //    auto duration = duration_cast<microseconds>(end - start);
    //    total_time += double(duration.count()) * microseconds::period::num /
    //                  microseconds::period::den;
    framework::TensorCopy(*last_h, device_ctx->GetPlace(), *device_ctx, output);
  }
};

template <typename T>
struct LSTMCell : Cell<T> {
  void operator()(const platform::CPUDeviceContext* device_ctx, Tensor* input,
                  const Tensor* weight_hh, const Tensor* init_h,
                  const Tensor* init_c, Tensor* last_h, Tensor* last_c,
                  Tensor* last_c_act, Tensor* output) override {
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
                   Tensor* last_c, const Tensor& mask_tensor, bool is_lstm) {
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
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));

      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse);
      // Print2DTensor<T>(&mask_matrix, "Mask Matrix");
      mask_tensor_list = Unbind(mask_matrix);
    }

    // define the init_h holder for the swap
    bool has_allocate_mem = false;

    Tensor* init_h_holder = nullptr;
    Tensor init_h_temp;
    Tensor* last_h_holder = &(*last_h_ptr)[layer_idx];
    Tensor* init_c_holder = nullptr;
    Tensor init_c_temp;
    Tensor* last_c_holder = nullptr;
    const Tensor* init_h_temp_holder = &init_h[layer_idx];
    const Tensor* init_c_temp_holder = nullptr;

    if (is_lstm) {
      last_c_holder = &(*last_c_ptr)[layer_idx];
      init_c_temp_holder = &init_c[layer_idx];
    }

    for (int i = 0; i < time_step; i++) {
      if (i > 0) {
        if (!has_allocate_mem) {
          init_h_temp.Resize(init_h[layer_idx].dims());
          init_h_temp.mutable_data<T>(context.GetPlace());
          init_h_holder = &init_h_temp;
          if (is_lstm) {
            init_c_temp.Resize(init_c[layer_idx].dims());
            init_c_temp.mutable_data<T>(context.GetPlace());
            init_c_holder = &init_c_temp;
          }
          has_allocate_mem = true;
        }
        SwapPoniter(&init_c_holder, &last_c_holder);
        SwapPoniter(&init_h_holder, &last_h_holder);
        init_h_temp_holder = init_h_holder;
        init_c_temp_holder = init_c_holder;
      }
      cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4],
            init_h_temp_holder, init_c_temp_holder, last_h_holder,
            last_c_holder, nullptr, &output_tensors[i]);
      if (has_sequence_length) {
        this->postprocess(context, &output_tensors[i], init_h_temp_holder,
                          init_c_temp_holder, last_h_holder, last_c_holder,
                          mask_tensor_list[i], is_lstm);
      }
    }
    if (time_step % 2 == 0) {
      framework::TensorCopy(*last_h_holder, context.GetPlace(), dev_ctx,
                            &(*last_h_ptr)[layer_idx]);
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
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));

      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse);
      mask_tensor_list = Unbind(mask_matrix);
    }

    // define the init_h holder for the swap
    bool has_allocate_mem = false;
    TensorList cell_value_tensors;
    TensorList cell_act_value_tensors;

    Tensor* init_h_holder = nullptr;
    Tensor init_h_temp;
    Tensor* last_h_holder = &(*last_h_ptr)[layer_idx];
    const Tensor* init_c_temp_holder = nullptr;
    const Tensor* init_h_temp_holder = &init_h[layer_idx];
    Tensor* last_c_temp_holder = nullptr;
    Tensor* last_c_act_temp_holder = nullptr;
    if (is_lstm) {
      cell_value->Resize({time_step, cell_value->numel() / time_step});
      cell_value_tensors = Unbind(*cell_value);
      cell_act_value->Resize({time_step, cell_value->numel() / time_step});
      cell_act_value_tensors = Unbind(*cell_act_value);
    }
    for (int i = 0; i < time_step; i++) {
      if (i > 0) {
        if (!has_allocate_mem) {
          init_h_temp.Resize(init_h[layer_idx].dims());
          init_h_temp.mutable_data<T>(context.GetPlace());
          init_h_holder = &init_h_temp;
          has_allocate_mem = true;
        }
        SwapPoniter(&init_h_holder, &last_h_holder);
        init_h_temp_holder = init_h_holder;
      }
      if (is_lstm) {
        if (i == 0) {
          init_c_temp_holder = &init_c[layer_idx];
        } else {
          init_c_temp_holder = &cell_value_tensors[i - 1];
        }
        cell_value_tensors[i].Resize(init_c[layer_idx].dims());
        cell_act_value_tensors[i].Resize(init_c[layer_idx].dims());
        last_c_temp_holder = &cell_value_tensors[i];
        last_c_act_temp_holder = &cell_act_value_tensors[i];
      }

      cell_(&dev_ctx, &input_tensors[i], &vec[1 + offset * 4],
            init_h_temp_holder, init_c_temp_holder, last_h_holder,
            last_c_temp_holder, last_c_act_temp_holder, &output_tensors[i]);
      if (has_sequence_length) {
        this->postprocess(context, &output_tensors[i], init_h_temp_holder,
                          init_c_temp_holder, last_h_holder, last_c_temp_holder,
                          mask_tensor_list[i], is_lstm);
      }
    }
    if (time_step % 2 == 0) {
      framework::TensorCopy(*last_h_holder, context.GetPlace(), dev_ctx,
                            &(*last_h_ptr)[layer_idx]);
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
  /** for lstm and gru **/
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
  // gate_data: 4 * num_layers * block_size
  // cell_data: num_layers * block_size
  // hidden_data: (num_layers - 1) * block_size
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
        // framework::TensorCopy(
        //    prev_hidden_data, ctx.GetPlace(),
        //    ctx.template device_context<platform::CPUDeviceContext>(),
        //    input_holder);
        input_holder = &prev_hidden_data;
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
  // VLOG(0) << "Spend " << total_time * 1000 << " ms";
  total_time = 0;
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
      RnnFunc<
          SimpleRNNCell<T, ReluFunctor, math::detail::ActivationType::kReLU>,
          Layer, SingleLayer, BidirLayer, T>(
          ctx, input, weight_list, init_h, init_c, sequence_length, last_h,
          last_c, output, dropout_mask, num_layers, gate_num, input_size,
          hidden_size, is_bidirec, cell_type, dropout_prob, is_test, seed,
          reserve_data);
    } else if (cell_type == "rnn_tanh") {
      gate_num = 0;
      last_c = nullptr;
      init_c = nullptr;
      RnnFunc<
          SimpleRNNCell<T, TanhFunctor, math::detail::ActivationType::kTanhV2>,
          Layer, SingleLayer, BidirLayer, T>(
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

template <typename T, typename GradCellType>
struct GradLayer {
  explicit GradLayer(const GradCellType& cell) : cell_(cell) {}
  virtual ~GradLayer() {}
  void run_rnn_grad_function(
      const framework::ExecutionContext& context,
      const platform::CPUDeviceContext& device_ctx, const Tensor* input,
      Tensor* input_grad, const Tensor* sequence_length,
      std::vector<Tensor>* init_h_unbind, std::vector<Tensor>* init_c_unbind,
      std::vector<Tensor>* init_h_grad_unbind,
      std::vector<Tensor>* init_c_grad_unbind, Tensor* layer_grad_gate_tensor,
      std::vector<Tensor>* layer_gate_tensor_unbind,
      std::vector<Tensor>* layer_grad_gate_tensor_unbind,
      std::vector<Tensor>* layer_state_tensor_unbind,
      std::vector<Tensor>* layer_act_state_tensor_unbind,
      std::vector<Tensor>* output_tensor_unbind,
      std::vector<Tensor>* output_grad_tensor_unbind,
      const TensorList& last_h_grad_unbind,
      const TensorList& last_c_grad_unbind,
      const std::vector<TensorList>& parameter_lists,
      std::vector<TensorList>* weight_list_grad, const int& layer_idx,
      const int& time_step, const bool& has_sequence_length,
      const bool& is_bidirec, const bool& is_reverse) {
    Print3DTensor<T>(input_grad, "print after zero step 1");

    const int& direction_num = is_bidirec ? 2 : 1;
    const int& current_reverse_idx = is_reverse ? 1 : 0;
    const int& current_layer_idx =
        direction_num * layer_idx + current_reverse_idx;
    int begin_idx = 0;
    if (is_reverse) {
      begin_idx = time_step;
    }

    Tensor mask_matrix;
    TensorList mask_tensor_list;
    if (has_sequence_length) {
      mask_matrix.Resize(framework::make_ddim({time_step, input->dims()[1]}));
      create_mask_matrix<T>(context, sequence_length, &mask_matrix, is_reverse);
      mask_tensor_list = Unbind(mask_matrix);
    }
    Print2DTensor<T>(&mask_matrix, "mask matrix");
    // create lstm_value and lstm_grad
    math::LstmMetaValue<T> lstm_value;
    math::LstmMetaGrad<T> lstm_grad;
    create_lstm_value(&lstm_value);
    create_lstm_grad(&lstm_grad);

    // copy the last_h, last_c for swaping pointer
    Tensor a, b;
    Tensor* dynamic_grad_last_h = &a;
    Tensor* dynamic_grad_last_c = &b;
    dynamic_grad_last_h->Resize(last_h_grad_unbind[current_layer_idx].dims());
    dynamic_grad_last_h->mutable_data<T>(context.GetPlace());
    framework::TensorCopy(last_h_grad_unbind[current_layer_idx],
                          context.GetPlace(), dynamic_grad_last_h);
    VLOG(0) << "last_c_grad_unbind.size = " << last_c_grad_unbind.size();
    if (last_c_grad_unbind.size() > 0) {
      dynamic_grad_last_c->Resize(last_c_grad_unbind[current_layer_idx].dims());
      dynamic_grad_last_c->mutable_data<T>(context.GetPlace());
      framework::TensorCopy(last_c_grad_unbind[current_layer_idx],
                            context.GetPlace(), dynamic_grad_last_c);
    } else {
      dynamic_grad_last_c = nullptr;
    }

    Tensor c, d;
    Tensor* dynamic_grad_pre_h = &c;
    Tensor* dynamic_grad_pre_c = &d;
    if (init_h_grad_unbind->size() > 0) {
      dynamic_grad_pre_h->ShareDataWith(
          (*init_h_grad_unbind)[current_layer_idx]);
    } else {
      dynamic_grad_pre_h->Resize(dynamic_grad_last_h->dims());
      dynamic_grad_pre_h->mutable_data<T>(context.GetPlace());
    }
    VLOG(0) << "init_c_grad_unbind.size = " << init_c_grad_unbind->size();
    if (init_c_grad_unbind->size() > 0) {
      dynamic_grad_pre_c->ShareDataWith(
          (*init_c_grad_unbind)[current_layer_idx]);
    } else {
      // dynamic_grad_pre_c->Resize(dynamic_grad_last_c->dims());
      // dynamic_grad_pre_c->mutable_data<T>(context.GetPlace());
      dynamic_grad_pre_c = nullptr;
    }

    if (is_reverse) {
      // must be reverse the input, output, input_grad, output_grad
      // the gate and grad_gate must be reverse
      std::reverse(layer_gate_tensor_unbind->begin(),
                   layer_gate_tensor_unbind->end());
      std::reverse(layer_grad_gate_tensor_unbind->begin(),
                   layer_grad_gate_tensor_unbind->end());
      /*
      if (has_sequence_length) {
        std::reverse(mask_tensor_list.begin(), mask_tensor_list.end());
      }*/
      std::reverse(output_tensor_unbind->begin(), output_tensor_unbind->end());
      std::reverse(output_grad_tensor_unbind->begin(),
                   output_grad_tensor_unbind->end());
    }

    Tensor* weight_grad =
        &((*weight_list_grad)[layer_idx][current_reverse_idx * 4 + 1]);
    weight_grad->mutable_data<T>(context.GetPlace());
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, weight_grad, static_cast<T>(0.0));

    Tensor* pre_hidden = nullptr;
    Tensor* pre_state = nullptr;
    Tensor* hidden = nullptr;

    for (int i = time_step - 1; i >= 0; --i) {
      if (has_sequence_length) {
        VLOG(0) << "in mask preprocess before";
        this->mask_preprocess(context, &(*output_grad_tensor_unbind)[i],
                              dynamic_grad_last_h, dynamic_grad_last_c,
                              dynamic_grad_pre_h, dynamic_grad_pre_c,
                              mask_tensor_list[i]);
        VLOG(0) << "in mask preprocess after";
      } else {
        this->preprocess(context, &(*output_grad_tensor_unbind)[i],
                         dynamic_grad_last_h);
      }
      VLOG(0) << "layer_idx:" << layer_idx << ", layer step 5";
      hidden = &(*output_tensor_unbind)[i];
      if (i == 0) {
        pre_hidden = &(*init_h_unbind)[current_layer_idx];
        if (init_c_unbind->size() > 0) {
          pre_state = &(*init_c_unbind)[current_layer_idx];
        }
      } else {
        pre_hidden = &(*output_tensor_unbind)[i - 1];
        if (layer_state_tensor_unbind->size() > 0) {
          pre_state = &(*layer_state_tensor_unbind)[begin_idx + i - 1];
        }
      }
      this->cell_(context, &(*layer_gate_tensor_unbind)[i],
                  &(*layer_state_tensor_unbind)[begin_idx + i],
                  &(*layer_act_state_tensor_unbind)[begin_idx + i], hidden,
                  &(parameter_lists[layer_idx][current_reverse_idx * 4 + 1]),
                  pre_hidden, pre_state, dynamic_grad_last_h,
                  dynamic_grad_last_c, &(*layer_grad_gate_tensor_unbind)[i],
                  weight_grad, dynamic_grad_pre_h, dynamic_grad_pre_c,
                  &lstm_value, &lstm_grad, mask_tensor_list[i],
                  has_sequence_length);
      VLOG(0) << "layer_idx:" << layer_idx << ", layer step 6";
      SwapPoniter(&dynamic_grad_last_h, &dynamic_grad_pre_h);
      SwapPoniter(&dynamic_grad_last_c, &dynamic_grad_pre_c);
    }

    Print3DTensor<T>(input_grad, "print after zero step 2");
    Print3DTensor<T>(layer_grad_gate_tensor, "print layer_grad_gate_tensor");
    Print3DTensor<T>(&(*layer_grad_gate_tensor_unbind)[0],
                     "double check print layer_grad_gate_tensor");
    // postproces for gradient for w_hi, X, bias_hi, bias_hh
    this->postprocess(context, *layer_grad_gate_tensor, *input, input_grad,
                      parameter_lists[layer_idx],
                      &((*weight_list_grad)[layer_idx]), is_reverse);

    VLOG(0) << "layer_idx:" << layer_idx << ", layer step 7";
    // copy the gradient to init_c init_h
    if ((*init_h_grad_unbind).size() > 0 && time_step % 2 == 0) {
      framework::TensorCopy(*dynamic_grad_last_h, context.GetPlace(),
                            &((*init_h_grad_unbind)[current_layer_idx]));
    }
    if ((*init_c_grad_unbind).size() > 0 && time_step % 2 == 0) {
      framework::TensorCopy(*dynamic_grad_last_c, context.GetPlace(),
                            &((*init_c_grad_unbind)[current_layer_idx]));
    }
    VLOG(0) << "layer_idx:" << layer_idx << ", layer step 8";
  }

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
    auto eigen_grad_pre_h = framework::EigenMatrix<T>::Reshape(
        *grad_pre_h, grad_pre_h->dims().size() - 1);
    auto eigen_grad_output = framework::EigenMatrix<T>::Reshape(
        *grad_output, grad_output->dims().size() - 1);
    eigen_grad_last_h.device(place) =
        eigen_grad_last_h + eigen_grad_output * eigen_mask_broadcast;
    eigen_grad_pre_h.device(place) =
        (1 - eigen_mask_broadcast) * eigen_grad_last_h;
    Print3DTensor<T>(grad_pre_h, "mask grad_pre_h");
    Print3DTensor<T>(grad_last_h, "mask grad_last_h");
    eigen_grad_last_h.device(place) = eigen_mask_broadcast * eigen_grad_last_h;

    if (grad_last_c && grad_pre_c) {
      auto eigen_grad_last_c = framework::EigenMatrix<T>::Reshape(
          *grad_last_c, grad_last_c->dims().size() - 1);
      auto eigen_grad_pre_c = framework::EigenMatrix<T>::Reshape(
          *grad_pre_c, grad_pre_c->dims().size() - 1);
      eigen_grad_pre_c.device(place) =
          (1 - eigen_mask_broadcast) * eigen_grad_last_c;
      eigen_grad_last_c.device(place) =
          eigen_mask_broadcast * eigen_grad_last_c;
      Print3DTensor<T>(grad_pre_c, "mask grad_pre_c");
    }
  }

  void postprocess(const framework::ExecutionContext& context,
                   const Tensor& grad_gate, const Tensor& input,
                   Tensor* input_grad, const TensorList& parameters,
                   TensorList* grad_parameters, const int& is_reverse) {
    // we get the grad_gate step by step, and need to bradocast the grad to the
    // grad_w_hi, grad_bias_hi, grad_bias_hh
    int begin_idx = 0;
    if (is_reverse) {
      begin_idx = 4;
    }
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(device_ctx);

    // calc the gradient for the w_hi
    auto mat_dim_out_grad =
        math::CreateMatrixDescriptor(grad_gate.dims(), 0, true);
    auto mat_dim_input = math::CreateMatrixDescriptor(input.dims(), 0, false);
    mat_dim_out_grad.width_ *= mat_dim_out_grad.batch_size_;
    mat_dim_out_grad.batch_size_ = 0;
    mat_dim_input.height_ *= mat_dim_input.batch_size_;
    mat_dim_input.batch_size_ = 0;
    blas.MatMul(grad_gate, mat_dim_out_grad, input, mat_dim_input,
                static_cast<T>(1.0), &((*grad_parameters)[begin_idx + 0]),
                T(0));

    // calc the gradient for the X
    auto mat_dim_out_grad_new =
        math::CreateMatrixDescriptor(grad_gate.dims(), 0, false);
    mat_dim_out_grad_new.height_ *= mat_dim_out_grad_new.batch_size_;
    mat_dim_out_grad_new.batch_size_ = 0;
    auto mat_dim_parameter =
        math::CreateMatrixDescriptor(parameters[0].dims(), 0, false);
    blas.MatMul(grad_gate, mat_dim_out_grad_new, parameters[begin_idx + 0],
                mat_dim_parameter, static_cast<T>(1.0), input_grad, T(1));

    // calc the gradient of Bias_hi, Bias_hh
    math::ColwiseSum<platform::CPUDeviceContext, T> col_sum;
    Tensor tmp_grad_gate;
    tmp_grad_gate.ShareDataWith(grad_gate);
    tmp_grad_gate.Resize(
        {grad_gate.dims()[0] * grad_gate.dims()[1], grad_gate.dims()[2]});
    col_sum(device_ctx, tmp_grad_gate, &((*grad_parameters)[begin_idx + 2]));
    col_sum(device_ctx, tmp_grad_gate, &((*grad_parameters)[begin_idx + 3]));
  }
  GradCellType cell_;
};

template <typename T, typename GradCellType>
struct SingleGradLayer : GradLayer<T, GradCellType> {
  // explicit SingleGradLayer(GradCellType& cell) : cell_(cell) {}
  explicit SingleGradLayer(const GradCellType& cell)
      : GradLayer<T, GradCellType>(cell) {}
  virtual ~SingleGradLayer() {}
  void operator()(
      const framework::ExecutionContext& context, const Tensor* input,
      const Tensor* output, std::vector<Tensor>* init_h_unbind,
      std::vector<Tensor>* init_c_unbind, const TensorList& last_h_grad_unbind,
      const TensorList& last_c_grad_unbind,
      const TensorList& gate_tensor_unbind,
      const TensorList& state_tensor_unbind,
      const TensorList& act_state_tensor_unbind, const Tensor* output_grad,
      const std::vector<TensorList>& parameter_lists,
      const Tensor* sequence_length, Tensor* input_grad,
      TensorList* init_h_grad_unbind, TensorList* init_c_grad_unbind,
      std::vector<TensorList>* weight_list_grad, const int& layer_idx,
      const int& gate_num) {
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, input_grad, static_cast<T>(0.0));

    const bool& is_bidirec = context.Attr<bool>("is_bidirec");
    const int& time_step = input->dims()[0];
    const int& batch_size = input->dims()[1];
    const int& direction_num = is_bidirec ? 2 : 1;
    const int& hidden_size = context.Attr<int>("hidden_size");

    // in this section, create the gate_state_grad for the postprocess calculate
    // ubind the output, the output from [time_step, batch_size, hidden_size]
    auto output_tensor_unbind = Unbind(*output);
    auto output_grad_tensor_unbind = Unbind(*output_grad);
    auto layer_gate_tensor = gate_tensor_unbind[layer_idx];
    layer_gate_tensor.Resize(
        {time_step * direction_num, batch_size, hidden_size * gate_num});
    auto layer_gate_tensor_unbind = Unbind(layer_gate_tensor);
    // the gate_tensor and the grad_gate_tensor must be unbind
    Tensor layer_grad_gate_tensor;
    layer_grad_gate_tensor.Resize(layer_gate_tensor.dims());
    layer_grad_gate_tensor.mutable_data<T>(context.GetPlace());
    auto layer_grad_gate_tensor_unbind = Unbind(layer_grad_gate_tensor);

    Tensor layer_state_tensor;
    TensorList layer_state_tensor_unbind;
    if (state_tensor_unbind.size() > 0) {
      layer_state_tensor = state_tensor_unbind[layer_idx];
      layer_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_state_tensor_unbind = Unbind(layer_state_tensor);
    }

    Tensor layer_act_state_tensor;
    TensorList layer_act_state_tensor_unbind;
    if (act_state_tensor_unbind.size() > 0) {
      layer_act_state_tensor = act_state_tensor_unbind[layer_idx];
      layer_act_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_act_state_tensor_unbind = Unbind(layer_act_state_tensor);
    }
    const bool& has_sequence_length = sequence_length == nullptr ? false : true;
    this->run_rnn_grad_function(
        context, device_ctx, input, input_grad, sequence_length, init_h_unbind,
        init_c_unbind, init_h_grad_unbind, init_c_grad_unbind,
        &layer_grad_gate_tensor, &layer_gate_tensor_unbind,
        &layer_grad_gate_tensor_unbind, &layer_state_tensor_unbind,
        &layer_act_state_tensor_unbind, &output_tensor_unbind,
        &output_grad_tensor_unbind, last_h_grad_unbind, last_c_grad_unbind,
        parameter_lists, weight_list_grad, layer_idx, time_step,
        has_sequence_length, is_bidirec, false);
  }
};
template <typename T>
void split_tensor_at_last_dim(const framework::ExecutionContext& context,
                              const platform::CPUDeviceContext& dev_ctx,
                              const Tensor* output,
                              std::vector<Tensor*>* output_vec,
                              const int& axis) {
  std::vector<const framework::Tensor*> shape_refer;
  (*output_vec)[0]->Resize(
      {output->dims()[0], output->dims()[1], output->dims()[2] / 2});
  (*output_vec)[0]->mutable_data<T>(context.GetPlace());
  (*output_vec)[1]->Resize(
      {output->dims()[0], output->dims()[1], output->dims()[2] / 2});
  (*output_vec)[1]->mutable_data<T>(context.GetPlace());
  shape_refer.emplace_back((*output_vec)[0]);
  shape_refer.emplace_back((*output_vec)[1]);
  VLOG(0) << "before in split functor";
  math::SplitFunctor<platform::CPUDeviceContext, T> functor;
  functor(dev_ctx, *output, shape_refer, axis, output_vec);
  VLOG(0) << "after in split functor";
}

template <typename T, typename GradCellType>
struct BidirGradLayer : GradLayer<T, GradCellType> {
  explicit BidirGradLayer(const GradCellType& cell)
      : GradLayer<T, GradCellType>(cell) {}
  // explicit BidirGradLayer(GradCellType& cell) : cell_(cell) {}
  virtual ~BidirGradLayer() {}
  void operator()(
      const framework::ExecutionContext& context, const Tensor* input,
      const Tensor* output, std::vector<Tensor>* init_h_unbind,
      std::vector<Tensor>* init_c_unbind, const TensorList& last_h_grad_unbind,
      const TensorList& last_c_grad_unbind,
      const TensorList& gate_tensor_unbind,
      const TensorList& state_tensor_unbind,
      const TensorList& act_state_tensor_unbind, const Tensor* output_grad,
      const std::vector<TensorList>& parameter_lists,
      const Tensor* sequence_length, Tensor* input_grad,
      TensorList* init_h_grad_unbind, TensorList* init_c_grad_unbind,
      std::vector<TensorList>* weight_list_grad, const int& layer_idx,
      const int& gate_num) {
    const bool& is_bidirec = context.Attr<bool>("is_bidirec");
    const int& time_step = input->dims()[0];
    const int& batch_size = input->dims()[1];
    const int& direction_num = is_bidirec ? 2 : 1;
    const int& hidden_size = context.Attr<int>("hidden_size");
    // split the output two tensor to output_forward, output_backward
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    math::SetConstant<platform::CPUDeviceContext, T> zero;
    zero(device_ctx, input_grad, static_cast<T>(0.0));
    Print3DTensor<T>(input_grad, "print after zero");

    std::vector<Tensor*> output_vec;
    Tensor forward_output;
    Tensor backward_output;
    std::vector<Tensor> forward_output_tensor_unbind;
    std::vector<Tensor> backward_output_tensor_unbind;
    // in the last layer, we will use the output as the last hidden
    // the output just the concat the forward hidden, backward hidden, so just
    // split it
    // in other layer, we just split the hidden in the rows
    output_vec.emplace_back(&forward_output);
    output_vec.emplace_back(&backward_output);
    split_tensor_at_last_dim<T>(context, device_ctx, output, &output_vec, 2);
    forward_output_tensor_unbind = Unbind(*(output_vec[0]));
    backward_output_tensor_unbind = Unbind(*(output_vec[1]));

    std::vector<Tensor*> output_grad_vec;
    Tensor grad_forward_output;
    Tensor grad_backward_output;
    output_grad_vec.emplace_back(&grad_forward_output);
    output_grad_vec.emplace_back(&grad_backward_output);
    split_tensor_at_last_dim<T>(context, device_ctx, output_grad,
                                &output_grad_vec, 2);
    auto forward_output_grad_tensor_unbind = Unbind(*(output_grad_vec[0]));
    auto backward_output_grad_tensor_unbind = Unbind(*(output_grad_vec[1]));

    // the gate_tensor and the grad_gate_tensor must be unbind
    auto layer_gate_tensor = gate_tensor_unbind[layer_idx];
    layer_gate_tensor.Resize(
        {time_step * 2, batch_size, hidden_size * gate_num});
    auto layer_forward_gate_tensor = layer_gate_tensor.Slice(0, time_step);
    auto layer_backward_gate_tensor =
        layer_gate_tensor.Slice(time_step, 2 * time_step);
    auto layer_forward_gate_tensor_unbind = Unbind(layer_forward_gate_tensor);
    auto layer_backward_gate_tensor_unbind = Unbind(layer_backward_gate_tensor);

    Tensor layer_grad_gate_tensor;
    layer_grad_gate_tensor.Resize(layer_gate_tensor.dims());
    layer_grad_gate_tensor.mutable_data<T>(context.GetPlace());
    zero(device_ctx, &layer_grad_gate_tensor, static_cast<T>(0.0));
    auto layer_forward_grad_gate_tensor =
        layer_grad_gate_tensor.Slice(0, time_step);
    auto layer_backward_grad_gate_tensor =
        layer_grad_gate_tensor.Slice(time_step, 2 * time_step);
    auto layer_forward_grad_gate_tensor_unbind =
        Unbind(layer_forward_grad_gate_tensor);
    auto layer_backward_grad_gate_tensor_unbind =
        Unbind(layer_backward_grad_gate_tensor);

    Tensor layer_state_tensor;
    TensorList layer_state_tensor_unbind;
    if (state_tensor_unbind.size() > 0) {
      layer_state_tensor = state_tensor_unbind[layer_idx];
      layer_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_state_tensor_unbind = Unbind(layer_state_tensor);
    }
    Print3DTensor<T>(&layer_state_tensor, "layer_state_tensor check");

    Tensor layer_act_state_tensor;
    TensorList layer_act_state_tensor_unbind;
    if (act_state_tensor_unbind.size() > 0) {
      layer_act_state_tensor = act_state_tensor_unbind[layer_idx];
      layer_act_state_tensor.Resize(
          {time_step * direction_num, batch_size, hidden_size});
      layer_act_state_tensor_unbind = Unbind(layer_act_state_tensor);
    }
    const bool& has_sequence_length = sequence_length == nullptr ? false : true;
    VLOG(0) << "data ready!";

    this->run_rnn_grad_function(
        context, device_ctx, input, input_grad, sequence_length, init_h_unbind,
        init_c_unbind, init_h_grad_unbind, init_c_grad_unbind,
        &layer_forward_grad_gate_tensor, &layer_forward_gate_tensor_unbind,
        &layer_forward_grad_gate_tensor_unbind, &layer_state_tensor_unbind,
        &layer_act_state_tensor_unbind, &forward_output_tensor_unbind,
        &forward_output_grad_tensor_unbind, last_h_grad_unbind,
        last_c_grad_unbind, parameter_lists, weight_list_grad, layer_idx,
        time_step, has_sequence_length, is_bidirec, false);
    VLOG(0) << "process forward run!";

    this->run_rnn_grad_function(
        context, device_ctx, input, input_grad, sequence_length, init_h_unbind,
        init_c_unbind, init_h_grad_unbind, init_c_grad_unbind,
        &layer_backward_grad_gate_tensor, &layer_backward_gate_tensor_unbind,
        &layer_backward_grad_gate_tensor_unbind, &layer_state_tensor_unbind,
        &layer_act_state_tensor_unbind, &backward_output_tensor_unbind,
        &backward_output_grad_tensor_unbind, last_h_grad_unbind,
        last_c_grad_unbind, parameter_lists, weight_list_grad, layer_idx,
        time_step, has_sequence_length, is_bidirec, true);
    VLOG(0) << "process backward run!";
  }
};

template <typename T>
struct GradCell {
  virtual ~GradCell() {}
  virtual void operator()(
      const framework::ExecutionContext& context, Tensor* gate_tensor,
      Tensor* state_tensor, Tensor* act_state_tensor, Tensor* hidden_tensor,
      const Tensor* weight_hh, Tensor* pre_hidden, Tensor* pre_state,
      Tensor* grad_hidden, Tensor* grad_state, Tensor* grad_gate,
      Tensor* grad_weight_hh, Tensor* grad_pre_hidden, Tensor* grad_pre_state,
      math::LstmMetaValue<T>* lstm_value, math::LstmMetaGrad<T>* lstm_grad,
      const Tensor& mask_tensor, bool has_sequence_length) {}
};

template <typename T, template <typename> class EigenActivationBackwardFunctor>
struct SimpleRNNGradCell : GradCell<T> {
  void operator()(const framework::ExecutionContext& context,
                  Tensor* gate_tensor, Tensor* state_tensor,
                  Tensor* act_state_tensor, Tensor* hidden_tensor,
                  const Tensor* weight_hh, Tensor* pre_hidden,
                  Tensor* pre_state, Tensor* grad_hidden, Tensor* grad_state,
                  Tensor* grad_gate, Tensor* grad_weight_hh,
                  Tensor* grad_pre_hidden, Tensor* grad_pre_state,
                  math::LstmMetaValue<T>* lstm_value,
                  math::LstmMetaGrad<T>* lstm_grad, const Tensor& mask_tensor,
                  bool has_sequence_length) override {
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(device_ctx);
    Tensor grad_pre_hidden_bak;
    if (has_sequence_length) {
      grad_pre_hidden_bak.Resize(grad_pre_hidden->dims());
      grad_pre_hidden_bak.mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*grad_pre_hidden, device_ctx.GetPlace(), device_ctx,
                            &grad_pre_hidden_bak);
    }
    VLOG(0) << "Before bp activate:";
    Print3DTensor<T>(pre_hidden, "pre_hidden");
    Print3DTensor<T>(hidden_tensor, "hidden_tensor");
    Print3DTensor<T>(gate_tensor, "gate_tensor");
    Print3DTensor<T>(grad_hidden, "grad_hidden");
    Print3DTensor<T>(grad_gate, "grad_gate");
    // h = act(z)
    // update dz
    auto dz = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(grad_gate, "Output", "dz", "Grad"));
    auto dh = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(grad_hidden, "Input", "dh", "Grad"));
    auto h = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(hidden_tensor, "Input", "h", "Value"));
    // useless, but need this argument to execute functor
    auto z = EigenVector<T>::Flatten(
        GET_DATA_SAFELY(gate_tensor, "Input", "z", "Value"));

    auto* place = device_ctx.eigen_device();
    EigenActivationBackwardFunctor<T> functor;
    functor(*place, z, h, dh, dz);

    VLOG(0) << "After bp activate:";
    Print3DTensor<T>(gate_tensor, "gate_tensor");
    Print3DTensor<T>(grad_hidden, "grad_hidden");
    Print3DTensor<T>(grad_gate, "grad_gate");

    VLOG(0) << "grad gate shape:" << grad_gate->dims();
    VLOG(0) << "weight_hh shape:" << weight_hh->dims();
    VLOG(0) << "pre_hidden shape:" << pre_hidden->dims();

    // update grad_weight_hh, grad_pre_hidden
    auto mat_dim_a = math::CreateMatrixDescriptor(grad_gate->dims(), 0, false);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    auto mat_dim_b = math::CreateMatrixDescriptor(weight_hh->dims(), 0, false);
    blas.MatMul(*grad_gate, mat_dim_a, *weight_hh, mat_dim_b,
                static_cast<T>(1.0), grad_pre_hidden, static_cast<T>(0.0));

    Print3DTensor<T>(grad_pre_hidden, "cell grad_pre_h before");
    if (has_sequence_length) {
      auto eigen_mask = framework::EigenMatrix<T>::From(
          mask_tensor, framework::make_ddim({mask_tensor.dims()[1], 1}));
      auto eigen_mask_broadcast = eigen_mask.broadcast(
          Eigen::DSizes<int, 2>(1, grad_hidden->dims()[2]));
      auto eigen_grad_pre_hidden = framework::EigenMatrix<T>::Reshape(
          *grad_pre_hidden, grad_pre_hidden->dims().size() - 1);
      auto eigen_grad_pre_hidden_bak = framework::EigenMatrix<T>::Reshape(
          grad_pre_hidden_bak, grad_pre_hidden_bak.dims().size() - 1);
      eigen_grad_pre_hidden.device(*place) =
          (1 - eigen_mask_broadcast) * eigen_grad_pre_hidden_bak +
          eigen_grad_pre_hidden * eigen_mask_broadcast;
    }
    Print3DTensor<T>(grad_pre_hidden, "cell grad_pre_h");
    Print3DTensor<T>(grad_hidden, "cell grad_hidden");

    auto mat_dim_c = math::CreateMatrixDescriptor(grad_gate->dims(), 0, true);
    mat_dim_c.height_ *= mat_dim_c.batch_size_;
    mat_dim_c.batch_size_ = 0;
    auto mat_dim_d = math::CreateMatrixDescriptor(pre_hidden->dims(), 0, false);
    mat_dim_d.height_ *= mat_dim_d.batch_size_;
    mat_dim_d.batch_size_ = 0;
    blas.MatMul(*grad_gate, mat_dim_c, *pre_hidden, mat_dim_d,
                static_cast<T>(1.0), grad_weight_hh, static_cast<T>(1.0));
  }
};

template <typename T>
struct LSTMGradCell : GradCell<T> {
  void operator()(const framework::ExecutionContext& context,
                  Tensor* gate_tensor, Tensor* state_tensor,
                  Tensor* act_state_tensor, Tensor* hidden_tensor,
                  const Tensor* weight_hh, Tensor* pre_hidden,
                  Tensor* pre_state, Tensor* grad_hidden, Tensor* grad_state,
                  Tensor* grad_gate, Tensor* grad_weight_hh,
                  Tensor* grad_pre_hidden, Tensor* grad_pre_state,
                  math::LstmMetaValue<T>* lstm_value,
                  math::LstmMetaGrad<T>* lstm_grad, const Tensor& mask_tensor,
                  bool has_sequence_length) override {
    auto& device_ctx =
        context.template device_context<platform::CPUDeviceContext>();
    auto blas = math::GetBlas<platform::CPUDeviceContext, T>(device_ctx);
    size_t frame_size = state_tensor->dims()[2];
    size_t batch_size = state_tensor->dims()[1];

    Tensor grad_pre_hidden_bak;
    Tensor grad_pre_state_bak;
    if (has_sequence_length) {
      grad_pre_hidden_bak.Resize(grad_pre_hidden->dims());
      grad_pre_hidden_bak.mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*grad_pre_hidden, device_ctx.GetPlace(), device_ctx,
                            &grad_pre_hidden_bak);
      grad_pre_state_bak.Resize(grad_pre_state->dims());
      grad_pre_state_bak.mutable_data<T>(context.GetPlace());
      framework::TensorCopy(*grad_pre_state, device_ctx.GetPlace(), device_ctx,
                            &grad_pre_state_bak);
    }
    Print3DTensor<T>(state_tensor, "state tensor");
    Print3DTensor<T>(act_state_tensor, "act_state_tensor");
    Print3DTensor<T>(gate_tensor, "gate_tensor");
    Print3DTensor<T>(grad_hidden, "grad_hidden");
    Print3DTensor<T>(grad_state, "grad_state");
    Print3DTensor<T>(grad_gate, "grad_gate");
    Print3DTensor<T>(grad_pre_state, "grad_pre_state");

    lstm_value->gate_value = gate_tensor->data<T>();
    lstm_value->state_value = state_tensor->data<T>();
    lstm_value->state_active_value = act_state_tensor->data<T>();
    lstm_value->prev_state_value = pre_state->data<T>();

    lstm_grad->state_grad = grad_state->data<T>();
    lstm_grad->gate_grad = grad_gate->data<T>();
    lstm_grad->output_grad = grad_hidden->data<T>();
    lstm_grad->prev_state_grad = grad_pre_state->data<T>();

    lstm_value->output_value = nullptr;
    lstm_grad->state_active_grad = nullptr;

    auto gate_act = math::detail::GetActivationType("sigmoid_v2");
    auto state_act = math::detail::GetActivationType("tanh_v2");
    auto cand_act = math::detail::GetActivationType("tanh_v2");

    T cell_clip = 0.0;
    math::LstmUnitGradFunctor<platform::CPUDeviceContext, T>::compute(
        device_ctx, *lstm_value, *lstm_grad, frame_size, batch_size, cell_clip,
        gate_act, state_act, cand_act, false);

    auto mat_dim_a = math::CreateMatrixDescriptor(grad_gate->dims(), 0, false);
    mat_dim_a.height_ *= mat_dim_a.batch_size_;
    mat_dim_a.batch_size_ = 0;
    auto mat_dim_b = math::CreateMatrixDescriptor(weight_hh->dims(), 0, false);
    VLOG(0) << "grad gate shape:" << grad_gate->dims();
    VLOG(0) << "weight_hh shape:" << weight_hh->dims();
    VLOG(0) << "pre_hidden shape:" << pre_hidden->dims();
    blas.MatMul(*grad_gate, mat_dim_a, *weight_hh, mat_dim_b,
                static_cast<T>(1.0), grad_pre_hidden, static_cast<T>(0.0));

    Print3DTensor<T>(grad_pre_hidden, "cell grad_pre_h before");
    Print3DTensor<T>(grad_pre_state, "cell grad_pre_state before");
    if (has_sequence_length) {
      auto& place =
          *context.template device_context<platform::CPUDeviceContext>()
               .eigen_device();
      auto eigen_mask = framework::EigenMatrix<T>::From(
          mask_tensor, framework::make_ddim({mask_tensor.dims()[1], 1}));
      auto eigen_mask_broadcast = eigen_mask.broadcast(
          Eigen::DSizes<int, 2>(1, grad_hidden->dims()[2]));
      auto eigen_grad_pre_hidden = framework::EigenMatrix<T>::Reshape(
          *grad_pre_hidden, grad_pre_hidden->dims().size() - 1);
      auto eigen_grad_pre_hidden_bak = framework::EigenMatrix<T>::Reshape(
          grad_pre_hidden_bak, grad_pre_hidden_bak.dims().size() - 1);
      auto eigen_grad_pre_state = framework::EigenMatrix<T>::Reshape(
          *grad_pre_state, grad_pre_state->dims().size() - 1);
      auto eigen_grad_pre_state_bak = framework::EigenMatrix<T>::Reshape(
          grad_pre_state_bak, grad_pre_state_bak.dims().size() - 1);
      eigen_grad_pre_hidden.device(place) =
          (1 - eigen_mask_broadcast) * eigen_grad_pre_hidden_bak +
          eigen_grad_pre_hidden * eigen_mask_broadcast;
      eigen_grad_pre_state.device(place) =
          (1 - eigen_mask_broadcast) * eigen_grad_pre_state_bak +
          eigen_grad_pre_state * eigen_mask_broadcast;
    }
    Print3DTensor<T>(grad_pre_hidden, "cell grad_pre_h");
    Print3DTensor<T>(grad_pre_state, "cell grad_pre_state");
    Print3DTensor<T>(grad_hidden, "cell grad_hidden");

    VLOG(0) << "first blas";
    auto mat_dim_c = math::CreateMatrixDescriptor(grad_gate->dims(), 0, true);
    mat_dim_c.height_ *= mat_dim_c.batch_size_;
    mat_dim_c.batch_size_ = 0;
    auto mat_dim_d = math::CreateMatrixDescriptor(pre_hidden->dims(), 0, false);
    mat_dim_d.height_ *= mat_dim_d.batch_size_;
    mat_dim_d.batch_size_ = 0;
    blas.MatMul(*grad_gate, mat_dim_c, *pre_hidden, mat_dim_d,
                static_cast<T>(1.0), grad_weight_hh, T(1.0));
  }
};

template <typename GradCellType,
          template <typename, typename> class SingleGradLayerT,
          template <typename, typename> class BidirGradLayerT, typename T>
void RnnGradFunc(const framework::ExecutionContext& context,
                 const int& gate_num) {
  // get the tensor pointer for the input
  auto* input = context.Input<Tensor>("Input");
  auto weight_list = context.MultiInput<Tensor>("WeightList");
  auto* init_h = context.Input<Tensor>("InitH");
  auto* init_c = context.Input<Tensor>("InitC");
  auto* reserve_state = context.Input<Tensor>("Reserve");
  auto* state_out = context.Input<Tensor>("StateOut");
  auto* output = context.Input<Tensor>("Out");
  auto* output_grad = context.Input<Tensor>(framework::GradVarName("Out"));
  auto* last_h_grad = context.Input<Tensor>(framework::GradVarName("LastH"));
  auto* last_c_grad = context.Input<Tensor>(framework::GradVarName("LastC"));

  Print3DTensor<T>(last_h_grad, "last_h_grad");

  bool has_seq_length = context.HasInput("SequenceLength");
  const Tensor* sequence_length = nullptr;
  if (has_seq_length) {
    sequence_length = context.Input<Tensor>("SequenceLength");
  }

  // get the tensor pointer for the output
  auto* input_grad = context.Output<Tensor>(framework::GradVarName("Input"));
  auto weight_grad_list = context.MultiOutput<framework::Tensor>(
      framework::GradVarName("WeightList"));
  auto* init_h_grad = context.Output<Tensor>(framework::GradVarName("InitH"));
  auto* init_c_grad = context.Output<Tensor>(framework::GradVarName("InitC"));

  // get the attributes for the calcluate
  const int& num_layers = context.Attr<int>("num_layers");
  const bool& is_bidirec = context.Attr<bool>("is_bidirec");
  const float& dropout_prob = context.Attr<float>("dropout_prob");
  const bool& is_test = context.Attr<bool>("is_test");

  // get the input_size, batch_size, time_step, hidden_size
  const int& time_step = input->dims()[0];
  const int& batch_size = input->dims()[1];
  const int& hidden_size = context.Attr<int>("hidden_size");
  const int& direction_num = is_bidirec ? 2 : 1;

  // allocate the memory and initization the input_grad
  input_grad->mutable_data<T>(input->dims(), context.GetPlace());

  if (init_h_grad) {
    init_h_grad->mutable_data<T>(init_h->dims(), context.GetPlace());
  }
  if (init_c_grad) {
    init_c_grad->mutable_data<T>(init_c->dims(), context.GetPlace());
  }

  // reset the parameter to sorted order and allocate the memory
  std::vector<TensorList> parameter_lists;
  parameter_lists.reserve(num_layers);
  reset_parameter_vector(weight_list, num_layers, gate_num, is_bidirec,
                         &parameter_lists);

  for (unsigned int i = 0; i < weight_grad_list.size(); ++i) {
    weight_grad_list[i]->mutable_data<T>(context.GetPlace());
  }
  std::vector<TensorList> parameter_lists_grad;
  parameter_lists_grad.reserve(num_layers);
  reset_parameter_vector(weight_grad_list, num_layers, gate_num, is_bidirec,
                         &parameter_lists_grad);

  // resolve the state of reverse_state
  Tensor gate_tensor;
  Tensor state_tensor;
  Tensor act_state_tensor;
  Tensor hidden_tensor;
  SplitReserveData(reserve_state, &gate_tensor, &state_tensor,
                   &act_state_tensor, &hidden_tensor, direction_num, time_step,
                   batch_size, hidden_size, gate_num, num_layers);
  int gate_num_tmp = gate_num;
  if (gate_num == 0) {
    gate_num_tmp = 1;
  }
  gate_tensor.Resize({num_layers, time_step * direction_num, batch_size,
                      hidden_size * gate_num_tmp});
  if (state_tensor.numel() > 0) {
    state_tensor.Resize(
        {num_layers, time_step * direction_num, batch_size, hidden_size});
    act_state_tensor.Resize(
        {num_layers, time_step * direction_num, batch_size, hidden_size});
  }
  if (num_layers > 1) {
    hidden_tensor.Resize(
        {num_layers - 1, time_step, batch_size, hidden_size * direction_num});
  }
  // unbind
  auto last_h_grad_unbind = Unbind(*last_h_grad);
  auto gate_tensor_unbind = Unbind(gate_tensor);
  TensorList last_c_grad_unbind;
  if (last_c_grad) {
    last_c_grad_unbind = Unbind(*last_c_grad);
    VLOG(0) << "last_c_grad_unbind is not null";
  }

  TensorList init_h_unbind, init_c_unbind;
  TensorList init_h_grad_unbind, init_c_grad_unbind;
  TensorList state_tensor_unbind, act_state_tensor_unbind;
  TensorList hidden_tensor_unbind;

  init_h_unbind = Unbind(*init_h);
  if (init_c) {
    init_c_unbind = Unbind(*init_c);
  }

  if (init_h_grad != nullptr) {
    init_h_grad_unbind = Unbind(*init_h_grad);
  }
  if (init_c_grad != nullptr) {
    init_c_grad_unbind = Unbind(*init_c_grad);
  }
  if (state_tensor.numel() > 0) {
    state_tensor_unbind = Unbind(state_tensor);
    act_state_tensor_unbind = Unbind(act_state_tensor);
  }
  if (num_layers > 1) {
    hidden_tensor_unbind = Unbind(hidden_tensor);
  }
  // squeeze the hidden first dim
  for (unsigned int i = 0; i < hidden_tensor_unbind.size(); i++) {
    hidden_tensor_unbind[i].Resize(
        framework::slice_ddim(hidden_tensor_unbind[i].dims(), 1,
                              hidden_tensor_unbind[i].dims().size()));
  }
  // add the output tensor to the hidden vector
  Tensor tmp;
  hidden_tensor_unbind.emplace_back(tmp);
  hidden_tensor_unbind[num_layers - 1].ShareDataWith(*output);

  GradCellType cell;
  Tensor layer_input;
  Tensor layer_output;
  Tensor* layer_input_grad_holder = nullptr;
  Tensor tmp_out;
  tmp_out.ShareDataWith(*output_grad);
  Tensor* layer_output_grad_holder = &tmp_out;
  Tensor input_grad_temp;
  Tensor output_grad_temp;

  bool has_allocate_mem = false;
  for (int i = num_layers - 1; i >= 0; --i) {
    // the layer input output had saved, just use the data
    VLOG(0) << "layer_idx:" << i << ", step 1";
    if (i > 0) {
      layer_input.ShareDataWith(hidden_tensor_unbind[i - 1]);
    } else {
      layer_input.ShareDataWith(*input);
    }
    layer_output.ShareDataWith(hidden_tensor_unbind[i]);
    VLOG(0) << "layer_idx:" << i << ", step 2";
    if (num_layers == 1) {
      layer_input_grad_holder = input_grad;
    } else {
      if (i == num_layers - 1) {
        input_grad_temp.Resize(layer_input.dims());
        input_grad_temp.mutable_data<T>(context.GetPlace());
        layer_input_grad_holder = &input_grad_temp;
      }
    }
    if (is_bidirec) {
      BidirGradLayerT<T, GradCellType> layer(cell);
      layer(context, &layer_input, &layer_output, &init_h_unbind,
            &init_c_unbind, last_h_grad_unbind, last_c_grad_unbind,
            gate_tensor_unbind, state_tensor_unbind, act_state_tensor_unbind,
            layer_output_grad_holder, parameter_lists, sequence_length,
            layer_input_grad_holder, &init_h_grad_unbind, &init_c_grad_unbind,
            &parameter_lists_grad, i, gate_num_tmp);
    } else {
      SingleGradLayerT<T, GradCellType> layer(cell);
      layer(context, &layer_input, &layer_output, &init_h_unbind,
            &init_c_unbind, last_h_grad_unbind, last_c_grad_unbind,
            gate_tensor_unbind, state_tensor_unbind, act_state_tensor_unbind,
            layer_output_grad_holder, parameter_lists, sequence_length,
            layer_input_grad_holder, &init_h_grad_unbind, &init_c_grad_unbind,
            &parameter_lists_grad, i, gate_num_tmp);
    }

    VLOG(0) << "layer_idx:" << i << ", step 3";
    // calcluate the dropout gradient for the layer_input_grad_holder
    // state_out save in the forward process
    if (i > 0) {
      if ((!is_test) && (dropout_prob != 0)) {
        dropout_cpu_grad_function_inplace<T>(context, layer_input_grad_holder,
                                             state_out, dropout_prob);
      }
    }

    if (i - 1 == 0) {
      layer_output_grad_holder = input_grad;
    } else {
      if (!has_allocate_mem) {
        output_grad_temp.Resize(layer_input_grad_holder->dims());
        output_grad_temp.mutable_data<T>(context.GetPlace());
        layer_output_grad_holder = &output_grad_temp;
        has_allocate_mem = true;
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
    if (cell_type == "lstm") {
      RnnGradFunc<LSTMGradCell<T>, SingleGradLayer, BidirGradLayer, T>(
          ctx, gate_num);
    } else if (cell_type == "gru") {
      gate_num = 3;
      // run gru
    } else if (cell_type == "rnn_relu") {
      gate_num = 0;
      RnnGradFunc<SimpleRNNGradCell<T, ReluGradFunctor>, SingleGradLayer,
                  BidirGradLayer, T>(ctx, gate_num);
      // run rnn
    } else if (cell_type == "rnn_tanh") {
      gate_num = 0;
      RnnGradFunc<SimpleRNNGradCell<T, TanhGradFunctor>, SingleGradLayer,
                  BidirGradLayer, T>(ctx, gate_num);
    }
  }
};
}  // namespace operators
}  // namespace paddle
