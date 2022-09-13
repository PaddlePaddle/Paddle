// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/phi/kernels/rnn_grad_kernel.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/xpu/rnn_util.h"

namespace phi {

template <typename T, typename Context>
void RnnGradKernel(const Context& dev_ctx,
                   const DenseTensor& x,
                   const std::vector<const DenseTensor*>& pre_state,
                   const std::vector<const DenseTensor*>& weight_list,
                   const paddle::optional<DenseTensor>& sequence_length,
                   const DenseTensor& out,
                   const DenseTensor& dropout_state,
                   const DenseTensor& reserve,
                   const DenseTensor& out_grad,
                   const std::vector<const DenseTensor*>& state_grad,
                   float dropout_prob,
                   bool is_bidirec,
                   int input_size,
                   int hidden_size,
                   int num_layers,
                   const std::string& mode,
                   int seed,
                   bool is_test,
                   DenseTensor* x_grad,
                   std::vector<DenseTensor*> pre_state_grad,
                   std::vector<DenseTensor*> weight_grad_list) {
  using XPUTyp = typename XPUTypeTrait<T>::Type;

  PADDLE_ENFORCE_EQ(
      mode,
      "LSTM",
      errors::InvalidArgument(
          "XPU only support LSTM mode now, current mode is %s", mode));

  auto init_h = pre_state[0];
  auto init_c = pre_state[1];

  auto last_h_grad = state_grad[0];
  auto last_c_grad = state_grad[1];

  // get the tensor pointer for the output
  DenseTensor* init_h_grad = nullptr;
  DenseTensor* init_c_grad = nullptr;
  if (pre_state_grad.size() > 0) {  // has gradient
    init_h_grad = pre_state_grad[0];
    init_c_grad = pre_state_grad[1];
  }

  // check shape
  const int& seq_len = x.dims()[0];
  const int& batch_size = x.dims()[1];
  const int& input_dim = x.dims()[2];
  const int& direction_num = is_bidirec ? 2 : 1;
  PADDLE_ENFORCE_EQ(
      init_h->dims()[0],
      num_layers * direction_num,
      errors::InvalidArgument("The num_layers of in RNN layer must"
                              " be the same as first dim of init "
                              "hidden, but received num_layers:%d,"
                              " dim:%d",
                              num_layers,
                              init_h->dims()[0]));

  PADDLE_ENFORCE_EQ(
      init_c->dims()[0],
      num_layers * direction_num,
      errors::InvalidArgument(
          "The num_layers of in RNN layer must"
          " be the same as first dim of cell state hidden, but received"
          " num_layers:%d, dim:%d",
          num_layers,
          init_c->dims()[0]));

  std::vector<std::vector<const T*>> parameter_lists;
  parameter_lists.resize(num_layers);
  ResetParameterVector(weight_list, num_layers, is_bidirec, &parameter_lists);

  for (unsigned int i = 0; i < weight_grad_list.size(); ++i) {
    dev_ctx.template Alloc<T>(weight_grad_list[i]);
  }
  std::vector<std::vector<T*>> parameter_lists_grad;
  parameter_lists_grad.resize(num_layers);
  ResetParameterVector(
      weight_grad_list, num_layers, is_bidirec, &parameter_lists_grad);

  // allocate the memory and initization the x_grad
  x_grad->Resize(x.dims());
  dev_ctx.template Alloc<T>(x_grad);

  phi::funcs::SetConstant<phi::XPUContext, T> zero;
  zero(dev_ctx, x_grad, static_cast<T>(0.0));

  DenseTensor a, b;
  DenseTensor* dynamic_grad_pre_h = &a;
  DenseTensor* dynamic_grad_pre_c = &b;
  if (init_h_grad) {
    init_h_grad->Resize(last_h_grad->dims());
    dev_ctx.template Alloc<T>(init_h_grad);

    zero(dev_ctx, init_h_grad, static_cast<T>(0.0));
  } else {
    dynamic_grad_pre_h->Resize(last_h_grad->dims());
    dev_ctx.template Alloc<T>(dynamic_grad_pre_h);

    zero(dev_ctx, dynamic_grad_pre_h, static_cast<T>(0.0));
    init_h_grad = dynamic_grad_pre_h;
  }
  if (init_c_grad) {
    init_c_grad->Resize(last_c_grad->dims());
    dev_ctx.template Alloc<T>(init_c_grad);
  } else {
    dynamic_grad_pre_c->Resize(last_h_grad->dims());
    dev_ctx.template Alloc<T>(dynamic_grad_pre_c);
    init_c_grad = dynamic_grad_pre_c;
  }

  DenseTensor temp_input_grad_1, temp_input_grad_2;
  T* input_grad_1_ptr = nullptr;
  T* input_grad_2_ptr = nullptr;
  if (num_layers >= 2) {
    temp_input_grad_1.Resize(x_grad->dims());
    input_grad_1_ptr = dev_ctx.template Alloc<T>(&temp_input_grad_1);
  }
  if (num_layers >= 3) {
    temp_input_grad_2.Resize(x_grad->dims());
    input_grad_2_ptr = dev_ctx.template Alloc<T>(&temp_input_grad_2);
  }

  // get ptr from tensor
  auto x_data = x.data<T>();
  auto init_h_ptr = init_h->data<T>();
  auto init_c_ptr = init_c->data<T>();
  auto y = out.data<T>();
  auto y_grad = out_grad.data<T>();
  auto last_h_grad_ptr = last_h_grad->data<T>();
  auto last_c_grad_ptr = last_c_grad->data<T>();
  auto x_grad_data = x_grad->data<T>();
  auto init_h_grad_ptr = init_h_grad->data<T>();
  auto init_c_grad_ptr = init_c_grad->data<T>();
  const int& block_size = direction_num * seq_len * batch_size * hidden_size;
  auto i_f_g_o_ptr = reserve.data<T>();
  auto c_ptr = i_f_g_o_ptr + num_layers * block_size * 4;
  auto hidden_data_ptr = c_ptr + num_layers * block_size * 1;
  int state_offset = pre_state[0]->dims()[1] * pre_state[0]->dims()[2];

  bool has_seq_length = sequence_length.is_initialized();
  std::vector<int> seq_len_tensor(batch_size, seq_len);
  if (has_seq_length) {
    seq_len_tensor =
        paddle::operators::GetDataFromTensor<int>(sequence_length.get_ptr());
  }

  for (int i = num_layers - 1; i >= 0; --i) {
    // the layer input output had saved, just use the data
    auto w_x = parameter_lists[i][0];
    auto w_h = parameter_lists[i][1];
    auto bw_x = parameter_lists[i][4];
    auto bw_h = parameter_lists[i][5];

    auto i_f_g_o = i_f_g_o_ptr + i * block_size * 4;
    auto c = c_ptr + i * block_size;

    DenseTensor layer_input_t;
    auto layer_input = x_data;
    if (i > 0) {
      layer_input_t.Resize(out.dims());
      layer_input = dev_ctx.template Alloc<T>(&layer_input_t);
      float scale = static_cast<float>(1.0f - dropout_prob);
      auto hidden_data = hidden_data_ptr + (i - 1) * block_size;
      int r = xpu::scale(dev_ctx.x_context(),
                         reinterpret_cast<const XPUTyp*>(hidden_data),
                         const_cast<XPUTyp*>(layer_input),
                         out.numel(),
                         false,
                         scale,
                         0.0f);
      PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
    } else {
      layer_input = x_data;
    }

    auto layer_output = y;
    if (i == num_layers - 1) {
      layer_output = y;
    } else {
      layer_output = hidden_data_ptr + i * block_size;
    }

    const T* cur_input_ptr = nullptr;
    if (i == num_layers - 1) {
      cur_input_ptr = y_grad;
    } else if (i % 2 != 0) {
      cur_input_ptr = input_grad_2_ptr;
    } else {
      cur_input_ptr = input_grad_1_ptr;
    }

    T* cur_output_ptr = nullptr;
    int cur_xdim = -1;
    if (i == 0) {
      cur_output_ptr = x_grad_data;
      cur_xdim = input_dim;
    } else if (i % 2 != 0) {
      cur_output_ptr = input_grad_1_ptr;
      cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
    } else {
      cur_output_ptr = input_grad_2_ptr;
      cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
    }

    auto w_x_grad = parameter_lists_grad[i][0];
    auto w_h_grad = parameter_lists_grad[i][1];
    auto b_x_grad = parameter_lists_grad[i][2];
    auto b_h_grad = parameter_lists_grad[i][3];

    auto h_0 = init_h_ptr + direction_num * i * state_offset;
    auto c_0 = init_c_ptr + direction_num * i * state_offset;

    auto h_0_grad = init_h_grad_ptr + direction_num * i * state_offset;
    auto c_0_grad = init_c_grad_ptr + direction_num * i * state_offset;
    auto h_t_grad = last_h_grad_ptr + direction_num * i * state_offset;
    auto c_t_grad = last_c_grad_ptr + direction_num * i * state_offset;

    if (is_bidirec) {
      auto bw_x_grad = parameter_lists_grad[i][4];
      auto bw_h_grad = parameter_lists_grad[i][5];
      auto bb_x_grad = parameter_lists_grad[i][6];
      auto bb_h_grad = parameter_lists_grad[i][7];

      int r =
          xpu::bilstm_grad<T, T, int16_t>(dev_ctx.x_context(),
                                          (const T*)layer_input,
                                          (const T*)h_0,
                                          (const T*)c_0,
                                          (const T*)w_x,
                                          (const T*)w_h,
                                          (const T*)bw_x,
                                          (const T*)bw_h,
                                          (const T*)layer_output,
                                          (const T*)cur_input_ptr,
                                          (const T*)h_t_grad,
                                          (const T*)c_t_grad,
                                          reinterpret_cast<T*>(cur_output_ptr),
                                          reinterpret_cast<T*>(h_0_grad),
                                          reinterpret_cast<T*>(c_0_grad),
                                          w_x_grad,
                                          w_h_grad,
                                          b_x_grad,
                                          b_h_grad,
                                          bw_x_grad,
                                          bw_h_grad,
                                          bb_x_grad,
                                          bb_h_grad,
                                          batch_size,
                                          cur_xdim,
                                          hidden_size,
                                          seq_len,
                                          seq_len_tensor,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          nullptr,
                                          i_f_g_o,
                                          c);

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "bilstm_grad");
    } else {
      int r =
          xpu::lstm_grad<T, T, int16_t>(dev_ctx.x_context(),
                                        (const T*)layer_input,
                                        (const T*)h_0,
                                        (const T*)c_0,
                                        (const T*)w_x,
                                        (const T*)w_h,
                                        (const T*)layer_output,
                                        (const T*)cur_input_ptr,
                                        (const T*)h_t_grad,
                                        (const T*)c_t_grad,
                                        reinterpret_cast<T*>(cur_output_ptr),
                                        reinterpret_cast<T*>(h_0_grad),
                                        reinterpret_cast<T*>(c_0_grad),
                                        w_x_grad,
                                        w_h_grad,
                                        b_x_grad,
                                        b_h_grad,
                                        batch_size,
                                        cur_xdim,
                                        hidden_size,
                                        seq_len,
                                        seq_len_tensor,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        nullptr,
                                        i_f_g_o,
                                        c);

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "lstm_grad");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rnn_grad, XPU, ALL_LAYOUT, phi::RnnGradKernel, float) {}
