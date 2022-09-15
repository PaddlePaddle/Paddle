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

#include "paddle/phi/kernels/rnn_kernel.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/phi/backends/xpu/enforce_xpu.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/kernels/funcs/math_function.h"
#include "paddle/phi/kernels/xpu/rnn_util.h"

namespace phi {

template <typename T, typename Context>
void RnnKernel(const Context& dev_ctx,
               const DenseTensor& x,
               const std::vector<const DenseTensor*>& pre_state,
               const std::vector<const DenseTensor*>& weight_list,
               const paddle::optional<DenseTensor>& sequence_length,
               float dropout_prob,
               bool is_bidirec,
               int input_size,
               int hidden_size,
               int num_layers,
               const std::string& mode,
               int seed,
               bool is_test,
               DenseTensor* out,
               DenseTensor* dropout_state,
               std::vector<DenseTensor*> state,
               DenseTensor* reserve) {
  using XPUTyp = typename XPUTypeTrait<T>::Type;
  if (dropout_state->IsInitialized()) {
    if (dropout_state->numel() != out->numel()) dropout_state->clear();
  }

  dropout_state->Resize(out->dims());
  dev_ctx.template Alloc<T>(dropout_state);

  phi::funcs::SetConstant<phi::XPUContext, uint8_t> ones;
  ones(dev_ctx, dropout_state, static_cast<uint8_t>(1));

  PADDLE_ENFORCE_EQ(
      mode,
      "LSTM",
      errors::InvalidArgument(
          "XPU only support LSTM mode now, current mode is %s", mode));

  auto init_h = pre_state[0];
  auto init_c = pre_state[1];
  auto last_h = state[0];
  auto last_c = state[1];

  // check shape
  const int& seq_len = x.dims()[0];  // time_step
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
  // weightlist
  std::vector<std::vector<const T*>> parameter_lists;
  parameter_lists.resize(num_layers);
  ResetParameterVector(weight_list, num_layers, is_bidirec, &parameter_lists);

  // init the output and allocate the memory
  dev_ctx.template Alloc<T>(out);
  dev_ctx.template Alloc<T>(last_h);
  dev_ctx.template Alloc<T>(last_c);

  int gate_num = 4;
  int hidden_data_idx = (num_layers - 1);
  hidden_data_idx += (gate_num + 1) * num_layers;
  const int& block_size = direction_num * seq_len * batch_size * hidden_size;
  reserve->Resize({hidden_data_idx, block_size});
  dev_ctx.template Alloc<T>(reserve);

  // get ptr from tensor
  auto x_data = x.data<T>();
  auto init_h_ptr = init_h->data<T>();
  auto init_c_ptr = init_c->data<T>();
  auto y = out->data<T>();
  auto last_h_ptr = last_h->data<T>();
  auto last_c_ptr = last_c->data<T>();
  auto i_f_g_o_ptr = reserve->data<T>();
  auto c_ptr =
      i_f_g_o_ptr + num_layers * block_size * 4;  // 4 for i_f_g_o offset
  auto hidden_data_ptr = c_ptr + num_layers * block_size * 1;  // 1 for c offset

  std::vector<int> seq_len_tensor(batch_size, seq_len);

  bool has_seq_length = sequence_length.is_initialized();

  if (has_seq_length) {
    seq_len_tensor =
        paddle::operators::GetDataFromTensor<int>(sequence_length.get_ptr());
  }

  int state_offset = pre_state[0]->dims()[1] * pre_state[0]->dims()[2];

  const T* cur_input_ptr = nullptr;
  int cur_xdim = -1;
  T* cur_output_ptr = y;
  for (int i = 0; i < num_layers; i++) {
    auto i_f_g_o = i_f_g_o_ptr + i * block_size * 4;
    auto c = c_ptr + i * block_size;

    cur_output_ptr = y;
    if (i < num_layers - 1 && num_layers > 1) {
      cur_output_ptr = hidden_data_ptr + i * block_size;
    }

    if (i == 0) {
      cur_input_ptr = x_data;
      cur_xdim = input_dim;
    } else {
      cur_input_ptr = hidden_data_ptr + (i - 1) * block_size;
      cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
    }

    auto h_0 = init_h_ptr + direction_num * i * state_offset;
    auto c_0 = init_c_ptr + direction_num * i * state_offset;
    auto last_h = last_h_ptr + direction_num * i * state_offset;
    auto last_c = last_c_ptr + direction_num * i * state_offset;

    auto w_x = parameter_lists[i][0];
    auto w_h = parameter_lists[i][1];
    auto b_x = parameter_lists[i][2];
    auto b_h = parameter_lists[i][3];
    if (is_bidirec) {
      auto bw_x = parameter_lists[i][4];
      auto bw_h = parameter_lists[i][5];
      auto bb_x = parameter_lists[i][6];
      auto bb_h = parameter_lists[i][7];

      int r =
          xpu::bilstm_train<T, T, int16_t>(dev_ctx.x_context(),
                                           (const T*)cur_input_ptr,
                                           (const T*)h_0,
                                           (const T*)c_0,
                                           (const T*)w_x,
                                           (const T*)w_h,
                                           (const T*)b_x,
                                           (const T*)b_h,
                                           (const T*)bw_x,
                                           (const T*)bw_h,
                                           (const T*)bb_x,
                                           (const T*)bb_h,
                                           reinterpret_cast<T*>(cur_output_ptr),
                                           reinterpret_cast<T*>(last_h),
                                           reinterpret_cast<T*>(last_c),
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
                                           reinterpret_cast<T*>(i_f_g_o),
                                           reinterpret_cast<T*>(c));

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "bilstm_train");
    } else {
      int r =
          xpu::lstm_train<T, T, int16_t>(dev_ctx.x_context(),
                                         (const T*)cur_input_ptr,
                                         (const T*)h_0,
                                         (const T*)c_0,
                                         (const T*)w_x,
                                         (const T*)w_h,
                                         (const T*)b_x,
                                         (const T*)b_h,
                                         reinterpret_cast<T*>(cur_output_ptr),
                                         reinterpret_cast<T*>(last_h),
                                         reinterpret_cast<T*>(last_c),
                                         batch_size,
                                         cur_xdim,
                                         hidden_size,
                                         seq_len,
                                         seq_len_tensor,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         nullptr,
                                         reinterpret_cast<T*>(i_f_g_o),
                                         reinterpret_cast<T*>(c),
                                         xpu::Activation_t::TANH,
                                         xpu::Activation_t::SIGMOID);

      PADDLE_ENFORCE_XDNN_SUCCESS(r, "lstm_train");
    }
  }
}

}  // namespace phi

PD_REGISTER_KERNEL(rnn, XPU, ALL_LAYOUT, phi::RnnKernel, float) {}
