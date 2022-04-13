/* Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef PADDLE_WITH_XPU

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/utils.h"
#include "paddle/fluid/platform/device/device_wrapper.h"
#include "paddle/fluid/platform/device/xpu/xpu_header.h"
#include "paddle/fluid/platform/device_context.h"
#include "paddle/phi/kernels/funcs/math_function.h"

namespace paddle {
namespace operators {

using Tensor = framework::Tensor;
using DDim = framework::DDim;
using TensorList = std::vector<framework::Tensor>;
template <typename TensorType, typename T>
void reset_parameter_vector(const std::vector<TensorType>& raw_params_vec,
                            const int& num_layers, const bool& is_bidirec,
                            std::vector<std::vector<T*>>* params_vec) {
  // the parameter raw seuquence is [FWhi, FWhh, BWhi, BWhh] * num_layers
  // + [FBhi, FBhh, BBhi, BBhh] * num_layers, we will reset the parameter to
  // ([FWhi, FWhh, FBhi, FBhh] + [BWhi, BWhh, BBhi, BBhh]) * num_layers
  const int& direction_num = is_bidirec ? 2 : 1;
  const int& layer_weight_size = 4 * direction_num;
  const int& all_weight_size = num_layers * layer_weight_size;
  const int& bias_start_idx = all_weight_size / 2;
  for (int i = 0; i < num_layers; i++) {
    params_vec->at(i).resize(layer_weight_size);
    for (int j = 0; j < layer_weight_size; j++) {
      int k = j % 4;
      const int& section = j / 4;
      int tensor_idx = i * 2 * direction_num + section * 2 + k % 2;
      if (k >= 2) {
        tensor_idx += bias_start_idx;
      }
      using remove_cv_t = typename std::remove_cv<T>::type;
      params_vec->at(i)[j] =
          raw_params_vec[tensor_idx]->template data<remove_cv_t>();
    }
  }
}

template <typename DeviceContext, typename T>
void RunLSTMLayer(const framework::ExecutionContext& ctx, int seq_len,
                  int batch_size, int xdim, int hidden_size, const T* x, T* y,
                  const T* init_h, const T* init_c, T* last_h, T* last_c,
                  int state_offset, const std::vector<int>& seq_len_tensor,
                  const std::vector<const T*>& param_list, T* i_f_g_o, T* c,
                  bool is_bidirect, int layer_idx, int offset) {
  bool is_reverse = false;
  if (is_bidirect) {
    layer_idx = 2 * layer_idx + offset;
    if (offset > 0) {
      is_reverse = true;
    }
  }
  auto w_x = param_list[0 + offset * 4];
  auto w_h = param_list[1 + offset * 4];
  auto b_x = param_list[2 + offset * 4];
  auto b_h = param_list[3 + offset * 4];

  auto h_0 = init_h + layer_idx * state_offset;
  auto c_0 = init_c + layer_idx * state_offset;
  auto last_h_ptr = last_h + layer_idx * state_offset;
  auto last_c_ptr = last_c + layer_idx * state_offset;
  auto& dev_ctx = ctx.template device_context<DeviceContext>();
  int r = xpu::lstm_train<T, T, int16_t>(
      dev_ctx.x_context(), (const T*)x, (const T*)h_0, (const T*)c_0,
      (const T*)w_x, (const T*)w_h, (const T*)b_x, (const T*)b_h,
      reinterpret_cast<T*>(y), reinterpret_cast<T*>(last_h_ptr),
      reinterpret_cast<T*>(last_c_ptr), batch_size, xdim, hidden_size, seq_len,
      seq_len_tensor, is_reverse, nullptr, nullptr, nullptr, nullptr,
      reinterpret_cast<T*>(i_f_g_o), reinterpret_cast<T*>(c),
      xpu::Activation_t::TANH, xpu::Activation_t::SIGMOID);
  PADDLE_ENFORCE_XDNN_SUCCESS(r, "lstm_train");
}

template <typename DeviceContext, typename T>
class RnnXPUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Input
    auto* input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    bool has_seq_length = ctx.HasInput("SequenceLength");
    // Output
    auto state = ctx.MultiOutput<Tensor>("State");
    auto* output = ctx.Output<Tensor>("Out");
    auto* dropout_mask = ctx.Output<Tensor>("DropoutState");
    auto* reserve_data = ctx.Output<Tensor>("Reserve");
    // Attrbutes
    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const std::string& mode = ctx.Attr<std::string>("mode");

    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }

    if (dropout_mask->IsInitialized()) {
      if (dropout_mask->numel() != output->numel()) dropout_mask->clear();
    }
    dropout_mask->mutable_data<uint8_t>(output->dims(), ctx.GetPlace());
    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    phi::funcs::SetConstant<platform::XPUDeviceContext, uint8_t> ones;
    ones(dev_ctx, dropout_mask, static_cast<uint8_t>(1));

    PADDLE_ENFORCE_EQ(
        mode, "LSTM",
        platform::errors::InvalidArgument(
            "XPU only support LSTM mode now, current mode is %s", mode));

    auto init_h = pre_state[0];
    auto init_c = pre_state[1];
    auto last_h = state[0];
    auto last_c = state[1];

    // check shape
    const int& seq_len = input->dims()[0];  // time_step
    const int& batch_size = input->dims()[1];
    const int& input_dim = input->dims()[2];
    const int& direction_num = is_bidirec ? 2 : 1;

    PADDLE_ENFORCE_EQ(
        init_h->dims()[0], num_layers * direction_num,
        platform::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init "
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers, init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0], num_layers * direction_num,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers, init_c->dims()[0]));
    // weightlist
    std::vector<std::vector<const T*>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec,
                           &parameter_lists);

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());
    last_h->mutable_data<T>(ctx.GetPlace());
    last_c->mutable_data<T>(ctx.GetPlace());

    reserve_data->Resize(
        {num_layers * direction_num * seq_len * batch_size * hidden_size * 5});
    reserve_data->mutable_data<T>(ctx.GetPlace());
    Tensor internal_output_1_tensor, internal_output_2_tensor;
    T* internal_output_1_ptr = nullptr;
    T* internal_output_2_ptr = nullptr;
    if (num_layers >= 2) {
      internal_output_1_tensor.Resize(output->dims());
      internal_output_1_ptr =
          internal_output_1_tensor.mutable_data<T>(ctx.GetPlace());
    }
    if (num_layers >= 3) {
      internal_output_2_tensor.Resize(output->dims());
      internal_output_2_ptr =
          internal_output_2_tensor.mutable_data<T>(ctx.GetPlace());
    }
    // get ptr from tensor
    auto x = input->data<T>();
    auto init_h_ptr = init_h->data<T>();
    auto init_c_ptr = init_c->data<T>();
    auto y = output->data<T>();
    auto last_h_ptr = last_h->data<T>();
    auto last_c_ptr = last_c->data<T>();
    auto i_f_g_o = reserve_data->data<T>();
    auto c =
        i_f_g_o +
        num_layers * direction_num * seq_len * batch_size * hidden_size * 4;

    std::vector<int> seq_len_tensor(batch_size, seq_len);
    if (has_seq_length) {
      seq_len_tensor = operators::GetDataFromTensor(sequence_length);
    }

    int state_offset = pre_state[0]->dims()[1] * pre_state[0]->dims()[2];

    for (int i = 0; i < num_layers; i++) {
      const T* cur_input_ptr = nullptr;
      int cur_xdim = -1;
      i_f_g_o += i * direction_num * seq_len * batch_size * hidden_size * 4;
      c += i * direction_num * seq_len * batch_size * hidden_size;

      if (i == 0) {
        cur_input_ptr = x;
        cur_xdim = input_dim;
      } else if (i % 2 != 0) {
        cur_input_ptr = internal_output_1_ptr;
        cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
      } else {
        cur_input_ptr = internal_output_2_ptr;
        cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
      }

      T* cur_output_ptr = nullptr;
      if (i == num_layers - 1) {
        cur_output_ptr = y;
      } else if (i % 2 != 0) {
        cur_output_ptr = internal_output_2_ptr;
      } else {
        cur_output_ptr = internal_output_1_ptr;
      }

      if (is_bidirec) {
        std::vector<Tensor> output_vec(2);
        std::vector<T*> output_ptr_vec(2);
        for (int k = 0; k < 2; ++k) {
          output_vec[k].Resize({seq_len, batch_size, output->dims()[2] / 2});
          output_ptr_vec[k] = output_vec[k].mutable_data<T>(ctx.GetPlace());
        }
        RunLSTMLayer<DeviceContext, T>(
            ctx, seq_len, batch_size, cur_xdim, hidden_size, cur_input_ptr,
            output_ptr_vec[0], init_h_ptr, init_c_ptr, last_h_ptr, last_c_ptr,
            state_offset, seq_len_tensor, parameter_lists[i], i_f_g_o, c,
            is_bidirec, i, 0);

        T* bw_i_f_g_o = i_f_g_o + seq_len * batch_size * hidden_size * 4;
        T* bw_c = c + seq_len * batch_size * hidden_size;
        RunLSTMLayer<DeviceContext, T>(
            ctx, seq_len, batch_size, cur_xdim, hidden_size, cur_input_ptr,
            output_ptr_vec[1], init_h_ptr, init_c_ptr, last_h_ptr, last_c_ptr,
            state_offset, seq_len_tensor, parameter_lists[i], bw_i_f_g_o, bw_c,
            is_bidirec, i, 1);

        // concat
        int r = xpu::concat<T>(
            dev_ctx.x_context(), {output_ptr_vec[0], output_ptr_vec[1]},
            cur_output_ptr, {{seq_len, batch_size, hidden_size},
                             {seq_len, batch_size, hidden_size}},
            2);
        PADDLE_ENFORCE_XDNN_SUCCESS(r, "concat");
        xpu_wait(dev_ctx.x_context()->xpu_stream);
      } else {
        RunLSTMLayer<DeviceContext, T>(
            ctx, seq_len, batch_size, cur_xdim, hidden_size, cur_input_ptr,
            cur_output_ptr, init_h_ptr, init_c_ptr, last_h_ptr, last_c_ptr,
            state_offset, seq_len_tensor, parameter_lists[i], i_f_g_o, c,
            is_bidirec, i, 0);
      }
    }
  }
};

template <typename DeviceContext, typename T>
class RnnXPUGradKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // get the tensor pointer for the input
    auto* input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    auto* output = ctx.Input<Tensor>("Out");
    auto* reserve_data = ctx.Input<Tensor>("Reserve");
    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const std::string& mode = ctx.Attr<std::string>("mode");

    bool has_seq_length = ctx.HasInput("SequenceLength");
    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }

    PADDLE_ENFORCE_EQ(
        mode, "LSTM",
        platform::errors::InvalidArgument(
            "XPU only support LSTM mode now, current mode is %s", mode));

    PADDLE_ENFORCE_EQ(is_bidirec, false,
                      platform::errors::InvalidArgument(
                          "XPU only support unidirectional LSTM now"));

    PADDLE_ENFORCE_EQ(
        num_layers, 1,
        platform::errors::InvalidArgument(
            "XPU only support 1 layer LSTM now, current layer num is %s",
            num_layers));

    auto init_h = pre_state[0];
    auto init_c = pre_state[1];

    auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
    auto state_grad = ctx.MultiInput<Tensor>(framework::GradVarName("State"));
    auto last_h_grad = state_grad[0];
    auto last_c_grad = state_grad[1];

    // get the tensor pointer for the output
    auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
    auto weight_grad_list = ctx.MultiOutput<framework::Tensor>(
        framework::GradVarName("WeightList"));
    auto pre_state_grad =
        ctx.MultiOutput<Tensor>(framework::GradVarName("PreState"));
    Tensor* init_h_grad = nullptr;
    Tensor* init_c_grad = nullptr;
    if (pre_state_grad.size() > 0) {  // has gradient
      init_h_grad = pre_state_grad[0];
      init_c_grad = pre_state_grad[1];
    }

    // check shape
    int seq_len = input->dims()[0];
    int batch_size = input->dims()[1];
    int input_dim = input->dims()[2];
    PADDLE_ENFORCE_EQ(
        init_h->dims()[0], num_layers,
        platform::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init "
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers, init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0], num_layers,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers, init_c->dims()[0]));

    std::vector<std::vector<const T*>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec,
                           &parameter_lists);

    for (unsigned int i = 0; i < weight_grad_list.size(); ++i) {
      weight_grad_list[i]->mutable_data<T>(ctx.GetPlace());
    }
    std::vector<std::vector<T*>> parameter_lists_grad;
    parameter_lists_grad.resize(num_layers);
    reset_parameter_vector(weight_grad_list, num_layers, is_bidirec,
                           &parameter_lists_grad);

    // allocate the memory and initization the input_grad
    input_grad->mutable_data<T>(input->dims(), ctx.GetPlace());
    if (init_h_grad) {
      init_h_grad->mutable_data<T>(init_h->dims(), ctx.GetPlace());
    }
    if (init_c_grad) {
      init_c_grad->mutable_data<T>(init_c->dims(), ctx.GetPlace());
    }

    // get ptr from tensor
    auto x = input->data<T>();
    auto h_0 = init_h->data<T>();
    auto c_0 = init_c->data<T>();
    auto w_x = parameter_lists[0][0];
    auto w_h = parameter_lists[0][1];
    auto y = output->data<T>();
    auto y_grad = output_grad->data<T>();
    auto last_h_grad_ptr = last_h_grad->data<T>();
    auto last_c_grad_ptr = last_c_grad->data<T>();
    auto x_grad = input_grad->data<T>();
    auto h_0_grad = init_h_grad ? init_h_grad->data<T>() : nullptr;
    auto c_0_grad = init_c_grad ? init_c_grad->data<T>() : nullptr;
    auto w_x_grad = parameter_lists_grad[0][0];
    auto w_h_grad = parameter_lists_grad[0][1];
    auto b_x_grad = parameter_lists_grad[0][2];
    auto b_h_grad = parameter_lists_grad[0][3];
    auto i_f_g_o = reserve_data->data<T>();
    auto c = i_f_g_o + seq_len * batch_size * hidden_size * 4;

    std::vector<int> seq_len_tensor(batch_size, seq_len);
    if (has_seq_length) {
      seq_len_tensor = operators::GetDataFromTensor(sequence_length);
    }

    auto& dev_ctx = ctx.template device_context<DeviceContext>();
    int r = xpu::lstm_grad<T, T, int16_t>(
        dev_ctx.x_context(), (const T*)x, (const T*)h_0, (const T*)c_0,
        (const T*)w_x, (const T*)w_h, (const T*)y, (const T*)y_grad,
        (const T*)last_h_grad_ptr, (const T*)last_c_grad_ptr,
        reinterpret_cast<T*>(x_grad), reinterpret_cast<T*>(h_0_grad),
        reinterpret_cast<T*>(c_0_grad), w_x_grad, w_h_grad, b_x_grad, b_h_grad,
        batch_size, input_dim, hidden_size, seq_len, seq_len_tensor, nullptr,
        nullptr, nullptr, nullptr, i_f_g_o, c);
    PADDLE_ENFORCE_EQ(
        r, xpu::Error_t::SUCCESS,
        platform::errors::External("RnnXPUGrad(lstm) return wrong "
                                   "value[%d %s]",
                                   r, XPUAPIErrorMsg[r]));
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_XPU_KERNEL(
    rnn, ops::RnnXPUKernel<paddle::platform::XPUDeviceContext, float>);
REGISTER_OP_XPU_KERNEL(
    rnn_grad, ops::RnnXPUGradKernel<paddle::platform::XPUDeviceContext, float>);

#endif  // PADDLE_WITH_XPU
