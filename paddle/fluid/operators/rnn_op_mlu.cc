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

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/mlu/mlu_baseop.h"
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
void reset_parameter_vector(
    const std::vector<TensorType>& raw_params_vec, const int& num_layers,
    const bool& is_bidirec,
    std::vector<std::vector<std::pair<const T*, size_t>>>* params_vec) {
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
      params_vec->at(i)[j] = std::make_pair(
          raw_params_vec[tensor_idx]->template data<remove_cv_t>(),
          raw_params_vec[tensor_idx]->numel() * sizeof(T));
    }
  }
}

template <typename DeviceContext, typename T>
class RNNMLUKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    // Input
    auto& dev_ctx = GetDevCtxFromCTX(ctx);
    auto* input = ctx.Input<Tensor>("Input");
    auto pre_state = ctx.MultiInput<Tensor>("PreState");
    auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
    bool has_seq_length = ctx.HasInput("SequenceLength");
    // Output
    auto state = ctx.MultiOutput<Tensor>("State");
    auto* output = ctx.Output<Tensor>("Out");
    // auto* dropout_mask = ctx.Output<Tensor>("DropoutState");
    auto* reserve_data = ctx.Output<Tensor>("Reserve");
    // Attributes
    const int& num_layers = ctx.Attr<int>("num_layers");
    const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
    const int& hidden_size = ctx.Attr<int>("hidden_size");
    const std::string& mode = ctx.Attr<std::string>("mode");

    const Tensor* sequence_length = nullptr;
    if (has_seq_length) {
      sequence_length = ctx.Input<Tensor>("SequenceLength");
    }

    // if (dropout_mask->IsInitialized()) {
    //   if (dropout_mask->numel() != output->numel()) dropout_mask->clear();
    // }
    // dropout_mask->mutable_data<uint8_t>(output->dims(), ctx.GetPlace());
    // auto& dev_ctx = ctx.template device_context<DeviceContext>();
    // phi::funcs::SetConstant<platform::XPUDeviceContext, uint8_t> ones;
    // ones(dev_ctx, dropout_mask, static_cast<uint8_t>(1));

    auto init_h = pre_state[0];  // -> hx
    auto init_c = pre_state[1];  // -> cx
    auto last_h = state[0];
    auto last_c = state[1];

    // check shape
    const int in_out_dim_num = input->dims().size();
    const int& seq_len = input->dims()[0];  // time_step
    const int& batch_size = input->dims()[1];
    const int& input_dim = input->dims()[2];
    const int& direction_num = is_bidirec ? 2 : 1;
    int in_dim_arr[in_out_dim_num] = {seq_len, batch_size, input_dim};
    int out_dim_arr[in_out_dim_num] = {seq_len, batch_size,
                                       direction_num * hidden_size};
    int proj_size = hidden_size;

    std::vector<int> seq_len_vec(batch_size, seq_len);
    if (has_seq_length) {  // set seq_len if no padding, otherwise seq_len for
                           // each element.
      seq_len_vec = operators::GetDataFromTensor(sequence_length);
    }
    cnnlDirectionMode_t direction =
        is_bidirec ? CNNL_RNN_BIDIRECTIONAL : CNNL_RNN_UNIDIRECTIONAL;

    PADDLE_ENFORCE_EQ(
        mode, "LSTM",
        platform::errors::InvalidArgument(
            "MLU only support LSTM mode now, current mode is %s", mode));
    PADDLE_ENFORCE_EQ(
        num_layers, 1,
        platform::errors::InvalidArgument(
            "MLU only support 1 num_layers, current num_layers is %s",
            num_layers));
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
    std::vector<std::vector<std::pair<const T*, size_t>>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(weight_list, num_layers, is_bidirec,
                           &parameter_lists);

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());  // -> y in cnnl
    last_h->mutable_data<T>(ctx.GetPlace());  // -> hy in cnnl
    last_c->mutable_data<T>(ctx.GetPlace());  // -> cy in cnnl

    MLUSeqDataDesc input_seq_data_desc(
        CNNL_SEQDATA_TNC, ToCnnlDataType(input->dtype()), in_out_dim_num,
        in_dim_arr, static_cast<int>(seq_len_vec.size()), seq_len_vec.data(),
        nullptr);
    MLUSeqDataDesc out_seq_data_desc(
        CNNL_SEQDATA_TNC, ToCnnlDataType(input->dtype()), in_out_dim_num,
        out_dim_arr, static_cast<int>(seq_len_vec.size()), seq_len_vec.data(),
        nullptr);
    MLUCnnlTensorDesc hx_desc(*init_h);
    MLUCnnlTensorDesc cx_desc(*init_c);

    MLURNNDesc rnn_desc(CNNL_LSTM, CNNL_RNN_DOUBLE_BIAS, direction,
                        CNNL_RNN_LINEAR_INPUT, ToCnnlDataType(input->dtype()),
                        ToCnnlDataType(input->dtype()), input_dim, hidden_size,
                        /*projection*/ proj_size, num_layers, nullptr,
                        CNNL_RNN_PADDED_IO_DISABLED);
    rnn_desc.SetRNNMaskMode(CNNL_LSTM_MASK_ENABLED);

    // copy weight params
    size_t weightspace_size;
    framework::Tensor weightspace;
    PADDLE_ENFORCE_MLU_SUCCESS(cnnlGetRNNWeightSpaceSize(
        GetHandleFromCTX(ctx), rnn_desc.get(), &weightspace_size));

    weightspace = ctx.AllocateTmpTensor<T, DeviceContext>(
        {static_cast<int64_t>(weightspace_size)}, dev_ctx);
    void* weightspace_ptr = weightspace.mutable_data(ctx.GetPlace());
    auto w_x = parameter_lists[0][0];
    auto w_h = parameter_lists[0][1];
    auto b_x = parameter_lists[0][2];
    auto b_h = parameter_lists[0][3];
    auto actual_total_w_size =
        w_x.second + w_h.second + b_x.second + b_h.second;

    void* w_x_ptr = weightspace_ptr;
    void* w_h_ptr = static_cast<char*>(weightspace_ptr) + w_x.second;
    void* b_x_ptr =
        static_cast<char*>(weightspace_ptr) + w_x.second + w_h.second;
    void* b_h_ptr = static_cast<char*>(weightspace_ptr) + w_x.second +
                    w_h.second + b_x.second;

    memory::Copy(weightspace.place(), w_x_ptr, weightspace.place(), w_x.first,
                 w_x.second, nullptr);
    memory::Copy(weightspace.place(), w_h_ptr, weightspace.place(), w_h.first,
                 w_h.second, nullptr);
    memory::Copy(weightspace.place(), b_x_ptr, weightspace.place(), b_x.first,
                 b_x.second, nullptr);
    memory::Copy(weightspace.place(), b_h_ptr, weightspace.place(), b_h.first,
                 b_h.second, nullptr);

    if (is_bidirec) {
      auto bw_x = parameter_lists[0][4];
      auto bw_h = parameter_lists[0][5];
      auto bb_x = parameter_lists[0][6];
      auto bb_h = parameter_lists[0][7];
      void* bw_x_ptr =
          static_cast<char*>(weightspace_ptr) + actual_total_w_size;
      void* bw_h_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second;
      void* bb_x_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second + bw_h.second;
      void* bb_h_ptr = static_cast<char*>(weightspace_ptr) +
                       actual_total_w_size + bw_x.second + bw_h.second +
                       bb_x.second;
      actual_total_w_size +=
          bw_x.second + bw_h.second + bb_x.second + bb_h.second;

      memory::Copy(weightspace.place(), bw_x_ptr, weightspace.place(),
                   bw_x.first, bw_x.second, nullptr);
      memory::Copy(weightspace.place(), bw_h_ptr, weightspace.place(),
                   bw_h.first, bw_h.second, nullptr);
      memory::Copy(weightspace.place(), bb_x_ptr, weightspace.place(),
                   bb_x.first, bb_x.second, nullptr);
      memory::Copy(weightspace.place(), bb_h_ptr, weightspace.place(),
                   bb_h.first, bb_h.second, nullptr);
    }

    PADDLE_ENFORCE_EQ(weightspace_size, actual_total_w_size,
                      platform::errors::InvalidArgument(
                          "The weightsize doesn't match"
                          " weightspace_size:%d, actual_total_w_size:%d",
                          weightspace_size, actual_total_w_size));

    // get reservespace_ptr
    int gate_num = 4;
    int hidden_data_idx = (num_layers - 1);
    hidden_data_idx += (gate_num + 1) * num_layers;
    const int& block_size = direction_num * seq_len * batch_size * hidden_size;
    reserve_data->Resize({hidden_data_idx, block_size});

    reserve_data->mutable_data<T>(ctx.GetPlace());

    MLUCnnl::RNNForward(
        ctx, rnn_desc.get(), seq_len_vec.data(), weightspace_ptr,
        weightspace_size, input_seq_data_desc.get(), GetBasePtr(input),
        out_seq_data_desc.get(), GetBasePtr(output), hx_desc.get(),
        GetBasePtr(init_h), GetBasePtr(last_h), cx_desc.get(),
        GetBasePtr(init_c), GetBasePtr(last_c), GetBasePtr(reserve_data));

    if (has_seq_length) {
      // if has_seq_length, do mask out the output of cnnlRNNForwardTraining
      auto masked_mode = CNNL_MASKED_FILL;
      float off_value = 0.0f;

      framework::Tensor on_value_tensor(input->dtype());
      framework::Tensor masked_tensor(framework::TransToPhiDataType(VT::INT8));
      framework::Tensor h_masked_tensor(
          framework::TransToPhiDataType(VT::INT8));
      on_value_tensor.Resize({1});
      masked_tensor.Resize({seq_len, batch_size, direction_num * hidden_size});
      h_masked_tensor.Resize(
          {seq_len, batch_size, direction_num * hidden_size});

      on_value_tensor.mutable_data<T>(ctx.GetPlace());
      masked_tensor.mutable_data<int8_t>(ctx.GetPlace());
      int8_t* h_masked_ptr =
          h_masked_tensor.mutable_data<int8_t>(platform::CPUPlace());

      for (int t = 0; t < seq_len; ++t) {
        for (int n = 0; n < batch_size; ++n) {
          for (int c = 0; c < direction_num * hidden_size; ++c) {
            auto tmp_seq_len = seq_len_vec[n];
            auto offset = t * batch_size * direction_num * hidden_size +
                          n * direction_num * hidden_size + c;
            *(h_masked_ptr + offset) = t >= tmp_seq_len ? 1 : 0;
          }
        }
      }

      framework::TensorCopy(h_masked_tensor, ctx.GetPlace(), dev_ctx,
                            &masked_tensor);
      dev_ctx.Wait();

      FillMLUTensorWithHostValue(ctx, off_value, &on_value_tensor);
      MLUCnnlTensorDesc on_value_desc(on_value_tensor);
      MLUCnnlTensorDesc output_desc(*output);
      MLUCnnlTensorDesc masked_desc(masked_tensor);

      MLUCnnl::Mask(ctx, masked_mode, output_desc.get(), GetBasePtr(output),
                    masked_desc.get(), GetBasePtr(&masked_tensor),
                    on_value_desc.get(), GetBasePtr(&on_value_tensor),
                    output_desc.get(), GetBasePtr(output), nullptr);
    }
  }
};

// template <typename DeviceContext, typename T>
// class RNNGradMLUKernel : public framework::OpKernel<T> {
//   using XPUTyp = typename XPUTypeTrait<T>::Type;

//  public:
//   void Compute(const framework::ExecutionContext& ctx) const override {
//     // get the tensor pointer for the input
//     auto* input = ctx.Input<Tensor>("Input");
//     auto pre_state = ctx.MultiInput<Tensor>("PreState");
//     auto weight_list = ctx.MultiInput<framework::Tensor>("WeightList");
//     auto* output = ctx.Input<Tensor>("Out");
//     auto* reserve_data = ctx.Input<Tensor>("Reserve");
//     const int& num_layers = ctx.Attr<int>("num_layers");
//     const bool& is_bidirec = ctx.Attr<bool>("is_bidirec");
//     const float& dropout_prob = ctx.Attr<float>("dropout_prob");
//     const int& hidden_size = ctx.Attr<int>("hidden_size");
//     const std::string& mode = ctx.Attr<std::string>("mode");

//     bool has_seq_length = ctx.HasInput("SequenceLength");
//     const Tensor* sequence_length = nullptr;
//     if (has_seq_length) {
//       sequence_length = ctx.Input<Tensor>("SequenceLength");
//     }

//     PADDLE_ENFORCE_EQ(
//         mode, "LSTM",
//         platform::errors::InvalidArgument(
//             "XPU only support LSTM mode now, current mode is %s", mode));

//     auto init_h = pre_state[0];
//     auto init_c = pre_state[1];

//     auto output_grad = ctx.Input<Tensor>(framework::GradVarName("Out"));
//     auto state_grad =
//     ctx.MultiInput<Tensor>(framework::GradVarName("State")); auto last_h_grad
//     = state_grad[0]; auto last_c_grad = state_grad[1];

//     // get the tensor pointer for the output
//     auto* input_grad = ctx.Output<Tensor>(framework::GradVarName("Input"));
//     auto weight_grad_list = ctx.MultiOutput<framework::Tensor>(
//         framework::GradVarName("WeightList"));
//     auto pre_state_grad =
//         ctx.MultiOutput<Tensor>(framework::GradVarName("PreState"));
//     Tensor* init_h_grad = nullptr;
//     Tensor* init_c_grad = nullptr;
//     if (pre_state_grad.size() > 0) {  // has gradient
//       init_h_grad = pre_state_grad[0];
//       init_c_grad = pre_state_grad[1];
//     }

//     // check shape
//     const int& seq_len = input->dims()[0];
//     const int& batch_size = input->dims()[1];
//     const int& input_dim = input->dims()[2];
//     const int& direction_num = is_bidirec ? 2 : 1;
//     PADDLE_ENFORCE_EQ(
//         init_h->dims()[0], num_layers * direction_num,
//         platform::errors::InvalidArgument("The num_layers of in RNN layer
//         must"
//                                           " be the same as first dim of init
//                                           " "hidden, but received
//                                           num_layers:%d," " dim:%d",
//                                           num_layers, init_h->dims()[0]));

//     PADDLE_ENFORCE_EQ(
//         init_c->dims()[0], num_layers * direction_num,
//         platform::errors::InvalidArgument(
//             "The num_layers of in RNN layer must"
//             " be the same as first dim of cell state hidden, but received"
//             " num_layers:%d, dim:%d",
//             num_layers, init_c->dims()[0]));

//     std::vector<std::vector<const T*>> parameter_lists;
//     parameter_lists.resize(num_layers);
//     reset_parameter_vector(weight_list, num_layers, is_bidirec,
//                            &parameter_lists);

//     for (unsigned int i = 0; i < weight_grad_list.size(); ++i) {
//       weight_grad_list[i]->mutable_data<T>(ctx.GetPlace());
//     }
//     std::vector<std::vector<T*>> parameter_lists_grad;
//     parameter_lists_grad.resize(num_layers);
//     reset_parameter_vector(weight_grad_list, num_layers, is_bidirec,
//                            &parameter_lists_grad);

//     // allocate the memory and initization the input_grad
//     input_grad->mutable_data<T>(input->dims(), ctx.GetPlace());
//     auto& dev_ctx = ctx.template device_context<DeviceContext>();
//     phi::funcs::SetConstant<platform::XPUDeviceContext, T> zero;
//     zero(dev_ctx, input_grad, static_cast<T>(0.0));

//     Tensor a, b;
//     Tensor* dynamic_grad_pre_h = &a;
//     Tensor* dynamic_grad_pre_c = &b;
//     if (init_h_grad) {
//       init_h_grad->mutable_data<T>(last_h_grad->dims(), ctx.GetPlace());
//       zero(dev_ctx, init_h_grad, static_cast<T>(0.0));
//     } else {
//       dynamic_grad_pre_h->Resize(last_h_grad->dims());
//       dynamic_grad_pre_h->mutable_data<T>(ctx.GetPlace());
//       zero(dev_ctx, dynamic_grad_pre_h, static_cast<T>(0.0));
//       init_h_grad = dynamic_grad_pre_h;
//     }
//     if (init_c_grad) {
//       init_c_grad->mutable_data<T>(last_c_grad->dims(), ctx.GetPlace());
//     } else {
//       dynamic_grad_pre_c->Resize(last_h_grad->dims());
//       dynamic_grad_pre_c->mutable_data<T>(ctx.GetPlace());
//       init_c_grad = dynamic_grad_pre_c;
//     }

//     Tensor temp_input_grad_1, temp_input_grad_2;
//     T* input_grad_1_ptr = nullptr;
//     T* input_grad_2_ptr = nullptr;
//     if (num_layers >= 2) {
//       temp_input_grad_1.Resize(output_grad->dims());
//       input_grad_1_ptr = temp_input_grad_1.mutable_data<T>(ctx.GetPlace());
//     }
//     if (num_layers >= 3) {
//       temp_input_grad_2.Resize(output_grad->dims());
//       input_grad_2_ptr = temp_input_grad_2.mutable_data<T>(ctx.GetPlace());
//     }

//     // get ptr from tensor
//     auto x = input->data<T>();
//     auto init_h_ptr = init_h->data<T>();
//     auto init_c_ptr = init_c->data<T>();
//     auto y = output->data<T>();
//     auto y_grad = output_grad->data<T>();
//     auto last_h_grad_ptr = last_h_grad->data<T>();
//     auto last_c_grad_ptr = last_c_grad->data<T>();
//     auto x_grad = input_grad->data<T>();
//     auto init_h_grad_ptr = init_h_grad->data<T>();
//     auto init_c_grad_ptr = init_c_grad->data<T>();
//     const int& block_size = direction_num * seq_len * batch_size *
//     hidden_size; auto i_f_g_o_ptr = reserve_data->data<T>(); auto c_ptr =
//     i_f_g_o_ptr + num_layers * block_size * 4; auto hidden_data_ptr = c_ptr +
//     num_layers * block_size * 1; int state_offset = pre_state[0]->dims()[1] *
//     pre_state[0]->dims()[2];

//     std::vector<int> seq_len_tensor(batch_size, seq_len);
//     if (has_seq_length) {
//       seq_len_tensor = operators::GetDataFromTensor(sequence_length);
//     }

//     for (int i = num_layers - 1; i >= 0; --i) {
//       // the layer input output had saved, just use the data
//       auto w_x = parameter_lists[i][0];
//       auto w_h = parameter_lists[i][1];
//       auto bw_x = parameter_lists[i][4];
//       auto bw_h = parameter_lists[i][5];

//       auto i_f_g_o = i_f_g_o_ptr + i * block_size * 4;
//       auto c = c_ptr + i * block_size;

//       Tensor layer_input_t;
//       auto layer_input = x;
//       if (i > 0) {
//         layer_input_t.Resize(output->dims());
//         layer_input = layer_input_t.mutable_data<T>(ctx.GetPlace());
//         float scale = static_cast<float>(1.0f - dropout_prob);
//         auto hidden_data = hidden_data_ptr + (i - 1) * block_size;
//         int r = xpu::scale(dev_ctx.x_context(),
//                            reinterpret_cast<const XPUTyp*>(hidden_data),
//                            const_cast<XPUTyp*>(layer_input), output->numel(),
//                            false, scale, 0.0f);
//         PADDLE_ENFORCE_XDNN_SUCCESS(r, "scale");
//       } else {
//         layer_input = x;
//       }

//       auto layer_output = y;
//       if (i == num_layers - 1) {
//         layer_output = y;
//       } else {
//         layer_output = hidden_data_ptr + i * block_size;
//       }

//       const T* cur_input_ptr = nullptr;
//       if (i == num_layers - 1) {
//         cur_input_ptr = y_grad;
//       } else if (i % 2 != 0) {
//         cur_input_ptr = input_grad_2_ptr;
//       } else {
//         cur_input_ptr = input_grad_1_ptr;
//       }

//       T* cur_output_ptr = nullptr;
//       int cur_xdim = -1;
//       if (i == 0) {
//         cur_output_ptr = x_grad;
//         cur_xdim = input_dim;
//       } else if (i % 2 != 0) {
//         cur_output_ptr = input_grad_1_ptr;
//         cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
//       } else {
//         cur_output_ptr = input_grad_2_ptr;
//         cur_xdim = is_bidirec ? 2 * hidden_size : hidden_size;
//       }

//       auto w_x_grad = parameter_lists_grad[i][0];
//       auto w_h_grad = parameter_lists_grad[i][1];
//       auto b_x_grad = parameter_lists_grad[i][2];
//       auto b_h_grad = parameter_lists_grad[i][3];

//       auto h_0 = init_h_ptr + direction_num * i * state_offset;
//       auto c_0 = init_c_ptr + direction_num * i * state_offset;

//       auto h_0_grad = init_h_grad_ptr + direction_num * i * state_offset;
//       auto c_0_grad = init_c_grad_ptr + direction_num * i * state_offset;
//       auto h_t_grad = last_h_grad_ptr + direction_num * i * state_offset;
//       auto c_t_grad = last_c_grad_ptr + direction_num * i * state_offset;

//       if (is_bidirec) {
//         auto bw_x_grad = parameter_lists_grad[i][4];
//         auto bw_h_grad = parameter_lists_grad[i][5];
//         auto bb_x_grad = parameter_lists_grad[i][6];
//         auto bb_h_grad = parameter_lists_grad[i][7];

//         int r = xpu::bilstm_grad<T, T, int16_t>(
//             dev_ctx.x_context(), (const T*)layer_input, (const T*)h_0,
//             (const T*)c_0, (const T*)w_x, (const T*)w_h, (const T*)bw_x,
//             (const T*)bw_h, (const T*)layer_output, (const T*)cur_input_ptr,
//             (const T*)h_t_grad, (const T*)c_t_grad,
//             reinterpret_cast<T*>(cur_output_ptr),
//             reinterpret_cast<T*>(h_0_grad), reinterpret_cast<T*>(c_0_grad),
//             w_x_grad, w_h_grad, b_x_grad, b_h_grad, bw_x_grad, bw_h_grad,
//             bb_x_grad, bb_h_grad, batch_size, cur_xdim, hidden_size, seq_len,
//             seq_len_tensor, nullptr, nullptr, nullptr, nullptr, nullptr,
//             nullptr, i_f_g_o, c);

//         PADDLE_ENFORCE_XDNN_SUCCESS(r, "bilstm_grad");
//       } else {
//         int r = xpu::lstm_grad<T, T, int16_t>(
//             dev_ctx.x_context(), (const T*)layer_input, (const T*)h_0,
//             (const T*)c_0, (const T*)w_x, (const T*)w_h, (const
//             T*)layer_output, (const T*)cur_input_ptr, (const T*)h_t_grad,
//             (const T*)c_t_grad, reinterpret_cast<T*>(cur_output_ptr),
//             reinterpret_cast<T*>(h_0_grad), reinterpret_cast<T*>(c_0_grad),
//             w_x_grad, w_h_grad, b_x_grad, b_h_grad, batch_size, cur_xdim,
//             hidden_size, seq_len, seq_len_tensor, nullptr, nullptr, nullptr,
//             nullptr, i_f_g_o, c);

//         PADDLE_ENFORCE_XDNN_SUCCESS(r, "lstm_grad");
//       }
//     }
//   }
// };

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(
    rnn, ops::RNNMLUKernel<paddle::platform::MLUDeviceContext, float>);
// REGISTER_OP_MLU_KERNEL(
//     rnn_grad, ops::RNNMLUGradKernel<paddle::platform::XPUDeviceContext,
//     float>);
