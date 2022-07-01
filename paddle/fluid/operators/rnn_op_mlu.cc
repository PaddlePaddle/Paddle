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
    const std::vector<TensorType>& raw_params_vec,
    const int& num_layers,
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
    int out_dim_arr[in_out_dim_num] = {
        seq_len, batch_size, direction_num * hidden_size};
    int proj_size = hidden_size;

    std::vector<int> seq_len_vec(batch_size, seq_len);
    if (has_seq_length) {  // set seq_len if no padding, otherwise seq_len for
                           // each element.
      seq_len_vec = operators::GetDataFromTensor(sequence_length);
    }
    cnnlDirectionMode_t direction =
        is_bidirec ? CNNL_RNN_BIDIRECTIONAL : CNNL_RNN_UNIDIRECTIONAL;

    PADDLE_ENFORCE_EQ(
        mode,
        "LSTM",
        platform::errors::InvalidArgument(
            "MLU only support LSTM mode now, current mode is %s", mode));
    PADDLE_ENFORCE_EQ(
        num_layers,
        1,
        platform::errors::InvalidArgument(
            "MLU only support 1 num_layers, current num_layers is %s",
            num_layers));
    PADDLE_ENFORCE_EQ(
        init_h->dims()[0],
        num_layers * direction_num,
        platform::errors::InvalidArgument("The num_layers of in RNN layer must"
                                          " be the same as first dim of init "
                                          "hidden, but received num_layers:%d,"
                                          " dim:%d",
                                          num_layers,
                                          init_h->dims()[0]));

    PADDLE_ENFORCE_EQ(
        init_c->dims()[0],
        num_layers * direction_num,
        platform::errors::InvalidArgument(
            "The num_layers of in RNN layer must"
            " be the same as first dim of cell state hidden, but received"
            " num_layers:%d, dim:%d",
            num_layers,
            init_c->dims()[0]));

    // weightlist
    std::vector<std::vector<std::pair<const T*, size_t>>> parameter_lists;
    parameter_lists.resize(num_layers);
    reset_parameter_vector(
        weight_list, num_layers, is_bidirec, &parameter_lists);

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());  // -> y in cnnl
    last_h->mutable_data<T>(ctx.GetPlace());  // -> hy in cnnl
    last_c->mutable_data<T>(ctx.GetPlace());  // -> cy in cnnl

    MLUSeqDataDesc input_seq_data_desc(CNNL_SEQDATA_TNC,
                                       ToCnnlDataType(input->dtype()),
                                       in_out_dim_num,
                                       in_dim_arr,
                                       static_cast<int>(seq_len_vec.size()),
                                       seq_len_vec.data(),
                                       nullptr);
    MLUSeqDataDesc out_seq_data_desc(CNNL_SEQDATA_TNC,
                                     ToCnnlDataType(input->dtype()),
                                     in_out_dim_num,
                                     out_dim_arr,
                                     static_cast<int>(seq_len_vec.size()),
                                     seq_len_vec.data(),
                                     nullptr);
    MLUCnnlTensorDesc hx_desc(*init_h);
    MLUCnnlTensorDesc cx_desc(*init_c);

    MLURNNDesc rnn_desc(CNNL_LSTM,
                        CNNL_RNN_DOUBLE_BIAS,
                        direction,
                        CNNL_RNN_LINEAR_INPUT,
                        ToCnnlDataType(input->dtype()),
                        ToCnnlDataType(input->dtype()),
                        input_dim,
                        hidden_size,
                        /*projection*/ proj_size,
                        num_layers,
                        nullptr,
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

    memory::Copy(weightspace.place(),
                 w_x_ptr,
                 weightspace.place(),
                 w_x.first,
                 w_x.second,
                 nullptr);
    memory::Copy(weightspace.place(),
                 w_h_ptr,
                 weightspace.place(),
                 w_h.first,
                 w_h.second,
                 nullptr);
    memory::Copy(weightspace.place(),
                 b_x_ptr,
                 weightspace.place(),
                 b_x.first,
                 b_x.second,
                 nullptr);
    memory::Copy(weightspace.place(),
                 b_h_ptr,
                 weightspace.place(),
                 b_h.first,
                 b_h.second,
                 nullptr);

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

      memory::Copy(weightspace.place(),
                   bw_x_ptr,
                   weightspace.place(),
                   bw_x.first,
                   bw_x.second,
                   nullptr);
      memory::Copy(weightspace.place(),
                   bw_h_ptr,
                   weightspace.place(),
                   bw_h.first,
                   bw_h.second,
                   nullptr);
      memory::Copy(weightspace.place(),
                   bb_x_ptr,
                   weightspace.place(),
                   bb_x.first,
                   bb_x.second,
                   nullptr);
      memory::Copy(weightspace.place(),
                   bb_h_ptr,
                   weightspace.place(),
                   bb_h.first,
                   bb_h.second,
                   nullptr);
    }

    PADDLE_ENFORCE_EQ(weightspace_size,
                      actual_total_w_size,
                      platform::errors::InvalidArgument(
                          "The weightsize doesn't match"
                          " weightspace_size:%d, actual_total_w_size:%d",
                          weightspace_size,
                          actual_total_w_size));

    // get reservespace_ptr
    int gate_num = 4;
    int hidden_data_idx = (num_layers - 1);
    hidden_data_idx += (gate_num + 1) * num_layers;
    const int& block_size = direction_num * seq_len * batch_size * hidden_size;
    reserve_data->Resize({hidden_data_idx, block_size});

    reserve_data->mutable_data<T>(ctx.GetPlace());

    MLUCnnl::RNNForward(ctx,
                        rnn_desc.get(),
                        seq_len_vec.data(),
                        weightspace_ptr,
                        weightspace_size,
                        input_seq_data_desc.get(),
                        GetBasePtr(input),
                        out_seq_data_desc.get(),
                        GetBasePtr(output),
                        hx_desc.get(),
                        GetBasePtr(init_h),
                        GetBasePtr(last_h),
                        cx_desc.get(),
                        GetBasePtr(init_c),
                        GetBasePtr(last_c),
                        GetBasePtr(reserve_data));

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

      framework::TensorCopy(
          h_masked_tensor, ctx.GetPlace(), dev_ctx, &masked_tensor);
      dev_ctx.Wait();

      FillMLUTensorWithHostValue(ctx, off_value, &on_value_tensor);
      MLUCnnlTensorDesc on_value_desc(on_value_tensor);
      MLUCnnlTensorDesc output_desc(*output);
      MLUCnnlTensorDesc masked_desc(masked_tensor);

      MLUCnnl::Mask(ctx,
                    masked_mode,
                    output_desc.get(),
                    GetBasePtr(output),
                    masked_desc.get(),
                    GetBasePtr(&masked_tensor),
                    on_value_desc.get(),
                    GetBasePtr(&on_value_tensor),
                    output_desc.get(),
                    GetBasePtr(output),
                    nullptr);
    }
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OP_MLU_KERNEL(
    rnn, ops::RNNMLUKernel<paddle::platform::MLUDeviceContext, float>);
