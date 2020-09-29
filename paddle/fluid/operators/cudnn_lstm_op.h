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
#include <string>
#include <vector>
#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/operators/dropout_op.h"

namespace paddle {
namespace operators {

using LoDTensor = framework::LoDTensor;
using Tensor = framework::Tensor;
using TensorList = std::vector<framework::Tensor>;

template <typename T>
struct Cell {
  virtual ~Cell() {}
  virtual void operator()(const Tensor* input, const TensorList& vec,
                          const Tensor* init_h, const Tensor* init_c,
                          Tensor* last_h, Tensor* last_c, Tensor* output,
                          const int& layer_idx, const int& init_offset,
                          const int& time_step) {}
};

template <typename T>
struct LSTMCell : Cell<T> {
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const Tensor* init_h, const Tensor* init_c, Tensor* last_h,
                  Tensor* last_c, Tensor* output) {
    // TODO(wawltor)
    return;
  }
};

// input_w = input * w_ih + b_ih
// input_tensors = unbind(input_w)
// input_tensors[i], w_hh, b_hh
// init_h N*D

template <typename T>
struct Layer {
  virtual ~Layer() {}
  virtual void operator()(Tensor* input) const = 0;
};

template <typename T>
struct SingleLayer {
  explicit SingleLayer(Cell<T>& cell) : cell_(cell) {}
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const Tensor* init_h, const Tensor* init_c, Tensor* last_h,
                  Tensor* last_c, Tensor* output, const int& layer_idx,
                  const int& init_offset) {
    const int& time_step = input->dims()[0];
    TensorList output_tensors;
    output_tensors.reserve(time_step);
    for (int i = 0; i < time_step; i++) {
      cell_(context, input, vec, init_h, init_c, last_h, last_c, output);
      init_h = last_h;
      init_c = last_h;
    }
    // TODO(wawltor)
  }

  // Cell for the rnn module
  Cell<T> cell_;
};

template <typename T>
struct BidirLayer {
  explicit BidirLayer(Cell<T>& cell) : cell_(cell) {}
  void operator()(const framework::ExecutionContext& context,
                  const Tensor* input, const TensorList& vec,
                  const Tensor* init_h, const Tensor* init_c, Tensor* last_h,
                  Tensor* last_c, Tensor* output, const int& layer_idx,
                  const int& init_offset) {}
  Cell<T> cell_;
};

template <typename T>
void dropout_cpu_function_inplace(const framework::ExecutionContext& context,
                                  Tensor* x, Tensor* mask,
                                  const float& dropout_prob,
                                  const int& seed_number,
                                  const bool& upscale_in_train,
                                  const bool& is_test) {
  auto* x_data = x->data<T>();
  if (!is_test) {
    size_t size = framework::product(mask->dims());

    if (!mask->IsInitialized()) {
      auto mask_data = mask->mutable_data<uint8_t>(context.GetPlace());
      // Special case when dropout_prob is 1.0
      if (dropout_prob == 1.0f) {
        std::memset(x_data, 0, size * sizeof(*x_data));
        std::memset(mask_data, 0, size * sizeof(*mask_data));  // NOLINT
        return;
      }
      auto engine = framework::GetCPURandomEngine(seed_number);
      std::uniform_real_distribution<float> dist(0, 1);
      for (size_t i = 0; i < size; ++i) {
        if (dist(*engine) < dropout_prob) {
          mask_data[i] = 0;
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
      std::memset(x_data, 0, size * sizeof(*x_data));
      return;
    }
    for (size_t i = 0; i < size; ++i) {
      if (mask_data[i] == 1) {
        if (upscale_in_train) {
          x_data[i] /= static_cast<T>(1.0f - dropout_prob);
        }
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
std::vector<TensorList> parameter_split(
    const Tensor* weight, const int& gate_num, const int& layers_num,
    const int& input_size, const int& hidden_size, const int& is_bidirec) {
  // if the weight of RNN is flatten, we need to convert the
  // the flattened weight to single split tensor
  std::vector<TensorList> params_vec;
  params_vec.reserve(layers_num);

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
    params_vec.emplace_back(tensor_list);
  }
  return params_vec;
}

template <typename CellType, template <typename> class LayerT,
          template <typename> class BidirLayerT, typename T>
void CacluateLSTMLayer(const framework::ExecutionContext& ctx,
                       const Tensor* input, const Tensor* weight,
                       const Tensor* init_h, const Tensor* init_c,
                       Tensor* last_h, Tensor* last_c, Tensor* output,
                       Tensor* dropout_mask, const int& num_layers,
                       const int& gate_num, const int& input_size,
                       const int& hidden_size, const bool& is_bidirec,
                       const std::string& cell_type, const float& dropout_prob,
                       const bool& is_test, const int& seed) {
  // check the dim message of init state
  const auto& init_h_dims = init_h->dims();
  const auto& init_c_dims = init_c->dims();
  PADDLE_ENFORCE_EQ(init_h_dims[0], num_layers,
                    platform::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of init hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers, init_h_dims[0]));
  PADDLE_ENFORCE_EQ(init_c_dims[0], num_layers,
                    platform::errors::InvalidArgument(
                        "The num_layers of in RNN layer must be the same as "
                        "first dim of cell state hidden, but received"
                        " num_layers:%d, dim:%d",
                        num_layers, init_h_dims[0]));
  // define the swap function to swap the pointer
  auto SwapPoniter = [](Tensor* a, Tensor* b) {
    Tensor* c;
    c = a;
    a = b;
    b = c;
  };

  CellType cell;
  const int& init_offset = init_h->numel() / num_layers;
  const std::vector<TensorList>& parameter_lists = parameter_split<T>(
      weight, gate_num, num_layers, input_size, hidden_size, is_bidirec);
  Tensor* input_holder;
  Tensor* output_holder = output;
  Tensor temp;
  bool has_allocate_mem = false;
  for (int i = 0; i < num_layers; i++) {
    if (i > 0) {
      if (!has_allocate_mem) {
        temp.Resize(output->dims());
        temp.mutable_data<T>(ctx.GetPlace());
        input_holder = &temp;
        has_allocate_mem = true;
      }
      SwapPoniter(output, input_holder);
      if (dropout_prob != 0 && (!is_test)) {
        // only train mode just need dropout
        dropout_cpu_function_inplace<T>(ctx, input_holder, dropout_mask,
                                        dropout_prob, seed,
                                        true /*upscale_in_train*/, is_test);
      }
    }
    if (is_bidirec) {
      BidirLayerT<T> layer(cell);
      if (i == 0) {
        layer(ctx, input, parameter_lists[i], init_h, init_c, last_h, last_c,
              output_holder, i, init_offset);
      } else {
        layer(ctx, input_holder, parameter_lists[i], init_h, init_c, last_h,
              last_c, output_holder, i, init_offset);
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
    auto* weight = ctx.Input<Tensor>("W");
    auto* init_h = ctx.Input<Tensor>("InitH");
    auto* init_c = ctx.Input<Tensor>("InitC");
    auto* last_h = ctx.Output<Tensor>("LastH");
    auto* last_c = ctx.Output<Tensor>("LastC");
    auto* output = ctx.Output<Tensor>("Out");
    auto* dropout_mask = ctx.Output<Tensor>("StateOut");

    // init the output and allocate the memory
    output->mutable_data<T>(ctx.GetPlace());
    // dropout_mask->mutable_data<T>(ctx.GetPlace());
    if (cell_type == "lstm") {
      CacluateLSTMLayer<LSTMCell<T>, SingleLayer, BidirLayer, T>(
          ctx, input, weight, init_h, init_c, last_h, last_c, output,
          dropout_mask, num_layers, 4, input_size, hidden_size, is_bidirec,
          cell_type, dropout_prob, is_test, seed);
    }
  }
};

}  // namespace operators
}  // namespace paddle
