/* Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.

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
#include <vector>
#include "paddle/fluid/platform/device/gcu/register/register.h"

namespace paddle {
namespace platform {
namespace gcu {
/* the definition of rnn: paddle/fluid/operators/rnn_op.cc */
const char *const kRNN = "rnn";
const char *const kRNNGrad = "rnn_grad";

static void printVector(std::vector<int64_t> const &input, std::string name) {
  std::cout << ">>>> " << name << ": ";
  int size = static_cast<int>(input.size());
  for (int i = 0; i < size; i++) {
    std::cout << input.at(i) << ' ';
  }
  std::cout << std::endl;
}

static builder::Op transpose_func(const builder::Op &op,
                                  const std::vector<int64_t> &perm) {
  auto op_shape = op.GetType().GetShape();
  auto dtype = op.GetType().GetPrimitiveType();
  std::vector<int64_t> new_shape(op_shape);
  int perm_size = perm.size();
  for (int i = 0; i < perm_size; i++) {
    new_shape[i] = op_shape[perm[i]];
  }
  return builder::Transpose(op, perm, builder::Type(new_shape, dtype));
}

static void split_weight_list(const std::vector<builder::Op> &weight_list,
                              std::vector<builder::Op> *W_T_forward_layers,
                              std::vector<builder::Op> *R_T_forward_layers,
                              std::vector<builder::Op> *B_forward_layers,
                              std::vector<builder::Op> *W_T_backward_layers,
                              std::vector<builder::Op> *R_T_backward_layers,
                              std::vector<builder::Op> *B_backward_layers,
                              int64_t num_directions,
                              int64_t num_layers) {
  /* e.g. num_directions = 2, num_layers = 2,
          weight_list.size() = 2 * 2 * 4 = 16
     weight_list[0]: W_forward_layer0  (4*hidden_size, input_size)
     weight_list[1]: R_forward_layer0  (4*hidden_size, hidden_size)
     weight_list[2]: W_backward_layer0  (4*hidden_size, input_size)
     weight_list[3]: R_backward_layer0  (4*hidden_size, hidden_size)
     weight_list[4]: W_forward_layer1  (4*hidden_size,
     num_directions*hidden_size) weight_list[5]: R_forward_layer1
     (4*hidden_size, hidden_size) weight_list[6]: W_backward_layer1
     (4*hidden_size, num_directions*hidden_size) weight_list[7]:
     R_backward_layer1  (4*hidden_size, hidden_size) weight_list[8]:
     WB_forward_layer0  (4*hidden_size, ) weight_list[9]: RB_forward_layer0
     (4*hidden_size, ) weight_list[10]: WB_backward_layer0  (4*hidden_size, )
     weight_list[11]: RB_backward_layer0  (4*hidden_size, )
     weight_list[12]: WB_forward_layer1  (4*hidden_size, )
     weight_list[13]: RB_forward_layer1  (4*hidden_size, )
     weight_list[14]: WB_backward_layer1  (4*hidden_size, )
     weight_list[15]: RB_backward_layer1  (4*hidden_size, )
  */
  std::vector<int64_t> perm = {1, 0};
  int64_t index_delta_layer = (num_directions == 2) ? 4 : 2;
  int64_t weight_list_size = weight_list.size();
  for (int64_t layer_i = 0; layer_i < num_layers; layer_i++) {
    int64_t W_T_forward_idx = layer_i * index_delta_layer;
    (*W_T_forward_layers)
        .push_back(transpose_func(weight_list[W_T_forward_idx], perm));
    (*R_T_forward_layers)
        .push_back(transpose_func(weight_list[W_T_forward_idx + 1], perm));
    if (num_directions == 2) {
      (*W_T_backward_layers)
          .push_back(transpose_func(weight_list[W_T_forward_idx + 2], perm));
      (*R_T_backward_layers)
          .push_back(transpose_func(weight_list[W_T_forward_idx + 3], perm));
    }
    int64_t WB_forward_idx = weight_list_size / 2 + layer_i * index_delta_layer;
    auto WB_forward = weight_list[WB_forward_idx];
    auto RB_forward = weight_list[WB_forward_idx + 1];
    (*B_forward_layers)
        .push_back(
            builder::Add(WB_forward, RB_forward, {}, WB_forward.GetType()));
    if (num_directions == 2) {
      auto WB_backward = weight_list[WB_forward_idx + 2];
      auto RB_backward = weight_list[WB_forward_idx + 3];
      (*B_backward_layers)
          .push_back(builder::Add(
              WB_backward, RB_backward, {}, WB_backward.GetType()));
    }
  }
}

static builder::Op squeeze_dim(builder::Op op, int64_t axis) {
  // op shape (1, d1, d2,) -> (d1, d2),  or (d1, 1, d2) -> (d1, d2)
  auto op_shape = op.GetType().GetShape();
  auto dtype = op.GetType().GetPrimitiveType();
  if (op_shape[axis] != 1) return op;
  std::vector<int64_t> new_shape;
  int64_t op_shape_size = op_shape.size();
  for (int i = 0; i < op_shape_size; i++) {
    if (i != axis) new_shape.push_back(op_shape[i]);
  }
  return builder::Reshape(op, builder::Type(new_shape, dtype));
}

static builder::Op unsqueeze_dim(builder::Op op, int64_t axis) {
  auto op_shape = op.GetType().GetShape();
  auto dtype = op.GetType().GetPrimitiveType();
  std::vector<int64_t> new_shape;
  int64_t op_shape_size = op_shape.size();
  for (int i = 0; i < op_shape_size; i++) {
    if (i == axis) {
      new_shape.push_back(1);
    }
    new_shape.push_back(op_shape[i]);
  }
  return builder::Reshape(op, builder::Type(new_shape, dtype));
}

static builder::Op concatenate_with_unsqueeze(
    const std::vector<builder::Op> &ops, int64_t axis, builder::Type type) {
  // size:seq_len, each: (batch_size, num_directions*hidden_size)
  std::vector<builder::Op> unsqueezed;
  int64_t size = ops.size();
  for (int64_t t = 0; t < size; t++) {
    unsqueezed.push_back(unsqueeze_dim(ops[t], axis));
  }
  return builder::Concatenate(unsqueezed, axis, type);
}

static std::vector<builder::Op> concatenate_two_vectors(
    const std::vector<builder::Op> &ops1,
    const std::vector<builder::Op> &ops2,
    int64_t each_axis,
    builder::Type each_type) {
  std::vector<builder::Op> ret;
  PADDLE_ENFORCE(
      ops1.size() == ops2.size(),
      platform::errors::InvalidArgument(
          "Two vectors should have same size, but got size1 = %d, size2 = %d",
          ops1.size(),
          ops2.size()));
  int64_t vec_len = ops1.size();
  for (int64_t t = 0; t < vec_len; ++t) {
    ret.push_back(
        builder::Concatenate({ops1[t], ops2[t]}, each_axis, each_type));
  }
  return ret;
}

static std::vector<builder::Op> get_slice_ops(builder::Op op,
                                              const int64_t axis,
                                              const int64_t division,
                                              bool need_squeeze) {
  std::vector<builder::Op> result;
  auto op_shape = op.GetType().GetShape();
  auto dtype = op.GetType().GetPrimitiveType();
  std::vector<int64_t> new_start(op_shape.size(), 0);
  std::vector<int64_t> new_end(op_shape);
  std::vector<int64_t> new_step(op_shape.size(), 1);
  int64_t step = op_shape[axis] / division;
  std::vector<int64_t> slice_offset;
  for (int64_t i = 0; i < division + 1; i++) {
    slice_offset.push_back(i * step);
  }
  std::vector<int64_t> slice_shape(op_shape);
  slice_shape[axis] = step;
  auto slice_type = builder::Type(slice_shape, dtype);
  for (int64_t i = 0; i < division; i++) {
    new_start[axis] = slice_offset[i];
    new_end[axis] = slice_offset[i + 1];
    auto current_slice =
        builder::Slice(op, new_start, new_end, new_step, slice_type);
    if (need_squeeze) {
      current_slice = squeeze_dim(current_slice, axis);
    }
    result.push_back(current_slice);
  }
  return result;
}

static void go_lstm_one_direction(const std::vector<builder::Op> &X_slices,
                                  const builder::Op &W_T,
                                  const builder::Op &R_T,
                                  const builder::Op &bias,
                                  const builder::Op &initial_h,
                                  const builder::Op &initial_c,
                                  const int64_t batch_size,
                                  const int64_t hidden_size,
                                  const int64_t seq_len,
                                  const builder::PrimitiveType input_dtype,
                                  std::vector<builder::Op> *h_list,
                                  builder::Op *h_o,
                                  builder::Op *c_o) {
  /*
      X_slices: size: seq_len, each: (batch_size, input_size)
      W_T: (input_size, 4 * hidden_size)
      R_T: (hidden_size, 4 * hidden_size)
      bias: (4 * hidden_size, )
      initial_h: (batch_size, hidden_size)
      initial_c: (batch_size, hidden_size)
      h_list: size: seq_len, each: (batch_size, hidden_size)
      h_o: (batch_size, hidden_size)
      c_o: (batch_size, hidden_size)
  */
  auto iofc_type = builder::Type({batch_size, 4 * hidden_size}, input_dtype);

  // (batch_size, 4 * hidden_size)
  builder::Op x_dot_w_op, h_dot_r_op, dotsum_op, iofc_op;
  // each result[i]: (batch_size, hidden_size)
  std::vector<builder::Op> result;
  // (batch_size, hidden_size)
  builder::Op i_op, o_op, f_op, c_op;
  auto gate_type = builder::Type({batch_size, hidden_size}, input_dtype);
  // (batch_size, hidden_size)
  builder::Op f_mul_c_t_op, i_mul_c_op, tanh_c_t_op;

  builder::Op h_t = initial_h;  // (batch_size, hidden_size)
  builder::Op c_t = initial_c;  // (batch_size, hidden_size)

  for (int64_t i = 0; i < seq_len; i++) {
    std::vector<const char *> precision_config = {};
    x_dot_w_op = builder::Dot(X_slices[i], W_T, precision_config, iofc_type);
    h_dot_r_op = builder::Dot(h_t, R_T, precision_config, iofc_type);
    dotsum_op = builder::Add(x_dot_w_op, h_dot_r_op, {}, iofc_type);
    auto bias_brc = builder::BroadcastInDim(bias, {1}, iofc_type);
    iofc_op = builder::Add(dotsum_op, bias_brc, {}, iofc_type);

    result = get_slice_ops(iofc_op, 1, 4, false);
    c_op = builder::Tanh(result[3], gate_type);
    i_op = builder::Sigmoid(result[0], gate_type);
    o_op = builder::Sigmoid(result[1], gate_type);
    f_op = builder::Sigmoid(result[2], gate_type);

    f_mul_c_t_op = builder::Mul(f_op, c_t, {}, gate_type);
    i_mul_c_op = builder::Mul(i_op, c_op, {}, gate_type);
    c_t = builder::Add(f_mul_c_t_op, i_mul_c_op, {}, gate_type);
    tanh_c_t_op = builder::Tanh(c_t, gate_type);
    h_t = builder::Mul(tanh_c_t_op, o_op, {}, gate_type);

    (*h_list).push_back(h_t);
  }
  *h_o = h_t;
  *c_o = c_t;
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RNNEquivalenceTrans) {
  std::cout << "<<< hello rnn! >>>" << std::endl;
  auto *op = node->Op();

  // get attributes
  auto mode = PADDLE_GET_CONST(std::string, op->GetAttr("mode"));
  PADDLE_ENFORCE(
      "LSTM" == mode,
      platform::errors::InvalidArgument(
          "Now gcu only support LSTM mode for rnn op, but got mode = %s.",
          mode));
  auto is_bidirec = PADDLE_GET_CONST(bool, op->GetAttr("is_bidirec"));
  int64_t num_directions = is_bidirec ? 2 : 1;

  // get input ops and details
  // (seq_len, batch_size, input_size)
  builder::Op X = *(map_inputs["Input"].at(0));
  PADDLE_ENFORCE(
      2 == map_inputs["PreState"].size(),
      platform::errors::InvalidArgument(
          "rnn op's input PreState should have size 2, but got size = %d.",
          map_inputs["PreState"].size()));
  // (num_layers * num_directions, batch_size, hidden_size)
  builder::Op init_h = *(map_inputs["PreState"].at(0));
  builder::Op init_c = *(map_inputs["PreState"].at(1));

  auto input_dtype = X.GetType().GetPrimitiveType();
  auto input_shape = X.GetType().GetShape();
  auto init_h_shape = init_h.GetType().GetShape();
  int64_t seq_len = input_shape[0];
  int64_t batch_size = input_shape[1];
  int64_t input_size = input_shape[2];
  int64_t num_layers = init_h_shape[0] / num_directions;
  int64_t hidden_size = init_h_shape[2];
  std::cout << "is_bidirec: " << is_bidirec << std::endl;
  std::cout << "seq_len: " << seq_len << std::endl;
  std::cout << "batch_size: " << batch_size << std::endl;
  std::cout << "input_size: " << input_size << std::endl;
  std::cout << "num_layers: " << num_layers << std::endl;
  std::cout << "hidden_size: " << hidden_size << std::endl;

  std::vector<builder::Op> weight_list;
  int64_t weight_list_size = map_inputs["WeightList"].size();
  PADDLE_ENFORCE(
      num_layers * num_directions * 4 == weight_list_size,
      platform::errors::InvalidArgument(
          "rnn op's input WeightList should have size %d, but got size = %d.",
          num_layers * num_directions * 4,
          weight_list_size));
  for (int64_t i = 0; i < weight_list_size; i++) {
    builder::Op weight_ = *(map_inputs["WeightList"].at(i));
    weight_list.push_back(weight_);
  }

  std::cout << "weight_list.size: " << weight_list_size << std::endl;
  for (int64_t i = 0; i < weight_list_size; i++) {
    builder::Op weight_ = weight_list[i];
    auto w_shape = weight_.GetType().GetShape();
    printVector(w_shape, "weight" + std::to_string(i) + ".shape");
  }

  // each slice: (batch_size, input_size)
  std::vector<builder::Op> X_slices = get_slice_ops(X, 0, seq_len, true);
  std::vector<builder::Op> W_T_forward_layers;
  std::vector<builder::Op> R_T_forward_layers;
  std::vector<builder::Op> B_forward_layers;
  std::vector<builder::Op> W_T_backward_layers;
  std::vector<builder::Op> R_T_backward_layers;
  std::vector<builder::Op> B_backward_layers;
  split_weight_list(weight_list,
                    &W_T_forward_layers,
                    &R_T_forward_layers,
                    &B_forward_layers,
                    &W_T_backward_layers,
                    &R_T_backward_layers,
                    &B_backward_layers,
                    num_directions,
                    num_layers);

  /* e.g. num_layers = 2, num_directions = 2
     init_h_slices[0]: layer 0, forward
     init_h_slices[1]: layer 0, backward
     init_h_slices[2]: layer 1, forward
     init_h_slices[3]: layer 1, backward
     each slice has shape (batch_size, hidden_size) */
  std::vector<builder::Op> init_h_slices =
      get_slice_ops(init_h, 0, num_layers * num_directions, true);
  std::vector<builder::Op> init_c_slices =
      get_slice_ops(init_c, 0, num_layers * num_directions, true);

  // do lstm
  builder::Op out, state_h, state_c;
  auto out_type = builder::Type(
      {seq_len, batch_size, hidden_size * num_directions}, input_dtype);
  auto state_type = builder::Type(
      {num_layers * num_directions, batch_size, hidden_size}, input_dtype);
  if (num_directions == 1) {
    std::vector<std::vector<builder::Op>> h_list_layers(num_layers);
    std::vector<builder::Op> h_o_layers(num_layers);
    std::vector<builder::Op> c_o_layers(num_layers);
    for (int64_t k = 0; k < num_layers; k++) {
      go_lstm_one_direction(k == 0 ? X_slices : h_list_layers[k - 1],
                            W_T_forward_layers[k],
                            R_T_forward_layers[k],
                            B_forward_layers[k],
                            init_h_slices[k],
                            init_c_slices[k],
                            batch_size,
                            hidden_size,
                            seq_len,
                            input_dtype,
                            &h_list_layers[k],
                            &h_o_layers[k],
                            &c_o_layers[k]);
    }
    out =
        concatenate_with_unsqueeze(h_list_layers[num_layers - 1], 0, out_type);
    state_h = concatenate_with_unsqueeze(h_o_layers, 0, state_type);
    state_c = concatenate_with_unsqueeze(c_o_layers, 0, state_type);
  } else {
    std::vector<std::vector<builder::Op>> h_list_forward_layers(num_layers);
    std::vector<std::vector<builder::Op>> h_list_backward_layers(num_layers);
    std::vector<builder::Op> h_o_forward_layers(num_layers);
    std::vector<builder::Op> h_o_backward_layers(num_layers);
    std::vector<builder::Op> c_o_forward_layers(num_layers);
    std::vector<builder::Op> c_o_backward_layers(num_layers);
    builder::Type input_step_type({batch_size, 2 * hidden_size}, input_dtype);

    for (int64_t k = 0; k < num_layers; k++) {
      auto input_seq =
          (k == 0) ? X_slices
                   : concatenate_two_vectors(h_list_forward_layers[k - 1],
                                             h_list_backward_layers[k - 1],
                                             1,
                                             input_step_type);
      go_lstm_one_direction(input_seq,
                            W_T_forward_layers[k],
                            R_T_forward_layers[k],
                            B_forward_layers[k],
                            init_h_slices[k * 2],
                            init_c_slices[k * 2],
                            batch_size,
                            hidden_size,
                            seq_len,
                            input_dtype,
                            &h_list_forward_layers[k],
                            &h_o_forward_layers[k],
                            &c_o_forward_layers[k]);
      std::reverse(input_seq.begin(), input_seq.end());
      go_lstm_one_direction(input_seq,
                            W_T_backward_layers[k],
                            R_T_backward_layers[k],
                            B_backward_layers[k],
                            init_h_slices[k * 2 + 1],
                            init_c_slices[k * 2 + 1],
                            batch_size,
                            hidden_size,
                            seq_len,
                            input_dtype,
                            &h_list_backward_layers[k],
                            &h_o_backward_layers[k],
                            &c_o_backward_layers[k]);
      std::reverse(input_seq.begin(), input_seq.end());
      std::reverse(h_list_backward_layers[k].begin(),
                   h_list_backward_layers[k].end());
    }
    auto out_bidirec =
        concatenate_two_vectors(h_list_forward_layers[num_layers - 1],
                                h_list_backward_layers[num_layers - 1],
                                1,
                                input_step_type);
    out = concatenate_with_unsqueeze(out_bidirec, 0, out_type);
    // size: 2 * num_layers, each shape: (batch_size, hidden_size)
    std::vector<builder::Op> h_o_merge, c_o_merge;
    for (int64_t k = 0; k < num_layers; k++) {
      h_o_merge.push_back(h_o_forward_layers[k]);
      h_o_merge.push_back(h_o_backward_layers[k]);
      c_o_merge.push_back(c_o_forward_layers[k]);
      c_o_merge.push_back(c_o_backward_layers[k]);
    }
    state_h = concatenate_with_unsqueeze(h_o_merge, 0, state_type);
    state_c = concatenate_with_unsqueeze(c_o_merge, 0, state_type);
  }

  // set output
  auto output_name_map = op->Outputs();
  std::cout << "op->outputs.size = " << output_name_map.size() << std::endl;
  for (auto kv : output_name_map) {
    std::cout << kv.first << " : " << kv.second.size() << " : ";
    for (auto x : kv.second) {
      std::cout << x << " ";
    }
    std::cout << std::endl;
  }
  std::vector<std::string> output_names{"Out", "State"};
  std::string output_names_attr(output_name_map[output_names[0]][0]);
  std::vector<builder::Op> outputs{out, state_h, state_c};
  // todo: how to calculate these two?
  auto set_placeholder = [&](std::string output_name) {
    if (output_name_map.count(output_name) != 0) {
      std::vector<float> tmp = {0};
      auto place_holder = builder::Const(gcu_builder,
                                         static_cast<void *>(tmp.data()),
                                         GcuType({}, input_dtype));
      output_names.emplace_back(output_name);
      outputs.emplace_back(place_holder);
    }
  };
  set_placeholder("Reserve");
  set_placeholder("DropoutState");

  for (size_t i = 1; i < output_names.size(); ++i) {
    auto var_name_count = output_name_map[output_names[i]].size();
    for (size_t j = 0; j < var_name_count; ++j) {
      output_names_attr += ";" + output_name_map[output_names[i]][j];
    }
  }
  std::vector<builder::PrimitiveType> tuple_dtype;
  std::vector<std::vector<int64_t>> tuple_shape;
  for (uint i = 0; i < outputs.size(); i++) {
    tuple_shape.push_back(outputs[i].GetType().GetShape());
    tuple_dtype.push_back(outputs[i].GetType().GetPrimitiveType());
  }
  GcuType outputs_type(tuple_shape, tuple_dtype);
  auto result = builder::Tuple(outputs, outputs_type);
  result.SetAttribute(kAttrOpOutVarName,
                      builder::Attribute(output_names_attr.c_str()));
  return std::make_shared<GcuOp>(result);
}

IMPLEMT_EQUIVALENCE_TRANS_FUNC(
    gcu_builder, node, map_inputs, running_mode, RNNGradEquivalenceTrans) {
  std::cout << "<<< hello rnn_grad! >>>" << std::endl;
  // builder::Op x = *(map_inputs["X"].at(0));
  // builder::Op out = *(map_inputs["Out@GRAD"].at(0));
  // int64_t dim = static_cast<int64_t>(x.GetType().GetShape().size());
  // std::vector<int64_t> axis;
  // for (int64_t i = 0; i < dim; i++) {
  //     axis.emplace_back(i);
  // }
  // auto output_size = out.GetType().GetSize();
  // if (output_size == 0) {
  //     output_size = 1;
  // }
  // auto input_size = x.GetType().GetSize();
  // if (input_size == 0) {
  //     input_size = 1;
  // }
  // float reduced_size = static_cast<float>(input_size / output_size);
  // float reciprocal = 1.0 / reduced_size;
  // builder::Op derivative = builder::FullLike(out, reciprocal);
  // auto grad = out * derivative;
  // auto output_rank = out.GetType().GetRank();
  // std::vector<int64_t> broadcast_dims;
  // int iter = 0;
  // for (int64_t i = 0; i < output_rank; ++i) {
  //     if (i == axis[iter]) {
  //         ++iter;
  //     } else {
  //         broadcast_dims.emplace_back(i);
  //     }
  // }
  // auto result = builder::BroadcastInDim(grad, broadcast_dims,
  //     x.GetType());
  builder::Op result;
  return std::make_shared<GcuOp>(result);
}

EQUIVALENCE_TRANS_FUNC_REG(kRNN, INSENSITIVE, RNNEquivalenceTrans);
EQUIVALENCE_TRANS_FUNC_REG(kRNNGrad, INSENSITIVE, RNNGradEquivalenceTrans);

}  // namespace gcu
}  // namespace platform
}  // namespace paddle
