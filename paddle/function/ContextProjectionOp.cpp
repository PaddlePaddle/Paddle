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

#include "ContextProjectionOp.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"

namespace paddle {
/**
 * Context Projection Forward with CPU Matrix Device.
 *
 */
template <>
void ContextProjectionForward<DEVICE_TYPE_CPU>(CpuMatrix& out_mat,
                                               const CpuMatrix& input_mat,
                                               const CpuMatrix& weight_mat,
                                               const CpuIVector& seq_vec,
                                               size_t context_length,
                                               int context_start,
                                               size_t begin_pad) {
  const int* starts = seq_vec.getData();
  const size_t num_sequences = seq_vec.getSize() - 1;
  for (size_t i = 0; i < num_sequences; ++i) {
    for (size_t j = 0; j < context_length; ++j) {
      int begin = starts[i] + context_start + j;
      int end = starts[i + 1] + context_start + j;
      int dst_begin = starts[i];
      int dst_end = starts[i + 1];
      if (begin < starts[i]) {
        int64_t pad_size =
            std::min(starts[i] - begin, starts[i + 1] - starts[i]);
        MatrixPtr mat = out_mat.subMatrix(starts[i], pad_size);
        if (weight_mat) {
          MatrixPtr sub =
              const_cast<CpuMatrix&>(weight_mat).subMatrix(j, pad_size);
          mat->addAtOffset(*sub, j * input_mat.getWidth());
        }
        dst_begin = starts[i] + pad_size;
        begin = starts[i];
      }
      if (end > starts[i + 1]) {
        int64_t pad_size =
            std::min(end - starts[i + 1], starts[i + 1] - starts[i]);
        MatrixPtr mat = out_mat.subMatrix(starts[i + 1] - pad_size, pad_size);
        if (weight_mat) {
          MatrixPtr sub =
              const_cast<CpuMatrix&>(weight_mat)
                  .subMatrix(begin_pad + context_start + j - pad_size,
                             pad_size);
          mat->addAtOffset(*sub, j * input_mat.getWidth());
        }
        dst_end = starts[i + 1] - pad_size;
        end = starts[i + 1];
      }
      if (end <= begin) continue;
      MatrixPtr src =
          const_cast<CpuMatrix&>(input_mat).subMatrix(begin, end - begin);
      MatrixPtr dst = out_mat.subMatrix(dst_begin, dst_end - dst_begin);
      dst->addAtOffset(*src, j * input_mat.getWidth());
    }
  }
}

/**
 * Paddle Function for Context Projection Forward.
 * Calculate the output layer value sequence after context projection.
 *
 * What is Context Projection for a sequence?
 * For example, assumed input (x) has 4 words and the dimension of each word
 * representation is 2. If we use zero to pad instead of learned weight to pad,
 * and the context_lenth is 3, the output (y) is:
 *
 * @code
 *  x = [a1, a2;
 *       b1, b2;
 *       c1, c2;
 *       d1, d2]
 *  y = [0,  0,  a1, a2, b1, b2;
 *       a1, a2, b1, b2, c1, c2;
 *       b1, b2, c1, c2, d1, d2;
 *       c1, c2, d1, d2, 0,  0]
 * @endcode
 *
 * \param outputs[0].matrix   output layer value, n * (d * l)
 * \param outputs[0].vector   start position sequence, n * 1
 * \param inputs[0].matrix    input layer value, n * d
 * \param inputs[0].vector    start position sequence, n * 1
 * \param inputs[1].matrix    input layer weight, pad * d
 */
template <DeviceType Device>
class ContextProjectionForwardFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    context_length_ = config.get<size_t>("context_length");
    context_start_ = config.get<int>("context_start");
    begin_pad_ = config.get<size_t>("begin_pad");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK(1UL == inputs.size() || 2UL == inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK(inputs[0].isSequenceArg() && outputs[0].isSequenceArg())
        << "SequenceArg required here";
    const auto val_seqs = dynamic_cast<const SequenceArg&>(inputs[0]);
    auto out_seq = dynamic_cast<const SequenceArg&>(outputs[0]);

    CHECK(out_seq.data() && val_seqs.data() && val_seqs.getSequenceId().data());
    CHECK_EQ(out_seq.shape().ndims(), 2UL);
    CHECK_EQ(val_seqs.shape().ndims(), 2UL);
    /// dim of output = dim of input * context_length
    CHECK_EQ(out_seq.shape()[1], val_seqs.shape()[1] * context_length_);
    /// input and output has the same batch_size
    CHECK_EQ(val_seqs.shape()[0], out_seq.shape()[0]);
    if (2UL == inputs.size()) {
      CHECK_EQ(inputs[1].shape().ndims(), 2UL);
      /// dim of input == dim of weight
      CHECK_EQ(val_seqs.shape()[1], inputs[1].shape()[1]);
    }

    CHECK_EQ(out_seq.getArgType(), ADD_TO);
    auto out_mat = out_seq.matrix<Device>();
    const auto in_mat = val_seqs.matrix<Device>();
    const auto w_mat =
        (2UL == inputs.size() && inputs[1].data())
            ? inputs[1].matrix<Device>()
            : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);
    const auto seq_vec = val_seqs.getSequenceId().vector<int, Device>();

    ContextProjectionForward<Device>(out_mat,
                                     in_mat,
                                     w_mat,
                                     seq_vec,
                                     context_length_,
                                     context_start_,
                                     begin_pad_);
  }

 private:
  size_t context_length_;
  int context_start_;
  size_t begin_pad_;
};

/**
 * Context Projection Backward with CPU Matrix Device.
 *
 */
template <>
void ContextProjectionBackward<DEVICE_TYPE_CPU>(const CpuMatrix& out_grad_mat,
                                                CpuMatrix& in_grad_mat,
                                                CpuMatrix& w_grad_mat,
                                                const CpuIVector& seq_vec,
                                                size_t context_length,
                                                int context_start,
                                                size_t begin_pad,
                                                bool is_padding,
                                                size_t total_pad) {
  size_t input_dim = in_grad_mat ? in_grad_mat.getWidth()
                                 : w_grad_mat ? w_grad_mat.getWidth() : 0;
  const int* starts = seq_vec.getData();
  size_t num_sequences = seq_vec.getSize() - 1;
  for (size_t i = 0; i < num_sequences; ++i) {
    for (size_t j = 0; j < context_length; ++j) {
      int begin = starts[i] + context_start + j;
      int end = starts[i + 1] + context_start + j;
      int dst_begin = starts[i];
      int dst_end = starts[i + 1];
      if (begin < starts[i]) {
        int64_t pad_size =
            std::min(starts[i] - begin, starts[i + 1] - starts[i]);
        if (is_padding && w_grad_mat) {
          MatrixPtr mat = const_cast<CpuMatrix&>(out_grad_mat)
                              .subMatrix(starts[i], pad_size);
          MatrixPtr sub = w_grad_mat.subMatrix(j, pad_size);
          sub->addAtOffset(*mat, j * input_dim);
        }
        dst_begin = starts[i] + pad_size;
        begin = starts[i];
      }
      if (end > starts[i + 1]) {
        int64_t pad_size =
            std::min(end - starts[i + 1], starts[i + 1] - starts[i]);
        if (is_padding && w_grad_mat) {
          MatrixPtr mat = const_cast<CpuMatrix&>(out_grad_mat)
                              .subMatrix(starts[i + 1] - pad_size, pad_size);
          MatrixPtr sub = w_grad_mat.subMatrix(
              begin_pad + context_start + j - pad_size, pad_size);
          sub->addAtOffset(*mat, j * input_dim);
        }
        dst_end = starts[i + 1] - pad_size;
        end = starts[i + 1];
      }
      if (end <= begin) continue;
      if (!in_grad_mat) continue;
      MatrixPtr src = in_grad_mat.subMatrix(begin, end - begin);
      MatrixPtr dst = const_cast<CpuMatrix&>(out_grad_mat)
                          .subMatrix(dst_begin, dst_end - dst_begin);
      src->addAtOffset(*dst, j * input_dim);
    }
  }
}

/**
 * Context Projection Backward Function.
 * Update the weight gradient and input layer gradient with backprop
 *
 * \param inputs[0].matrix          output layer grad, n * (d * l)
 * \param inputs[0].vector          start position sequence, n * 1
 * \param outputs[0].matrix         input layer grad, n * d
 * \param outputs[0].vector         start position sequence, n * 1
 * \param outputs[1]                weight grad, pad * d
 */
template <DeviceType Device>
class ContextProjectionBackwardFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    context_length_ = config.get<size_t>("context_length");
    context_start_ = config.get<int>("context_start");
    begin_pad_ = config.get<size_t>("begin_pad");
    is_padding_ = config.get<bool>("is_padding");
    total_pad_ = config.get<size_t>("total_pad");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK(1UL == outputs.size() || 2UL == outputs.size());
    CHECK(inputs[0].isSequenceArg() && outputs[0].isSequenceArg())
        << "SequenceArg required here";
    const auto in_seq = dynamic_cast<const SequenceArg&>(inputs[0]);
    auto out_seq = dynamic_cast<const SequenceArg&>(outputs[0]);
    CHECK(in_seq.data() && in_seq.getSequenceId().data());
    CHECK_EQ(in_seq.shape().ndims(), 2UL);
    CHECK_EQ(out_seq.shape().ndims(), 2UL);
    CHECK_EQ(out_seq.getSequenceId().shape().ndims(), 1UL);

    /// input and output grad has the same batch_size
    CHECK_EQ(out_seq.shape()[0], in_seq.shape()[0]);
    /// dim of output grad = dim of input grad * context_length
    CHECK_EQ(in_seq.shape()[1], out_seq.shape()[1] * context_length_);
    CHECK_EQ(out_seq.getArgType(), ADD_TO);

    if (2UL == outputs.size()) {
      CHECK_EQ(outputs[1].shape().ndims(), 2UL);
      /// dim of input grad == dim of weight
      CHECK_EQ(out_seq.shape()[1], outputs[1].shape()[1]);
      CHECK_EQ(outputs[1].getArgType(), ADD_TO);
    }

    const auto seq_vec = in_seq.getSequenceId().vector<int, Device>();
    const auto out_grad_mat = in_seq.matrix<Device>();
    auto in_grad_mat =
        !out_seq.data() ? typename Tensor<real, Device>::Matrix(nullptr, 0, 0)
                        : out_seq.matrix<Device>();
    auto w_grad_mat =
        (2UL == outputs.size() && outputs[1].data())
            ? outputs[1].matrix<Device>()
            : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);

    ContextProjectionBackward<Device>(out_grad_mat,
                                      in_grad_mat,
                                      w_grad_mat,
                                      seq_vec,
                                      context_length_,
                                      context_start_,
                                      begin_pad_,
                                      is_padding_,
                                      total_pad_);
  }

 private:
  size_t context_length_;
  int context_start_;
  size_t begin_pad_;
  bool is_padding_;
  size_t total_pad_;
};

/**
 * Context Projection Backward Data Function
 * Update input layer grad
 * input:  sequence of output layer grad
 * output: sequence of input layer grad
 *
 * \param outputs[0].matrix              input layer grad, n * d
 * \param outputs[0].vector              start position sequence, n * 1
 * \param inputs[0].matrix               output layer grad, n * (d * l)
 * \param inputs[0].vector               start positon sequence, n * 1
 */
template <DeviceType Device>
class ContextProjectionBackwardDataFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    context_length_ = config.get<size_t>("context_length");
    context_start_ = config.get<int>("context_start");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK(inputs[0].isSequenceArg() && outputs[0].isSequenceArg())
        << "SequenceArg required here";
    const auto in_seq = dynamic_cast<const SequenceArg&>(inputs[0]);
    const auto out_seq = dynamic_cast<const SequenceArg&>(outputs[0]);

    CHECK(in_seq.data() && out_seq.data() && in_seq.getSequenceId().data());
    CHECK_EQ(out_seq.shape().ndims(), 2UL);
    CHECK_EQ(in_seq.shape().ndims(), 2UL);
    CHECK_EQ(in_seq.getSequenceId().shape().ndims(), 1UL);
    /// output layer grad dim == input layer grad dim * context_length_
    CHECK_EQ(in_seq.shape().ndims(), out_seq.shape().ndims() * context_length_);
    /// input and output has the same batch_size
    CHECK_EQ(in_seq.shape()[0], out_seq.shape()[0]);
    CHECK_EQ(outputs[0].getArgType(), ASSIGN_TO);

    const auto out_grad_mat = in_seq.matrix<Device>();
    const auto seq_vec = in_seq.getSequenceId().vector<int, Device>();
    auto in_grad_mat = out_seq.matrix<Device>();

    ContextProjectionBackwardData<Device>(
        out_grad_mat, in_grad_mat, seq_vec, context_length_, context_start_);
  }

 private:
  size_t context_length_;
  int context_start_;
};

/**
 * Context Projection Backward Weight Function
 * Update weight grad by backprop
 * input:  sequence of output layer grad
 * output: weight grad
 *
 * \param outputs[0]                   weight grad, pad * d
 * \param inputs[0].matrix             output layer grad, n * (d * l)
 * \param inputs[0].vecotr             start positon sequence, n * 1
 */
template <DeviceType Device>
class ContextProjectionBackwardWeightFunc : public FunctionBase {
 public:
  void init(const FuncConfig& config) override {
    context_length_ = config.get<size_t>("context_length");
    context_start_ = config.get<int>("context_start");
    begin_pad_ = config.get<size_t>("begin_pad");
    total_pad_ = config.get<size_t>("total_pad");
  }

  void calc(const BufferArgs& inputs, const BufferArgs& outputs) override {
    CHECK_EQ(1UL, inputs.size());
    CHECK_EQ(1UL, outputs.size());
    CHECK(inputs[0].isSequenceArg()) << "SequenceArg required here";
    const auto in_seq = dynamic_cast<const SequenceArg&>(inputs[0]);
    CHECK(in_seq.data() && in_seq.getSequenceId().data() && outputs[0].data());
    CHECK_EQ(outputs[0].shape().ndims(), 2UL);
    CHECK_EQ(in_seq.shape().ndims(), 2UL);
    CHECK_EQ(in_seq.getSequenceId().shape().ndims(), 1UL);
    CHECK_EQ(in_seq.shape()[0], outputs[0].shape()[0]);
    /// output layer grad dim == weight dim * context_length_
    CHECK_EQ(in_seq.shape()[1], outputs[0].shape()[1] * context_length_);
    CHECK_EQ(outputs[0].getArgType(), ADD_TO);

    const auto seq_vec = in_seq.getSequenceId().vector<int, Device>();
    const auto out_grad_mat = in_seq.matrix<Device>();
    auto w_grad_mat = outputs[0].matrix<Device>();
    ContextProjectionBackwardWeight<Device>(out_grad_mat,
                                            w_grad_mat,
                                            seq_vec,
                                            context_length_,
                                            context_start_,
                                            total_pad_,
                                            begin_pad_);
  }

 private:
  size_t context_length_;
  int context_start_;
  size_t begin_pad_;
  size_t total_pad_;
};

REGISTER_TYPED_FUNC(ContextProjectionForward,
                    CPU,
                    ContextProjectionForwardFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackward,
                    CPU,
                    ContextProjectionBackwardFunc);
#ifdef PADDLE_WITH_CUDA
REGISTER_TYPED_FUNC(ContextProjectionForward,
                    GPU,
                    ContextProjectionForwardFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackward,
                    GPU,
                    ContextProjectionBackwardFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackwardData,
                    GPU,
                    ContextProjectionBackwardDataFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackwardWeight,
                    GPU,
                    ContextProjectionBackwardWeightFunc);
#endif
}  // namespace paddle
