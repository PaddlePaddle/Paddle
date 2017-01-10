/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
 * \param inputs[0] input value.
 * \param inputs[1] input weight.
 * \param inputs[2] input sequence.
 * \param outputs[0] output value.
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
    CHECK_EQ((size_t)3, inputs.size());
    CHECK_EQ((size_t)1, outputs.size());

    CHECK(outputs[0].data() && inputs[0].data() && inputs[2].data());
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[2].shape().ndims(), (size_t)1);
    /// dim of output = dim of input * context_length
    CHECK_EQ(outputs[0].shape()[1], inputs[0].shape()[1] * context_length_);
    /// dim of input == dim of weight
    CHECK_EQ(inputs[0].shape()[1], inputs[1].shape()[1]);
    /// input and output has the same batch_size
    CHECK_EQ(inputs[0].shape()[0], outputs[0].shape()[0]);

    CHECK_EQ(outputs[0].getArgType(), ADD_TO);
    auto out_mat = outputs[0].matrix<Device>();
    auto in_mat = inputs[0].matrix<Device>();
    auto w_mat = !inputs[1].data()
                     ? typename Tensor<real, Device>::Matrix(nullptr, 0, 0)
                     : inputs[1].matrix<Device>();
    auto seq_vec = inputs[2].vector<int, Device>();
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

template <>
void ContextProjectionBackward<DEVICE_TYPE_CPU>(CpuMatrix& out_grad_mat,
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
          MatrixPtr mat = out_grad_mat.subMatrix(starts[i], pad_size);
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
          MatrixPtr mat =
              out_grad_mat.subMatrix(starts[i + 1] - pad_size, pad_size);
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
      MatrixPtr dst = out_grad_mat.subMatrix(dst_begin, dst_end - dst_begin);
      src->addAtOffset(*dst, j * input_dim);
    }
  }
}

/**
 * \param inputs[0] input grad.
 * \param inputs[1] weight grad.
 * \param inputs[2] input sequence.
 * \param outputs[0] output value.
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
    CHECK_EQ((size_t)3, inputs.size());
    CHECK_EQ((size_t)1, outputs.size());

    CHECK(outputs[0].data() && inputs[2].data());
    CHECK_EQ(outputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[0].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[1].shape().ndims(), (size_t)2);
    CHECK_EQ(inputs[2].shape().ndims(), (size_t)1);

    /// dim of input == dim of weight
    CHECK_EQ(inputs[0].shape()[1], inputs[1].shape()[1]);
    /// input and output has the same batch_size
    CHECK_EQ(inputs[0].shape()[0], outputs[0].shape()[0]);
    /// dim of output = dim of input * context_length
    CHECK_EQ(outputs[0].shape()[1], inputs[0].shape()[1] * context_length_);

    CHECK_EQ(outputs[0].getArgType(), ADD_TO);

    auto out_grad_mat = outputs[0].matrix<Device>();
    auto in_grad_mat =
        !inputs[0].data() ? typename Tensor<real, Device>::Matrix(nullptr, 0, 0)
                          : inputs[0].matrix<Device>();
    auto w_grad_mat = !inputs[1].data()
                          ? typename Tensor<real, Device>::Matrix(nullptr, 0, 0)
                          : inputs[1].matrix<Device>();
    auto seq_vec = inputs[2].vector<int, Device>();
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

#if 0
/**
 * \param inputs[0] input grad.
 * \param inputs[1] input sequence.
 * \param outputs[0] output grad.
 */
template <DeviceType Device>
class ContextProjectionBackwardDataFunc : public FunctionBase {
public:
  void init(const FuncConfig& config) override {
    context_length_ = config.get<size_t>("context_length");
    context_start_ = config.get<int>("context_start");
  }

  void calc(const Arguments& inputs,
            const Arguments& outputs,
            const Arguments& inouts) override {
    CHECK_EQ(2, static_cast<int>(inputs.size()));
    CHECK_EQ(1, static_cast<int>(outputs.size()));
    CHECK_EQ(0, static_cast<int>(inouts.size()));
    CHECK(inputs[0].getData() && outputs[0].getData() && inputs[1].getData());
    CHECK_EQ(static_cast<int>(outputs[0].dims_.size()), 2);
    CHECK_EQ(static_cast<int>(inputs[0].dims_.size()), 2);
    CHECK_EQ(static_cast<int>(inputs[1].dims_.size()), 1);
    CHECK_EQ(outputs[0].dims_[1], inputs[0].dims_[1] * context_length_);
    /// input and output has the same batch_size
    CHECK_EQ(inputs[0].dims_[0], outputs[0].dims_[0]);

    auto out_grad_mat = std::make_shared<typename MatrixT<Device>::type>(
        outputs[0].getData(), outputs[0].dims_[0], outputs[0].dims_[1]);
    const auto in_grad_mat = std::make_shared<typename MatrixT<Device>::type>(
        inputs[0].getData(), inputs[0].dims_[0], inputs[0].dims_[1]);
    typename SequenceT<Device>::type seq_vec(
        inputs[1].dims_[0], reinterpret_cast<int*>(inputs[1].getData()));

    ContextProjectionBackwardData<Device>(out_grad_mat.get(),
                                          in_grad_mat.get(),
                                          seq_vec,
                                          context_length_,
                                          context_start_);
  }

private:
  size_t context_length_;
  int context_start_;
};

/**
 * \param inputs[0] weight grad.
 * \param inputs[1] input sequence.
 * \param outputs[0] output grad.
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

  void calc(const Arguments& inputs,
            const Arguments& outputs,
            const Arguments& inouts) override {
    CHECK_EQ(2, static_cast<int>(inputs.size()));
    CHECK_EQ(1, static_cast<int>(outputs.size()));
    CHECK_EQ(0, static_cast<int>(inouts.size()));

    CHECK(inputs[0].getData() && outputs[0].getData() && inputs[1].getData());
    CHECK_EQ(static_cast<int>(outputs[0].dims_.size()), 2);
    CHECK_EQ(static_cast<int>(inputs[0].dims_.size()), 2);
    CHECK_EQ(static_cast<int>(inputs[1].dims_.size()), 1);
    CHECK_EQ(outputs[0].dims_[1], inputs[0].dims_[1] * context_length_);

    auto out_grad_mat = std::make_shared<typename MatrixT<Device>::type>(
        outputs[0].getData(), outputs[0].dims_[0], outputs[0].dims_[1]);
    auto w_grad_mat = std::make_shared<typename MatrixT<Device>::type>(
        inputs[0].getData(), inputs[0].dims_[0], inputs[0].dims_[1]);
    typename SequenceT<Device>::type seq_vec(
        inputs[1].dims_[0], reinterpret_cast<int*>(inputs[1].getData()));

    ContextProjectionBackwardWeight<Device>(out_grad_mat.get(),
                                            w_grad_mat.get(),
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
#endif

REGISTER_TYPED_FUNC(ContextProjectionForward,
                    CPU,
                    ContextProjectionForwardFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackward,
                    CPU,
                    ContextProjectionBackwardFunc);
#ifndef PADDLE_ONLY_CPU
REGISTER_TYPED_FUNC(ContextProjectionForward,
                    GPU,
                    ContextProjectionForwardFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackward,
                    GPU,
                    ContextProjectionBackwardFunc);
#if 0
REGISTER_TYPED_FUNC(ContextProjectionBackwardData,
                    GPU,
                    ContextProjectionBackwardDataFunc);
REGISTER_TYPED_FUNC(ContextProjectionBackwardWeight,
                    GPU,
                    ContextProjectionBackwardWeightFunc);
#endif
#endif
}  // namespace paddle
