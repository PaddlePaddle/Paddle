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
#include "Register.h"
#include "paddle/math/Matrix.h"
#include "paddle/math/Vector.h"
#include "paddle/topology/Attribute.h"
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

struct ContextProjectionAttribute : public topology::Attribute {
  size_t contextLength;
  int contextStart;
  size_t beginPad;

  REGISTER_FUNC_ATTRIBUTE() {
    regAttr(&ContextProjectionAttribute::contextLength,
            "context_length",
            "The length of context projection window")
        .mustSet()
        .largerThan(1);
    regAttr(&ContextProjectionAttribute::contextStart,
            "context_start",
            "The start position of context projection")
        .defaultValue(0);
    regAttr(&ContextProjectionAttribute::beginPad,
            "begin_pad",
            "number of extra timesteps added at the beginning")
        .defaultValue(0)
        .largerThan(0);
  }
};

template <DeviceType Device>
Error contextProjFwd(const BufferArgs& in,
                     const BufferArgs& out,
                     const ContextProjectionAttribute& attr) {
  const auto val_seqs = dynamic_cast<const SequenceArg&>(in[0]);
  auto out_seq = dynamic_cast<const SequenceArg&>(out[0]);
  auto out_mat = out_seq.matrix<Device>();
  const auto in_mat = val_seqs.matrix<Device>();
  const auto w_mat = (2UL == in.size() && in[1].data())
                         ? in[1].matrix<Device>()
                         : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);
  const auto seq_vec = val_seqs.getSequenceId().vector<int, Device>();

  ContextProjectionForward<Device>(out_mat,
                                   in_mat,
                                   w_mat,
                                   seq_vec,
                                   attr.contextLength,
                                   attr.contextStart,
                                   attr.beginPad);
  return Error();
}

BEGIN_REGISTER_FUNCTION(ctxProjFwd, contextProjFwd, ContextProjectionAttribute)
setDescription(R"DOC(Paddle Function for Context Projection Forward.
Calculate the output layer value sequence after context projection.

What is Context Projection for a sequence?
For example, assumed input (x) has 4 words and the dimension of each word
representation is 2. If we use zero to pad instead of learned weight to
pad, and the context_lenth is 3, the output (y) is:

@code
 x = [a1, a2;
      b1, b2;
      c1, c2;
      d1, d2]
 y = [0,  0,  a1, a2, b1, b2;
      a1, a2, b1, b2, c1, c2;
      b1, b2, c1, c2, d1, d2;
      c1, c2, d1, d2, 0,  0]
@endcode
)DOC");
addTensor<INPUT>(2,
                 UNSPECIFIED,
                 {topology::DataType::DENSE,
                  topology::DataType::SPARSE,
                  topology::DataType::SPARSE_INTEGER,
                  topology::DataType::INTEGER},
                 {topology::SequenceType::SEQUENCE,
                  topology::SequenceType::NESTED_SEQUENCE});
addTensor<INPUT>(2)->setOptional();
addTensor<OUTPUT>(2,
                  ADD_TO,
                  {topology::DataType::DENSE},
                  {topology::SequenceType::NESTED_SEQUENCE,
                   topology::SequenceType::SEQUENCE});
setShapeInferer<ContextProjectionAttribute>(
    [](std::vector<topology::TensorPtr>& ins,
       std::vector<topology::TensorPtr>& outs,
       const ContextProjectionAttribute& attrs) -> Error {
      outs[0]->setDataType(ins[0]->dataType());
      outs[0]->setSequenceType(ins[0]->sequenceType());
      auto& shape = ins[0]->shape();
      outs[0]->setShape({shape[0], shape[1] * attrs.contextLength});
      if (ins.size() == 2) {
        if (ins[0]->shape()[1] != ins[1]->shape()[1]) {
          return Error("Context Project's Dim of input == dim of weight");
        }
      }
      return Error();
    });

END_REGISTER_FUNCTION(ctxProjFwd)

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

struct ContextProjBackwardAttribute : public ContextProjectionAttribute {
  bool isPadding;
  size_t totalPad;

  REGISTER_FUNC_ATTRIBUTE_EXTENDS(ContextProjectionAttribute) {
    regAttr<bool>(&ContextProjBackwardAttribute::isPadding,
                  "is_padding",
                  "Padding the context projection or not")
        .mustSet();

    regAttr<size_t>(&ContextProjBackwardAttribute::totalPad,
                    "total_pad",
                    "total padding length")
        .defaultValue(0)
        .largerThan(0);
  }
};

template <DeviceType Device>
Error ContextBwd(const BufferArgs& in,
                 const BufferArgs& out,
                 const ContextProjBackwardAttribute& attrs) {
  const auto in_seq = dynamic_cast<const SequenceArg&>(in[0]);
  auto out_seq = dynamic_cast<const SequenceArg&>(out[0]);
  const auto seq_vec = in_seq.getSequenceId().vector<int, Device>();
  const auto out_grad_mat = in_seq.matrix<Device>();
  auto in_grad_mat = !out_seq.data()
                         ? typename Tensor<real, Device>::Matrix(nullptr, 0, 0)
                         : out_seq.matrix<Device>();
  auto w_grad_mat = (2UL == out.size() && out[1].data())
                        ? out[1].matrix<Device>()
                        : typename Tensor<real, Device>::Matrix(nullptr, 0, 0);

  ContextProjectionBackward<Device>(out_grad_mat,
                                    in_grad_mat,
                                    w_grad_mat,
                                    seq_vec,
                                    attrs.contextLength,
                                    attrs.contextStart,
                                    attrs.beginPad,
                                    attrs.isPadding,
                                    attrs.totalPad);
  return Error();
}

BEGIN_REGISTER_FUNCTION(ctxProjBwd, ContextBwd, ContextProjBackwardAttribute)
setDescription(R"DOC(Context Projection Backward Function.
Update the weight gradient and input layer gradient with backprop
)DOC");
addTensor<INPUT>(2,
                 -1,
                 {topology::DataType::DENSE,
                  topology::DataType::SPARSE,
                  topology::DataType::SPARSE_INTEGER},
                 {topology::SequenceType::NESTED_SEQUENCE,
                  topology::SequenceType::SEQUENCE});

addTensor<OUTPUT>(2,
                  ADD_TO,
                  {topology::DataType::DENSE},
                  {topology::SequenceType::NESTED_SEQUENCE,
                   topology::SequenceType::SEQUENCE});

addTensor<OUTPUT>(2,
                  ADD_TO,
                  {topology::DataType::DENSE},
                  {topology::SequenceType::NO_SEQUENCE})
    ->setOptional();

setShapeInferer<ContextProjBackwardAttribute>(
    [](std::vector<topology::TensorPtr>& ins,
       std::vector<topology::TensorPtr>& outs,
       const ContextProjBackwardAttribute& attrs) -> Error {
      outs[0]->setDataType(ins[0]->dataType());
      outs[0]->setSequenceType(ins[0]->sequenceType());
      outs[0]->setShape(
          {ins[0]->shape()[0], ins[0]->shape()[1] / attrs.contextLength});

      if (outs.size() == 2) {
        outs[1]->setDataType(topology::DataType::DENSE);
        outs[1]->setSequenceType(topology::SequenceType::NO_SEQUENCE);
        outs[1]->setShape(
            {topology::meta::kTensorShape_NOT_SPECIFIC, outs[0]->shape()[1]});
      }
      return Error();
    });

END_REGISTER_FUNCTION(ctxProjBwd)
}  // namespace paddle
