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

#include "paddle/fluid/operators/beam_search_decode_op.h"

#include <string>

#include "paddle/fluid/framework/convert_utils.h"
#include "paddle/fluid/platform/device_context.h"

namespace paddle {
namespace framework {
class InferShapeContext;
class OpDesc;
class Scope;
template <typename T>
class EmptyGradOpMaker;
}  // namespace framework
namespace imperative {
class OpBase;
}  // namespace imperative
}  // namespace paddle

namespace paddle {
namespace operators {

int SetMeta(const LoDTensor& srcTensor, LoDTensor* dstTensor) {
  if (srcTensor.dtype() == paddle::experimental::DataType::INT64 ||
      srcTensor.dtype() == paddle::experimental::DataType::FLOAT32 ||
      srcTensor.dtype() == paddle::experimental::DataType::FLOAT16 ||
      srcTensor.dtype() == paddle::experimental::DataType::FLOAT64) {
    const phi::DenseTensorMeta meta_data(srcTensor.dtype(), srcTensor.dims());
    dstTensor->set_meta(meta_data);
  } else {
    return xpu::Error_t::INVALID_PARAM;
  }

  return xpu::Error_t::SUCCESS;
}
template <typename T>
int CopyTensorByXPU(const LoDTensor& srcTensor,
                    LoDTensor* dstTensor,
                    int flag) {
  const T* srcData = srcTensor.template data<T>();
  if (nullptr == srcData || nullptr == dstTensor || flag < 0 || flag > 1)
    return xpu::Error_t::INVALID_PARAM;

  int r = SetMeta(srcTensor, dstTensor);
  PADDLE_ENFORCE_EQ(
      r,
      xpu::Error_t::SUCCESS,
      platform::errors::External("Execute function SetMeta failed by [%d]", r));

  if (flag == 0) {
    T* dstData =
        dstTensor->template mutable_data<T>(paddle::platform::CPUPlace());
    paddle::memory::Copy(paddle::platform::CPUPlace(),
                         dstData,
                         paddle::platform::XPUPlace(),
                         srcData,
                         srcTensor.numel() * sizeof(T));
  } else {
    T* dstData =
        dstTensor->template mutable_data<T>(paddle::platform::XPUPlace());
    paddle::memory::Copy(paddle::platform::XPUPlace(),
                         dstData,
                         paddle::platform::CPUPlace(),
                         srcData,
                         srcTensor.numel() * sizeof(T));
  }

  return xpu::Error_t::SUCCESS;
}

const int CopyTensorByType(const LoDTensor& srcTensor,
                           LoDTensor* dstTensor,
                           int flag) {
  int r = 0;
  if (srcTensor.dtype() == paddle::experimental::DataType::FLOAT32)
    r = CopyTensorByXPU<float>(srcTensor, dstTensor, flag);
  else if (srcTensor.dtype() == paddle::experimental::DataType::FLOAT16)
    r = CopyTensorByXPU<paddle::platform::float16>(srcTensor, dstTensor, flag);
  else if (srcTensor.dtype() == paddle::experimental::DataType::FLOAT64)
    r = CopyTensorByXPU<double>(srcTensor, dstTensor, flag);
  else
    return xpu::Error_t::INVALID_PARAM;

  PADDLE_ENFORCE_EQ(r,
                    xpu::Error_t::SUCCESS,
                    platform::errors::External(
                        "Execute function CopyTensorByXPU failed by [%d]", r));

  return xpu::Error_t::SUCCESS;
}

struct BeamSearchDecodeFunctor {
  BeamSearchDecodeFunctor(const LoDTensorArray& step_ids,
                          const LoDTensorArray& step_scores,
                          LoDTensor* id_tensor,
                          LoDTensor* score_tensor,
                          size_t beam_size,
                          int end_id)
      : beam_size_(beam_size),
        end_id_(end_id),
        id_tensor_(id_tensor),
        score_tensor_(score_tensor) {
    tensor_on_gpu_ = false;
    tensor_on_npu_ = false;
    tensor_on_xpu_ = false;
    int r = 0;

    if (platform::is_xpu_place(step_ids[0].place()) == false) {
      for (auto& step_id : step_ids) step_ids_origin_.push_back(step_id);

      for (auto& step_score : step_scores)
        step_scores_origin_.push_back(step_score);
    }

    // First make a copy of GPU data on CPU
    if (platform::is_gpu_place(step_ids[0].place()) ||
        platform::is_npu_place(step_ids[0].place()) ||
        platform::is_xpu_place(step_ids[0].place())) {
      if (platform::is_gpu_place(step_ids[0].place())) {
        tensor_on_gpu_ = true;
      } else if (platform::is_npu_place(step_ids[0].place())) {
        tensor_on_npu_ = true;
      } else {
        tensor_on_xpu_ = true;
      }

      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();

      auto* dev_ctx = pool.Get(step_ids[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_id : step_ids) {
        framework::LoDTensor out;
        if (step_id.numel() > 0) {
          if (tensor_on_gpu_) {
            dev_ctx->Wait();
          } else if (tensor_on_xpu_) {
            r = CopyTensorByXPU<int64_t>(step_id, &out, 0);
            PADDLE_ENFORCE_EQ(
                r,
                xpu::Error_t::SUCCESS,
                platform::errors::External(
                    "Execute function CopyTensorByXPU failed by [%d]", r));
          } else {
            framework::TensorCopy(
                step_id, platform::CPUPlace(), *dev_ctx, &out);
            dev_ctx->Wait();
          }
        }

        out.set_lod(step_id.lod());
        step_ids_.push_back(out);
      }
    }

    if (platform::is_gpu_place(step_scores[0].place()) ||
        platform::is_npu_place(step_scores[0].place()) ||
        platform::is_xpu_place(step_scores[0].place())) {
      if (platform::is_gpu_place(step_scores[0].place())) {
        tensor_on_gpu_ = true;
      } else if (platform::is_npu_place(step_scores[0].place())) {
        tensor_on_npu_ = true;
      } else {
        tensor_on_xpu_ = true;
      }

      platform::DeviceContextPool& pool =
          platform::DeviceContextPool::Instance();

      auto* dev_ctx = pool.Get(step_scores[0].place());
      // Copy all tensors in the input tensor array
      for (auto& step_score : step_scores) {
        framework::LoDTensor out;
        if (step_score.numel() > 0) {
          if (tensor_on_gpu_) {
            dev_ctx->Wait();
          } else if (tensor_on_xpu_) {
            r = CopyTensorByType(step_score, &out, 0);
            PADDLE_ENFORCE_EQ(
                r,
                xpu::Error_t::SUCCESS,
                platform::errors::External(
                    "Execute function CopyTensorByType failed by [%d]", r));

          } else {
            framework::TensorCopy(
                step_score, platform::CPUPlace(), *dev_ctx, &out);
            dev_ctx->Wait();
          }
        }

        out.set_lod(step_score.lod());
        step_scores_.push_back(out);
      }
    }
  }

  template <typename T>
  void apply() const;

  bool tensor_on_gpu_;
  bool tensor_on_npu_;
  bool tensor_on_xpu_;
  size_t beam_size_;
  int end_id_;
  // TODO(Superjomn) Here might result serious performance issue in the
  // concurrency
  // scenarios.
  LoDTensorArray step_ids_origin_ = LoDTensorArray();
  LoDTensorArray step_scores_origin_ = LoDTensorArray();
  LoDTensorArray step_ids_ = LoDTensorArray();
  LoDTensorArray step_scores_ = LoDTensorArray();
  LoDTensor* id_tensor_;
  LoDTensor* score_tensor_;
};

template <typename T>
void BeamSearchDecodeFunctor::apply() const {
  BeamSearchDecoder<T> beam_search_decoder(beam_size_, end_id_);
  // Check if the tensor is on GPU or NPU. If so, use the CPU copy instead
  if (tensor_on_gpu_ || tensor_on_npu_ || tensor_on_xpu_) {
    beam_search_decoder.Backtrace(
        step_ids_, step_scores_, id_tensor_, score_tensor_);
  } else {
    beam_search_decoder.Backtrace(
        step_ids_origin_, step_scores_origin_, id_tensor_, score_tensor_);
  }
}

template <>
void BeamSearchDecodeFunctor::apply<bool>() const {
  PADDLE_THROW(platform::errors::InvalidArgument(
      "beam search decode op does not support bool!"));
}

class BeamSearchDecodeOp : public framework::OperatorBase {
 public:
  BeamSearchDecodeOp(const std::string& type,
                     const framework::VariableNameMap& inputs,
                     const framework::VariableNameMap& outputs,
                     const framework::AttributeMap& attrs)
      : OperatorBase(type, inputs, outputs, attrs) {}

 private:
  void RunImpl(const framework::Scope& scope,
               const platform::Place& dev_place) const override {
    platform::DeviceContextPool& pool = platform::DeviceContextPool::Instance();
    auto& dev_ctx = *pool.Get(dev_place);

    framework::RuntimeContext run_ctx(Inputs(), Outputs(), scope);
    framework::ExecutionContext ctx(*this, scope, dev_ctx, run_ctx);

    const LoDTensorArray* ids = ctx.Input<LoDTensorArray>("Ids");
    const LoDTensorArray* scores = ctx.Input<LoDTensorArray>("Scores");
    const size_t step_num = ids->size();
    PADDLE_ENFORCE_GT(
        step_num,
        0UL,
        platform::errors::InvalidArgument(
            "beam search steps, which is the"
            "size of Input(Ids) LoDTensorArray. beam search steps should "
            "be larger than 0, but received %d. ",
            step_num));

    const size_t source_num = ids->at(0).lod().at(0).size() - 1;
    PADDLE_ENFORCE_GT(
        source_num,
        0UL,
        platform::errors::InvalidArgument(
            "source_num is the sequence number of the"
            "first decoding step, indicating by Input(Ids)[0].lod[0].size. "
            "The number of source_num should be larger than"
            "0, but received %d. ",
            source_num));

    for (size_t i = 0; i < step_num; ++i) {
      PADDLE_ENFORCE_EQ(
          ids->at(i).lod().size(),
          2UL,
          platform::errors::InvalidArgument(
              "For the i step in beam search steps,"
              "the size of Input(Ids)[i].lod() should larger than 2,"
              "but received %d. ",
              ids->at(i).lod().size()));
    }

    size_t beam_size = ctx.Attr<int>("beam_size");
    int end_id = ctx.Attr<int>("end_id");

    // prepare output
    LoDTensor* sentenceIds = nullptr;
    LoDTensor* sentenceScores = nullptr;

    LoDTensor* sentenceIds_temp = ctx.Output<LoDTensor>("SentenceIds");
    LoDTensor* sentenceScores_temp = ctx.Output<LoDTensor>("SentenceScores");

    if (platform::is_xpu_place(ids->at(0).place())) {
      sentenceIds = new LoDTensor();
      sentenceIds->set_lod(sentenceIds_temp->lod());
    } else {
      sentenceIds = sentenceIds_temp;
    }

    if (platform::is_xpu_place(ids->at(0).place())) {
      sentenceScores = new LoDTensor();
      sentenceScores->set_lod(sentenceScores_temp->lod());
    } else {
      sentenceScores = sentenceScores_temp;
    }

    framework::VisitDataType(
        framework::TransToProtoVarType(scores->at(0).dtype()),
        BeamSearchDecodeFunctor(
            *ids, *scores, sentenceIds, sentenceScores, beam_size, end_id));

    if (platform::is_xpu_place(ids->at(0).place())) {
      int r = 0;
      r = CopyTensorByXPU<int64_t>(*sentenceIds, sentenceIds_temp, 1);
      PADDLE_ENFORCE_EQ(
          r,
          xpu::Error_t::SUCCESS,
          platform::errors::External(
              "Execute function CopyTensorByXPU failed by [%d]", r));

      r = CopyTensorByXPU<float>(*sentenceScores, sentenceScores_temp, 1);
      PADDLE_ENFORCE_EQ(
          r,
          xpu::Error_t::SUCCESS,
          platform::errors::External(
              "Execute function CopyTensorByXPU failed by [%d]", r));
    }
  }
};

class BeamSearchDecodeOpProtoMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Ids",
             "(LodTensorArray)"
             "The LodTensorArray containing the selected ids of all steps");
    AddInput("Scores",
             "(LodTensorArray)"
             "The LodTensorArray containing the selected scores of all steps");
    AddOutput(
        "SentenceIds",
        "(LodTensor)"
        "An LodTensor containing all generated id sequences for all source "
        "sentences");
    AddOutput(
        "SentenceScores",
        "(LodTensor)"
        "An LodTensor containing scores corresponding to Output(SentenceIds)");
    AddAttr<int>("beam_size", "beam size for beam search");
    AddAttr<int>("end_id",
                 "the token id which indicates the end of a sequence");
    AddComment(R"DOC(
Beam Search Decode Operator. This Operator constructs the full hypotheses for
each source sentence by walking back along the LoDTensorArray Input(ids)
whose lods can be used to restore the path in the beam search tree.

The Output(SentenceIds) and Output(SentenceScores) separately contain the
generated id sequences and the corresponding scores. The shapes and lods of the
two LodTensor are same. The lod level is 2 and the two levels separately
indicate how many hypotheses each source sentence has and how many ids each
hypothesis has.
)DOC");
  }
};

class BeamSearchDecodeInferShape : public framework::InferShapeBase {
 public:
  void operator()(framework::InferShapeContext* context) const override {
    OP_INOUT_CHECK(
        context->HasInput("Ids"), "Input", "Ids", "BeamSearchDecode");
    OP_INOUT_CHECK(
        context->HasInput("Scores"), "Input", "Scores", "BeamSearchDecode");
    OP_INOUT_CHECK(context->HasOutput("SentenceIds"),
                   "Output",
                   "SentenceIds",
                   "BeamSearchDecode");
    OP_INOUT_CHECK(context->HasOutput("SentenceScores"),
                   "Output",
                   "SentenceScores",
                   "BeamSearchDecode");
  }
};

class BeamSearchDecodeInferVarType : public framework::VarTypeInference {
 public:
  void operator()(framework::InferVarTypeContext* ctx) const override {
    ctx->SetOutputType("SentenceIds",
                       framework::proto::VarType::LOD_TENSOR,
                       framework::ALL_ELEMENTS);
    ctx->SetOutputType("SentenceScores",
                       framework::proto::VarType::LOD_TENSOR,
                       framework::ALL_ELEMENTS);
  }
};

}  // namespace operators
}  // namespace paddle

REGISTER_OPERATOR(
    beam_search_decode,
    paddle::operators::BeamSearchDecodeOp,
    paddle::operators::BeamSearchDecodeOpProtoMaker,
    paddle::operators::BeamSearchDecodeInferShape,
    paddle::operators::BeamSearchDecodeInferVarType,
    paddle::framework::EmptyGradOpMaker<paddle::framework::OpDesc>,
    paddle::framework::EmptyGradOpMaker<paddle::imperative::OpBase>);
