/* Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#include <utf8proc.h>
#include <algorithm>
#include <chrono>
#include <codecvt>
#include <fstream>
#include <iostream>
#include <numeric>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace paddle {
namespace operators {

class FasterTokenizerOp : public framework::OperatorWithKernel {
 public:
  using framework::OperatorWithKernel::OperatorWithKernel;

  void InferShape(framework::InferShapeContext* ctx) const override {
    OP_INOUT_CHECK(ctx->HasInput("Text"), "Input", "Text", "Tokenizer");
    OP_INOUT_CHECK(ctx->HasInput("Vocab"), "Input", "Vocab", "Tokenizer");
    OP_INOUT_CHECK(
        ctx->HasOutput("InputIds"), "Output", "InputIds", "Tokenizer");
    OP_INOUT_CHECK(
        ctx->HasOutput("SegmentIds"), "Output", "SegmentIds", "Tokenizer");

    ctx->SetOutputDim("InputIds", {-1, -1});
    ctx->SetOutputDim("SegmentIds", {-1, -1});
  }

 protected:
  phi::KernelKey GetExpectedKernelType(
      const framework::ExecutionContext& ctx) const override {
    return phi::KernelKey(framework::proto::VarType::INT64, phi::CPUPlace());
  }

  phi::KernelKey GetKernelTypeForVar(
      const std::string& var_name,
      const phi::DenseTensor& tensor,
      const phi::KernelKey& expected_kernel_type) const override {
    return phi::KernelKey(phi::Backend::ALL_BACKEND,
                          tensor.layout(),
                          expected_kernel_type.dtype());
  }
};

class FasterTokenizerOpMaker : public framework::OpProtoAndCheckerMaker {
 public:
  void Make() override {
    AddInput("Vocab",
             "(std::map<std::wstring, std::int>), The vocab to map "
             "token string to token id.");
    AddInput("Text",
             "(std::vector<std::string>), The sequence to be processed. "
             "One sequence is a string, a list of strings, "
             "or a list of integers depending on whether it "
             "has been pretokenized and converted to ids. ");
    AddInput("TextPair",
             "(std::vector<std::string>), Same as `text` argument, "
             "while it represents for the latter sequence of the "
             "sequence pair.")
        .AsDispensable();
    AddOutput("InputIds", "(Tensor), The token ids of the input text.");
    AddOutput("SegmentIds", "(Tensor), The segments ids of the input text.");
    AddAttr<bool>(
        "do_lower_case",
        "(bool), Whether or not to lowercase the input when tokenizing.")
        .SetDefault(false);
    AddAttr<bool>(
        "is_split_into_words",
        "(bool), Whether or not the input is already pre-tokenized "
        "(e.g., split into words). If set to True, the tokenizer "
        "assumes the input is already split into words (for instance, "
        "by splitting it on whitespace) which it will tokenize. This "
        "is useful for NER or token classification.")
        .SetDefault(false);
    AddAttr<int>("max_seq_len",
                 "(int), If set to a positive number, will limit the "
                 "total sequence returned so that it has a maximum length."
                 " If there are overflowing tokens, those overflowing "
                 "tokens will be added to the returned dictionary  when "
                 "`return_overflowing_tokens` is `True`.")
        .SetDefault(0);
    AddAttr<bool>("pad_to_max_seq_len",
                  "(bool), If set to `True`, the returned sequences would be"
                  " padded up to `max_seq_len` specified length according to"
                  " padding side and padding token id.")
        .SetDefault(false);
    AddComment(R"DOC(Performs tokenization and uses the tokenized tokens to "
    "prepare model inputs. It supports sequence or sequence pair as input, "
    "and batch input is not allowed.)DOC");
  }
};

}  // namespace operators
}  // namespace paddle

namespace ops = paddle::operators;
REGISTER_OPERATOR(faster_tokenizer,
                  ops::FasterTokenizerOp,
                  ops::FasterTokenizerOpMaker);
