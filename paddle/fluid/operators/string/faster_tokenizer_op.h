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

#pragma once

#include <utf8proc.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/fluid/framework/op_registry.h"
#include "paddle/fluid/framework/string_array.h"

namespace paddle {
namespace operators {

using std::endl;
using std::int64_t;
using std::size_t;
using std::string;
using std::shared_ptr;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::wstring;
using std::wcout;

inline bool IsControl(const wchar_t& ch);
inline bool IsChineseChar(const wchar_t& ch);
inline bool IsWhiteSpace(const wchar_t& ch);

using Vocab = unordered_map<wstring, int>;
using InvVocab = unordered_map<int, wstring>;

class BasicTokenizer {
 public:
  explicit BasicTokenizer(bool do_lower_case = true);
  void Tokenize(const string& text, vector<wstring>* res) const;

 private:
  wchar_t do_lower_case(wchar_t ch) const;

  bool do_lower_case_;
};

class WordPieceTokenizer {
 public:
  explicit WordPieceTokenizer(const framework::Vocab* vocab,
                              const wstring& unk_token = L"[UNK]",
                              const size_t max_input_chars_per_word = 100);
  void Tokenize(const wstring& text, vector<int64_t>* output) const;

 private:
  const framework::Vocab* vocab_;
  wstring unk_token_{L"[UNK]"};
  int64_t unk_token_id_;
  size_t max_input_chars_per_word_;
};

class BertTokenizer {
 public:
  explicit BertTokenizer(const framework::Vocab* vocab,
                         bool do_lower_case = false,
                         const wstring& unk_token = L"[UNK]",
                         const wstring& pad_token = L"[PAD]",
                         const wstring& cls_token = L"[CLS]",
                         const wstring& mask_token = L"[MASK]",
                         const wstring& sep_token = L"[SEP]",
                         const string& padding_site = "right");

  void Tokenize(const string& text, vector<int64_t>* split_tokens) const;
  void BuildInputsWithSpecialTokens(
      vector<int64_t>* res, const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void CreateTokenTypeIdsFromSequences(
      vector<int64_t>* token_type_ids, const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void TruncateSequence(vector<int64_t>* ids, vector<int64_t>* pair_ids,
                        const size_t num_tokens_to_remove = 0,
                        const size_t stride = 0) const;
  int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
  int Encode(unordered_map<string, vector<int64_t>>* encoded_inputs,
             const string& text, const string& text_pair = "",
             bool is_split_into_words = false, const size_t max_seq_len = 0,
             bool pad_to_max_seq_len = false) const;
  void BatchEncode(
      vector<unordered_map<string, vector<int64_t>>>* batch_encode_inputs,
      const vector<string>& batch_text,
      const vector<string>& batch_text_pair = vector<string>(),
      bool is_split_into_words = false, const size_t max_seq_len = 0,
      bool pad_to_max_seq_len = false) const;

  int64_t GetPadTokenID() const;

 private:
  bool do_lower_case_;
  wstring unk_token_, pad_token_, cls_token_, mask_token_, sep_token_;
  string padding_site_;
  const framework::Vocab* vocab_;
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
  int64_t unk_token_id_, cls_token_id_, mask_token_id_, pad_token_id_,
      sep_token_id_;
  vector<wstring> all_special_tokens_;
  unordered_set<int64_t> all_special_token_ids_;
  InvVocab inv_vocab_;
};

template <typename T>
class FasterTokenizerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* text = ctx.Input<framework::Strings>("Text");
    auto* vocab = ctx.Input<framework::Vocab>("Vocab");

    auto* input_ids = ctx.Output<framework::Tensor>("InputIds");
    auto* seg_ids = ctx.Output<framework::Tensor>("SegmentIds");

    auto do_lower_case = static_cast<bool>(ctx.Attr<bool>("do_lower_case"));
    auto is_split_into_words =
        static_cast<bool>(ctx.Attr<bool>("is_split_into_words"));
    auto max_seq_len = static_cast<size_t>(ctx.Attr<int>("max_seq_len"));
    auto pad_to_max_seq_len =
        static_cast<bool>(ctx.Attr<bool>("pad_to_max_seq_len"));

    auto* text_pair = ctx.Input<framework::Strings>("TextPair");
    if (text_pair && text->size() != text_pair->size()) {
      VLOG(3) << "The input text(list[str]) and text pair (list[str]) must"
              << "be the same number of text sequence. Please check the input!";
      return;
    }

    BertTokenizer tokenizer(vocab, do_lower_case);
    size_t batch_max_seq_len = 0;
    size_t batch_size = text->size();

    vector<unordered_map<string, vector<int64_t>>> batch_encode_inputs(
        batch_size);
    if (text_pair) {
      tokenizer.BatchEncode(&batch_encode_inputs, *text, *text_pair,
                            is_split_into_words, max_seq_len,
                            pad_to_max_seq_len);
    } else {
      tokenizer.BatchEncode(&batch_encode_inputs, *text, vector<string>(),
                            is_split_into_words, max_seq_len,
                            pad_to_max_seq_len);
    }

    for (size_t i = 0; i < batch_size; ++i) {
      size_t seq_len = batch_encode_inputs[i]["input_ids"].size();
      if (seq_len > batch_max_seq_len) {
        batch_max_seq_len = seq_len;
      }
    }

    input_ids->Resize(
        phi::make_ddim({static_cast<int64_t>(batch_size),
                        static_cast<int64_t>(batch_max_seq_len)}));
    auto* input_ids_data = input_ids->mutable_data<T>(ctx.GetPlace());
    seg_ids->Resize(phi::make_ddim({static_cast<int64_t>(batch_size),
                                    static_cast<int64_t>(batch_max_seq_len)}));
    auto* seg_ids_data = seg_ids->mutable_data<T>(ctx.GetPlace());

    auto pad_token_id = tokenizer.GetPadTokenID();
    for (size_t i = 0; i < batch_size; i++) {
      auto& encoder_input_ids = batch_encode_inputs[i]["input_ids"];
      auto& encoder_seg_ids = batch_encode_inputs[i]["token_type_ids"];
      const size_t& seq_len = encoder_input_ids.size();
      // Copy the memory
      std::memcpy(input_ids_data + i * batch_max_seq_len,
                  encoder_input_ids.data(), seq_len * sizeof(T));
      std::memcpy(seg_ids_data + i * batch_max_seq_len, encoder_seg_ids.data(),
                  seq_len * sizeof(T));
      std::memset(input_ids_data + i * batch_max_seq_len + seq_len,
                  pad_token_id, (batch_max_seq_len - seq_len) * sizeof(T));
      std::memset(seg_ids_data + i * batch_max_seq_len + seq_len, pad_token_id,
                  (batch_max_seq_len - seq_len) * sizeof(T));
    }
  }
};

}  // namespace operators
}  // namespace paddle
