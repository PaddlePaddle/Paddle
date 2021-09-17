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

using Vocab = unordered_map<wstring, int>;
using InvVocab = unordered_map<int, wstring>;

class BasicTokenizer {
 public:
  explicit BasicTokenizer(bool do_lower_case = true);
  vector<wstring> Tokenize(const string& text) const;

 private:
  wstring clean_text(const wstring& text) const;
  bool is_chinese_char(const wchar_t& ch) const;
  wstring tokenize_chinese_chars(const wstring& text) const;
  wstring run_strip_accents(const wstring& text) const;
  vector<wstring> run_split_on_punc(const wstring& text) const;

  bool do_lower_case_{true};
};

class WordPieceTokenizer {
 public:
  explicit WordPieceTokenizer(const framework::WSTRING_MAP vocab,
                              const wstring& unk_token = L"[UNK]",
                              const size_t max_input_chars_per_word = 100);
  vector<wstring> Tokenize(const wstring& text) const;

 private:
  framework::WSTRING_MAP vocab_;
  wstring unk_token_{L"[UNK]"};
  size_t max_input_chars_per_word_;
};

class BertTokenizer {
 public:
  explicit BertTokenizer(const framework::WSTRING_MAP vocab,
                         bool do_lower_case = false,
                         const wstring& unk_token = L"[UNK]",
                         const wstring& pad_token = L"[PAD]",
                         const wstring& cls_token = L"[CLS]",
                         const wstring& mask_token = L"[MASK]",
                         const wstring& sep_token = L"[SEP]",
                         const string& padding_site = "right");

  vector<wstring> Tokenize(const string& text) const;
  vector<int64_t> BuildInputsWithSpecialTokens(
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  vector<int64_t> CreateTokenTypeIdsFromSequences(
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  vector<int64_t> ConvertTokensToIds(const vector<wstring>& tokens) const;
  string ConvertTokensToString(const vector<wstring>& tokens) const;
  vector<wstring> ConvertIdsToTokens(const vector<int64_t>& token_ids);
  unordered_map<string, vector<int64_t>> TruncateSequence(
      vector<int64_t>* ids, vector<int64_t>* pair_ids,
      const size_t num_tokens_to_remove = 0,
      const string& truncation_strategy = "longest_first",
      const size_t stride = 0) const;
  vector<int64_t> GetSpecialTokensMask(
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>(),
      const bool already_has_special_tokens = false) const;
  int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
  unordered_map<string, vector<int64_t>> Encode(
      const string& text, const string& text_pair = "",
      const size_t max_seq_len = 0, bool pad_to_max_seq_len = false,
      bool return_length = false, bool return_token_type_ids = true,
      bool return_position_ids = false, bool return_attention_mask = false,
      const string& truncation_strategy = "longest_first",
      bool return_overflowing_tokens = false,
      bool return_special_tokens_mask = false) const;
  vector<unordered_map<string, vector<int64_t>>> BatchEncode(
      const vector<string>& batch_text,
      const vector<string>& batch_text_pair = vector<string>(),
      bool is_split_into_words = false, const size_t max_seq_len = 0,
      bool pad_to_max_seq_len = false, bool return_length = false,
      bool return_token_type_ids = true, bool return_position_ids = false,
      bool return_attention_mask = false,
      const string& truncation_strategy = "longest_first",
      const size_t stride = 0, bool return_overflowing_tokens = false,
      bool return_special_tokens_mask = false) const;

  int64_t GetUnkTokenID() const;
  int64_t GetPadTokenID() const;
  int64_t GetClsTokenID() const;
  int64_t GetMaskTokenID() const;
  int64_t GetSepTokenID() const;

 private:
  bool do_lower_case_;
  wstring unk_token_, pad_token_, cls_token_, mask_token_, sep_token_;
  string padding_site_;
  framework::WSTRING_MAP vocab_;
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
  int64_t unk_token_id_, cls_token_id_, mask_token_id_, pad_token_id_,
      sep_token_id_;
  vector<wstring> all_special_tokens_;
  unordered_set<int64_t> all_special_token_ids_;
  InvVocab inv_vocab_;

  vector<int64_t> get_input_ids(const string& text) const;
};

template <typename T>
class TokenizerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* text = ctx.Input<framework::STRINGS>("Text");
    auto* vocab = ctx.Input<framework::WSTRING_MAP>("Vocab");

    auto* input_ids = ctx.Output<framework::Tensor>("InputIds");
    auto* seg_ids = ctx.Output<framework::Tensor>("SegmentIds");

    auto is_split_into_words =
        static_cast<bool>(ctx.Attr<bool>("is_split_into_words"));
    auto max_seq_len = static_cast<size_t>(ctx.Attr<int>("max_seq_len"));
    auto pad_to_max_seq_len =
        static_cast<bool>(ctx.Attr<bool>("pad_to_max_seq_len"));

    auto* text_pair = ctx.Input<framework::STRINGS>("TextPair");
    if (text_pair && text->size() != text_pair->size()) {
      VLOG(3) << "The input text(list[str]) and text pair (list[str]) must"
              << "be the same number of text sequence. Please check the input!";
      return;
    }

    BertTokenizer* tokenizer_ptr = new BertTokenizer(*vocab);

    // only support cpu now
    size_t batch_max_seq_len = 0;
    size_t batch_size = text->size();

    unordered_map<size_t, vector<T>> batch_input_ids;
    unordered_map<size_t, vector<T>> batch_seg_ids;
    vector<unordered_map<string, vector<int64_t>>> batch_encode_inputs;
    if (text_pair) {
      batch_encode_inputs =
          tokenizer_ptr->BatchEncode(*text, *text_pair, is_split_into_words,
                                     max_seq_len, pad_to_max_seq_len);
    } else {
      batch_encode_inputs = tokenizer_ptr->BatchEncode(
          *text, vector<string>(), is_split_into_words, max_seq_len,
          pad_to_max_seq_len);
    }
    for (size_t i = 0; i < batch_size; ++i) {
      auto encoded_inputs = batch_encode_inputs[i];
      size_t seq_len = encoded_inputs["input_ids"].size();
      batch_input_ids[i] = encoded_inputs["input_ids"];
      batch_seg_ids[i] = encoded_inputs["token_type_ids"];
      if (seq_len > batch_max_seq_len) {
        batch_max_seq_len = seq_len;
      }
    }

    input_ids->Resize(
        framework::make_ddim({static_cast<int64_t>(batch_size),
                              static_cast<int64_t>(batch_max_seq_len)}));
    auto* input_ids_data = input_ids->mutable_data<T>(ctx.GetPlace());
    seg_ids->Resize(
        framework::make_ddim({static_cast<int64_t>(batch_size),
                              static_cast<int64_t>(batch_max_seq_len)}));
    auto* seg_ids_data = seg_ids->mutable_data<T>(ctx.GetPlace());

    auto pad_token_id = tokenizer_ptr->GetPadTokenID();
    for (size_t i = 0; i < batch_size; i++) {
      size_t seq_len = batch_input_ids[i].size();
      for (size_t j = 0; j < batch_max_seq_len; j++) {
        if (j < seq_len) {
          input_ids_data[i * batch_max_seq_len + j] = batch_input_ids[i][j];
          seg_ids_data[i * batch_max_seq_len + j] = batch_seg_ids[i][j];
        } else {
          input_ids_data[i * batch_max_seq_len + j] = pad_token_id;
          seg_ids_data[i * batch_max_seq_len + j] = pad_token_id;
        }
      }
    }
  }
};

}  // namespace operators
}  // namespace paddle
