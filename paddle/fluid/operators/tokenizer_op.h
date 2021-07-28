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

#include <unordered_set>
#include <string>
#include <vector>
#include <unordered_map>

#include "paddle/fluid/framework/op_registry.h"

namespace paddle {
namespace operators {

using std::wstring;
using std::string;
using std::shared_ptr;
using std::vector;
using std::unordered_map;
using std::unordered_set;
using std::int64_t;

using Vocab = unordered_map<std::wstring, int64_t>;
using InvVocab = unordered_map<int64_t, wstring>;

class BasicTokenizer {
 public:
  explicit BasicTokenizer(bool do_lower_case = true);
  vector<std::wstring> Tokenize(const string& text) const;

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
  explicit WordPieceTokenizer(
    const shared_ptr<Vocab>& vocab,
    const wstring& unk_token = L"[UNK]",
    const int64_t max_input_chars_per_word = 100);
  vector<wstring> Tokenize(const wstring& text) const;

 private:
  shared_ptr<Vocab> vocab_;
  wstring unk_token_{L"[UNK]"};
  int64_t max_input_chars_per_word_;
};

class FullTokenizer {
 public:
  explicit FullTokenizer(const string& vocab_file, bool do_lower_case = true);
  vector<wstring> Tokenize(const string& text) const;
  vector<int64_t> ConvertTokensToIds(const vector<wstring>& text) const;

 private:
  shared_ptr<Vocab> vocab_;
  InvVocab inv_vocab_;
  string vocab_file_;
  bool do_lower_case_{true};
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
};


class BertTokenizer {
 public:
    explicit BertTokenizer(
      const string& vocab_file,
      bool do_lower_case = true,
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
    vector<int64_t> ConvertTokensToIds(
      const vector<wstring>& tokens) const;
    string ConvertTokensToString(
      const vector<wstring>& tokens) const;
    vector<wstring> ConvertIdsToTokens(
      const vector<int64_t>& token_ids);
    unordered_map<string, vector<int64_t>> TruncateSequence(
      vector<int64_t>* ids,
      vector<int64_t>* pair_ids,
      const int64_t num_tokens_to_remove = 0,
      const string&  truncation_strategy = "longest_first",
      const int64_t stride = 0) const;
    vector<int64_t> GetSpecialTokensMask(
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>(),
      const bool already_has_special_tokens = false) const;
    int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
    unordered_map<string, vector<int64_t>> Encode(
      const string& text,
      const string& text_pair = "",
      const int64_t max_seq_len = -1,
      bool pad_to_max_seq_len = false,
      bool return_length = false,
      bool return_token_type_ids = true,
      bool return_position_ids = false,
      bool return_attention_mask = false,
      const string&  truncation_strategy = "longest_first",
      bool return_overflowing_tokens = false,
      bool return_special_tokens_mask = false) const;


 private:
    string vocab_file_;
    bool do_lower_case_;
    wstring unk_token_, pad_token_, cls_token_, mask_token_, sep_token_;
    string padding_site_;
    shared_ptr<Vocab> vocab_;
    BasicTokenizer basic_tokenizer_;
    WordPieceTokenizer word_piece_tokenizer_;
    int64_t unk_token_id_, cls_token_id_,
      mask_token_id_, pad_token_id_, sep_token_id_;
    vector<wstring> all_special_tokens_;
    unordered_set<int64_t> all_special_token_ids_;
    InvVocab inv_vocab_;

    vector<int64_t> get_input_ids(
      const string& text) const;
};


template <typename T>
class TokenizerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    auto* text = ctx.Input<std::vector<std::string>>("Text");
    auto* input_ids = ctx.Output<framework::Tensor>("InputIds");
    // auto* seg_ids = ctx.Output<framework::Tensor>("SegmentIds");

    input_ids->Resize(framework::make_ddim({static_cast<int64_t>(text->size())}));
    auto* input_ids_data = input_ids->mutable_data<T>(ctx.GetPlace());
    // only support cpu now
    VLOG(0) << "text size: " << text->size();
    for (int64_t i = 0; i < text->size(); ++i) {
      VLOG(0) << "text[" << i << "] = " << text->at(i)
              << ", size: " << text->at(i).size();
      input_ids_data[i] = text->at(i).size();
    }
  }
};

}  // namespace operators
}  // namespace paddle
