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
// #include <chrono>

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
  void Tokenize(const string& text, vector<wstring>* res) const;

 private:
  void clean_text(const wstring& text, wstring* output) const;
  bool is_chinese_char(const wchar_t& ch) const;
  void tokenize_chinese_chars(const wstring& text, wstring* output) const;
  void run_strip_accents(const wstring& text, wstring* output) const;
  void run_split_on_punc(const wstring& text, vector<wstring>* res) const;

  bool do_lower_case_{true};
};

class WordPieceTokenizer {
 public:
  explicit WordPieceTokenizer(framework::WSTRING_MAP* vocab,
                              const wstring& unk_token = L"[UNK]",
                              const size_t max_input_chars_per_word = 100);
  void Tokenize(const wstring& text, vector<int64_t>* output) const;

 private:
  framework::WSTRING_MAP* vocab_;
  wstring unk_token_;
  int64_t unk_token_id_;
  size_t max_input_chars_per_word_;
};

class BertTokenizer {
 public:
  explicit BertTokenizer(framework::WSTRING_MAP* vocab,
                         const bool& do_lower_case = false,
                         const wstring& unk_token = L"[UNK]",
                         const wstring& pad_token = L"[PAD]",
                         const wstring& cls_token = L"[CLS]",
                         const wstring& mask_token = L"[MASK]",
                         const wstring& sep_token = L"[SEP]",
                         const string& padding_site = "right");

  void Tokenize(const string& text, vector<int64_t>* split_token_ids) const;
  void BuildInputsWithSpecialTokens(
      vector<int64_t>* res, const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void CreateTokenTypeIdsFromSequences(
      vector<int64_t>* token_type_ids, const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void ConvertTokensToIds(const vector<wstring>& tokens,
                          vector<int64_t>* token_ids) const;
  string ConvertTokensToString(const vector<wstring>& tokens) const;
  int TruncateSequence(
      // unordered_map<string, vector<int64_t>>* res,
      vector<int64_t>* ids, vector<int64_t>* pair_ids,
      const size_t num_tokens_to_remove = 0,
      const string& truncation_strategy = "longest_first",
      const size_t stride = 0) const;
  vector<int64_t> GetSpecialTokensMask(
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>(),
      const bool already_has_special_tokens = false) const;
  int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
  int Encode(unordered_map<string, vector<int64_t>>* encoded_inputs,
             const string& text, const string& text_pair = "",
             const size_t max_seq_len = 0, bool pad_to_max_seq_len = false,
             bool return_length = false, bool return_token_type_ids = true,
             bool return_position_ids = false,
             bool return_attention_mask = false,
             const string& truncation_strategy = "longest_first",
             bool return_overflowing_tokens = false,
             bool return_special_tokens_mask = false) const;
  int BatchEncode(
      vector<unordered_map<string, vector<int64_t>>>* batch_encode_inputs,
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
  framework::WSTRING_MAP* vocab_;
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
  int64_t unk_token_id_, cls_token_id_, mask_token_id_, pad_token_id_,
      sep_token_id_;
  vector<wstring> all_special_tokens_;
  unordered_set<int64_t> all_special_token_ids_;
  InvVocab inv_vocab_;

  void get_input_ids(const string& text, vector<int64_t>* token_ids) const;
};

template <typename T>
class TokenizerKernel : public framework::OpKernel<T> {
 public:
  void Compute(const framework::ExecutionContext& ctx) const override {
    std::chrono::steady_clock::time_point begin =
        std::chrono::steady_clock::now();
    auto* text = ctx.Input<framework::STRINGS>("Text");
    auto* vocab = ctx.Input<framework::WSTRING_MAP>("Vocab");
    std::chrono::steady_clock::time_point end =
        std::chrono::steady_clock::now();
    VLOG(0) << "Time difference stage_0_0 = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;

    auto* input_ids = ctx.Output<framework::Tensor>("InputIds");
    auto* seg_ids = ctx.Output<framework::Tensor>("SegmentIds");
    end = std::chrono::steady_clock::now();
    VLOG(0) << "Time difference stage_0_1 = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;

    auto is_split_into_words =
        static_cast<bool>(ctx.Attr<bool>("is_split_into_words"));
    auto max_seq_len = static_cast<size_t>(ctx.Attr<int>("max_seq_len"));
    auto pad_to_max_seq_len =
        static_cast<bool>(ctx.Attr<bool>("pad_to_max_seq_len"));
    end = std::chrono::steady_clock::now();
    VLOG(0) << "Time difference stage_0_2 = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;

    auto* text_pair = ctx.Input<framework::STRINGS>("TextPair");
    if (text_pair && text->size() != text_pair->size()) {
      VLOG(3) << "The input text(list[str]) and text pair (list[str]) must"
              << "be the same number of text sequence. Please check the input!";
      return;
    }
    end = std::chrono::steady_clock::now();
    VLOG(0) << "Time difference stage_0_3 = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;

    BertTokenizer* tokenizer_ptr =
        new BertTokenizer(const_cast<framework::WSTRING_MAP*>(vocab));
    end = std::chrono::steady_clock::now();
    VLOG(0) << "Time difference stage_1 = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;
    // only support cpu now
    size_t batch_max_seq_len = 0;
    size_t batch_size = text->size();

    // unordered_map<size_t, vector<T>> batch_input_ids;
    // unordered_map<size_t, vector<T>> batch_seg_ids;
    vector<unordered_map<string, vector<int64_t>>> batch_encode_inputs;
    int status;
    if (text_pair) {
      status = tokenizer_ptr->BatchEncode(&batch_encode_inputs, *text,
                                          *text_pair, is_split_into_words,
                                          max_seq_len, pad_to_max_seq_len);
    } else {
      status = tokenizer_ptr->BatchEncode(&batch_encode_inputs, *text,
                                          vector<string>(), is_split_into_words,
                                          max_seq_len, pad_to_max_seq_len);
    }

    PADDLE_ENFORCE_EQ(
        status, 1,
        platform::errors::InvalidArgument(
            "Tokenizer op computes failly.  Please check the input."));

    for (size_t i = 0; i < batch_size; ++i) {
      size_t seq_len = batch_encode_inputs[i]["input_ids"].size();
      // batch_input_ids[i] = encoded_inputs["input_ids"];
      // batch_seg_ids[i] = encoded_inputs["token_type_ids"];
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
    end = std::chrono::steady_clock::now();
    VLOG(0) << "Time difference stage_2 = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[µs]" << std::endl;
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
      /*
      for (size_t j = 0; j < batch_max_seq_len; j++) {
        if (j < seq_len) {
          input_ids_data[i * batch_max_seq_len + j] = batch_input_ids[i][j];
          seg_ids_data[i * batch_max_seq_len + j] = batch_seg_ids[i][j];
        } else {
          input_ids_data[i * batch_max_seq_len + j] = pad_token_id;
          seg_ids_data[i * batch_max_seq_len + j] = pad_token_id;
        }
      }*/
    }
    delete tokenizer_ptr;
    end = std::chrono::steady_clock::now();
    VLOG(0) << "Time difference = "
            << std::chrono::duration_cast<std::chrono::microseconds>(end -
                                                                     begin)
                   .count()
            << "[us]" << std::endl;
  }
};

}  // namespace operators
}  // namespace paddle
