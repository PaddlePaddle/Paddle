// Copyright (c) 2024 PaddlePaddle Authors. All Rights Reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <utf8proc.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "glog/logging.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/tensor_utils.h"
#include "paddle/phi/core/vocab/string_array.h"

namespace phi {

using std::endl;
using std::ifstream;
using std::int64_t;
using std::shared_ptr;
using std::size_t;
using std::string;
using std::unordered_map;
using std::unordered_set;
using std::vector;
using std::wcout;
using std::wstring;

inline bool IsControl(const wchar_t& ch);
inline bool IsChineseChar(const wchar_t& ch);
inline bool IsWhiteSpace(const wchar_t& ch);

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
  explicit WordPieceTokenizer(const phi::Vocab* vocab,
                              const wstring& unk_token = L"[UNK]",
                              const size_t max_input_chars_per_word = 100);
  void Tokenize(const wstring& text, vector<int64_t>* output) const;

 private:
  const phi::Vocab* vocab_;
  wstring unk_token_{L"[UNK]"};
  int64_t unk_token_id_;
  size_t max_input_chars_per_word_;
};

class BertTokenizer {
 public:
  explicit BertTokenizer(const phi::Vocab* vocab,
                         bool do_lower_case = false,
                         const wstring& unk_token = L"[UNK]",
                         const wstring& pad_token = L"[PAD]",
                         const wstring& cls_token = L"[CLS]",
                         const wstring& mask_token = L"[MASK]",
                         const wstring& sep_token = L"[SEP]",
                         const string& padding_site = "right");

  void Tokenize(const string& text, vector<int64_t>* split_tokens) const;
  void BuildInputsWithSpecialTokens(
      vector<int64_t>* res,
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void CreateTokenTypeIdsFromSequences(
      vector<int64_t>* token_type_ids,
      const vector<int64_t>& token_ids_0,
      const vector<int64_t>& token_ids_1 = vector<int64_t>()) const;
  void TruncateSequence(vector<int64_t>* ids,
                        vector<int64_t>* pair_ids,
                        const size_t num_tokens_to_remove = 0,
                        const size_t stride = 0) const;
  int64_t GetNumSpecialTokensToAdd(const bool pair = false) const;
  int Encode(unordered_map<string, vector<int64_t>>* encoded_inputs,
             const string& text,
             const string& text_pair = "",
             bool is_split_into_words = false,
             const size_t max_seq_len = 0,
             bool pad_to_max_seq_len = false) const;
  void BatchEncode(
      vector<unordered_map<string, vector<int64_t>>>* batch_encode_inputs,
      const Strings& batch_text,
      const Strings& batch_text_pair = Strings(),
      bool is_split_into_words = false,
      const size_t max_seq_len = 0,
      bool pad_to_max_seq_len = false) const;

  int64_t GetPadTokenID() const;

 private:
  bool do_lower_case_;
  wstring unk_token_, pad_token_, cls_token_, mask_token_, sep_token_;
  string padding_site_;
  const phi::Vocab* vocab_;
  BasicTokenizer basic_tokenizer_;
  WordPieceTokenizer word_piece_tokenizer_;
  int64_t unk_token_id_, cls_token_id_, mask_token_id_, pad_token_id_,
      sep_token_id_;
  vector<wstring> all_special_tokens_;
  unordered_set<int64_t> all_special_token_ids_;
  InvVocab inv_vocab_;
};

const wstring kStripChars = L" \t\n\r\v\f";

inline bool IsControl(const wchar_t& ch) {
  if (ch == L'\t' || ch == L'\n' || ch == L'\r') return false;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_CC || cat == UTF8PROC_CATEGORY_CF) return true;
  return false;
}

inline bool IsChineseChar(const wchar_t& ch) {
  if ((ch >= 0x4E00 && ch <= 0x9FFF) || (ch >= 0x3400 && ch <= 0x4DBF) ||
      (ch >= 0x20000 && ch <= 0x2A6DF) || (ch >= 0x2A700 && ch <= 0x2B73F) ||
      (ch >= 0x2B740 && ch <= 0x2B81F) || (ch >= 0x2B820 && ch <= 0x2CEAF) ||
      (ch >= 0xF900 && ch <= 0xFAFF) || (ch >= 0x2F800 && ch <= 0x2FA1F))
    return true;
  return false;
}

inline bool IsWhiteSpace(const wchar_t& ch) {
  if (ch == L' ' || ch == L'\t' || ch == L'\n' || ch == L'\r') return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_ZS) return true;
  return false;
}

inline bool IsPunctuation(const wchar_t& ch) {
  if ((ch >= 33 && ch <= 47) || (ch >= 58 && ch <= 64) ||
      (ch >= 91 && ch <= 96) || (ch >= 123 && ch <= 126))
    return true;
  auto cat = utf8proc_category(ch);
  if (cat == UTF8PROC_CATEGORY_PD || cat == UTF8PROC_CATEGORY_PS ||
      cat == UTF8PROC_CATEGORY_PE || cat == UTF8PROC_CATEGORY_PC ||
      cat == UTF8PROC_CATEGORY_PO  // sometimes ¶ belong SO
      || cat == UTF8PROC_CATEGORY_PI || cat == UTF8PROC_CATEGORY_PF)
    return true;
  return false;
}

BasicTokenizer::BasicTokenizer(bool do_lower_case /* = true */)
    : do_lower_case_(do_lower_case) {}

wchar_t BasicTokenizer::do_lower_case(wchar_t ch) const {
  wchar_t new_ch = utf8proc_tolower(ch);
  return new_ch;
}

void BasicTokenizer::Tokenize(const string& text, vector<wstring>* res) const {
  std::wstring unicode_text;
  bool status = phi::ConvertStrToWstr(text, &unicode_text);
  if (!status) {
    // String is converted into wstring failedly.
    return;
  }
  std::wstring cache_text = L"";
  auto PushCacheText = [&]() {
    if (!cache_text.empty()) {
      res->emplace_back(cache_text);
      cache_text = L"";
    }
  };
  for (auto& ch : unicode_text) {
    if (ch == 0 || ch == 0xfffd || IsControl(ch)) {
      continue;
    }
    if (do_lower_case_) {
      ch = do_lower_case(ch);
    }
    if (IsChineseChar(ch) || IsPunctuation(ch)) {
      PushCacheText();
      res->emplace_back(std::wstring{ch});
    } else if (IsWhiteSpace(ch)) {
      PushCacheText();
    } else {
      cache_text += ch;
    }
  }
  PushCacheText();
}

WordPieceTokenizer::WordPieceTokenizer(
    const phi::Vocab* vocab,
    const wstring& unk_token /* = L"[UNK]"*/,
    const size_t max_input_chars_per_word /* = 100 */)
    : vocab_(vocab),
      unk_token_(unk_token),
      max_input_chars_per_word_(max_input_chars_per_word) {
  unk_token_id_ = vocab_->at(unk_token_);
}

void WordPieceTokenizer::Tokenize(const wstring& text,
                                  vector<int64_t>* token_ids) const {
  size_t len = text.size();
  if (len > max_input_chars_per_word_) {
    token_ids->emplace_back(unk_token_id_);
    return;
  }

  auto it = vocab_->find(text);
  if (it != vocab_->end()) {
    token_ids->emplace_back(it->second);
    return;
  }

  size_t start = 0;
  vector<int64_t> wordpiece_ids;
  while (start < len) {
    size_t end = len;
    std::wstring cur_substr;
    int64_t cur_substr_id = 0;
    while (start < end) {
      std::wstring sub = text.substr(start, end - start);
      if (start > 0) {
        sub.insert(0, L"##");
      }
      auto it = vocab_->find(sub);
      if (it != vocab_->end()) {
        cur_substr = sub;
        cur_substr_id = it->second;
        break;
      }
      end -= 1;
    }

    if (cur_substr.empty()) {
      token_ids->emplace_back(unk_token_id_);
      return;
    } else {
      start = end;
      wordpiece_ids.emplace_back(cur_substr_id);
    }
  }
  for (auto& token_id : wordpiece_ids) {
    token_ids->emplace_back(token_id);
  }
}

BertTokenizer::BertTokenizer(const phi::Vocab* vocab,
                             bool do_lower_case /* = false */,
                             const wstring& unk_token /* = L"[UNK]" */,
                             const wstring& pad_token /* = L"[PAD]" */,
                             const wstring& cls_token /* = L"[CLS]" */,
                             const wstring& mask_token /* = L"[MASK]" */,
                             const wstring& sep_token /* = L"[SEP]" */,
                             const string& padding_site /* = "right" */)
    : do_lower_case_(do_lower_case),
      unk_token_(unk_token),
      pad_token_(pad_token),
      cls_token_(cls_token),
      mask_token_(mask_token),
      sep_token_(sep_token),
      padding_site_(padding_site),
      vocab_(vocab),
      basic_tokenizer_(do_lower_case_),
      word_piece_tokenizer_(vocab_, unk_token) {
  unk_token_id_ = vocab_->at(unk_token_);
  pad_token_id_ = vocab_->at(pad_token_);
  cls_token_id_ = vocab_->at(cls_token_);
  mask_token_id_ = vocab_->at(mask_token_);
  sep_token_id_ = vocab_->at(sep_token_);

  all_special_tokens_ = vector<wstring>(
      {unk_token_, pad_token_, cls_token_, mask_token_, sep_token_});
  all_special_token_ids_ = unordered_set<int64_t>({unk_token_id_,
                                                   pad_token_id_,
                                                   cls_token_id_,
                                                   mask_token_id_,
                                                   sep_token_id_});
}

void BertTokenizer::Tokenize(const string& text,
                             vector<int64_t>* split_token_ids) const {
  std::vector<std::wstring> tmp_tokens;
  basic_tokenizer_.Tokenize(text, &tmp_tokens);
  if (tmp_tokens.empty()) return;
  split_token_ids->reserve(tmp_tokens.size());
  for (auto& w_token : tmp_tokens) {
    const auto& vec_size = w_token.size();
    if (vec_size == 1) {
      if (IsChineseChar(w_token[0])) {
        auto vocab_it = vocab_->find(w_token);
        if (vocab_it != vocab_->end()) {
          split_token_ids->emplace_back(vocab_it->second);
        } else {
          split_token_ids->emplace_back(unk_token_id_);
        }
      } else {
        word_piece_tokenizer_.Tokenize(w_token, split_token_ids);
      }
    } else if (vec_size > 1) {
      word_piece_tokenizer_.Tokenize(w_token, split_token_ids);
    } else {
      continue;
    }
  }
}

void BertTokenizer::BuildInputsWithSpecialTokens(
    vector<int64_t>* inputs,
    const vector<int64_t>& token_ids_0,
    const vector<int64_t>& token_ids_1 /* = vector<int64_t>() */) const {
  if (token_ids_1.empty()) {
    inputs->clear();
    inputs->resize(token_ids_0.size() + 2);
    inputs->at(0) = cls_token_id_;
    size_t i = 1;
    for (auto& token_id : token_ids_0) {
      inputs->at(i) = token_id;
      ++i;
    }
    inputs->at(i) = sep_token_id_;
  } else {
    inputs->clear();
    inputs->resize(token_ids_0.size() + token_ids_1.size() + 3);
    inputs->at(0) = cls_token_id_;
    size_t i = 1;
    for (auto& token_id : token_ids_0) {
      inputs->at(i) = token_id;
      ++i;
    }
    inputs->at(i) = sep_token_id_;
    ++i;
    for (auto& token_id : token_ids_1) {
      inputs->at(i) = token_id;
      ++i;
    }
    inputs->at(i) = sep_token_id_;
  }
}

int64_t BertTokenizer::GetNumSpecialTokensToAdd(const bool pair) const {
  if (pair) {
    return 3;
  } else {
    return 2;
  }
}

void BertTokenizer::CreateTokenTypeIdsFromSequences(
    vector<int64_t>* token_type_ids,
    const vector<int64_t>& token_ids_0,
    const vector<int64_t>& token_ids_1 /* = vector<int64_t>() */) const {
  if (token_ids_1.empty()) {
    vector<int64_t> tmp(token_ids_0.size() + 2, 0);
    token_type_ids->swap(tmp);
  } else {
    vector<int64_t> tmp(token_ids_0.size() + token_ids_1.size() + 3, 0);
    for (size_t i = token_ids_0.size() + 2; i < tmp.size(); i++) {
      tmp[i] = 1;
    }
    token_type_ids->swap(tmp);
  }
}

void BertTokenizer::TruncateSequence(
    vector<int64_t>* ids,
    vector<int64_t>* pair_ids,
    const size_t num_tokens_to_remove /* = 0 */,
    const size_t stride /* = 0 */) const {
  for (size_t i = 0; i < num_tokens_to_remove; i++) {
    if ((pair_ids->empty()) || (ids->size() > pair_ids->size())) {
      ids->pop_back();
    } else {
      pair_ids->pop_back();
    }
  }
}

int64_t BertTokenizer::GetPadTokenID() const { return pad_token_id_; }

int BertTokenizer::Encode(
    unordered_map<string, vector<int64_t>>* encoded_inputs,
    const string& text,
    const string& text_pair /* = "" */,
    bool is_split_into_words /* = false */,
    const size_t max_seq_len /* = 0 */,
    bool pad_to_max_seq_len /* = false */) const {
  vector<int64_t> ids;
  vector<int64_t> pair_ids;
  if (!is_split_into_words) {
    Tokenize(text, &ids);
    if (ids.empty()) return 0;
    if (!text_pair.empty()) {
      Tokenize(text_pair, &pair_ids);
      if (pair_ids.empty()) return 0;
    }
  } else {
    std::wstring unicode_text;
    bool status_a = phi::ConvertStrToWstr(text, &unicode_text);
    if (!status_a) {
      return 0;
    }
    for (size_t i = 0; i < unicode_text.size(); i++) {
      wstring token = unicode_text.substr(i, 1);
      auto it = vocab_->find(token);
      if (it != vocab_->end()) {
        ids.emplace_back(it->second);
      } else {
        ids.emplace_back(unk_token_id_);
      }
    }
  }

  bool pair = false;
  if (!pair_ids.empty()) {
    pair = true;
  }

  size_t len_ids = ids.size();
  size_t len_pair_ids = pair_ids.size();

  // Truncation: Handle max sequence length
  // If max_seq_len == 0, then do nothing and keep the real length.
  // If max_seq_len > 0 and
  // all the input sequence len is over the max_seq_len,
  // then we truncate it.
  size_t total_len = len_ids + len_pair_ids + GetNumSpecialTokensToAdd(pair);
  if (max_seq_len > 0 && total_len > max_seq_len) {
    TruncateSequence(&ids, &pair_ids, total_len - max_seq_len);
  }

  // Add special tokens
  vector<int64_t> sequence;
  BuildInputsWithSpecialTokens(&sequence, ids, pair_ids);
  size_t seq_len = sequence.size();
  vector<int64_t> token_type_ids;
  CreateTokenTypeIdsFromSequences(&token_type_ids, ids, pair_ids);

  // Build output dictionary
  encoded_inputs->emplace("input_ids", sequence);
  encoded_inputs->emplace("token_type_ids", token_type_ids);
  // Check lengths
  if (max_seq_len > 0 && seq_len > max_seq_len) {
    VLOG(3) << "There is something wrong with the input sequence length."
               " Please check it.";
    // Failed.
    return 0;
  }

  // Padding
  bool needs_to_be_padded = false;
  if (pad_to_max_seq_len && max_seq_len > 0 && (seq_len < max_seq_len)) {
    needs_to_be_padded = true;
  }

  if (needs_to_be_padded) {
    int64_t difference = static_cast<int64_t>(max_seq_len - seq_len);
    size_t pad_start = max_seq_len - 1 - difference;
    encoded_inputs->at("token_type_ids").resize(max_seq_len);
    for (size_t i = max_seq_len - 1; i > pad_start; i--) {
      encoded_inputs->at("token_type_ids")[i] = pad_token_id_;
    }

    encoded_inputs->at("input_ids").resize(max_seq_len);
    for (size_t i = max_seq_len - 1; i > pad_start; i--) {
      encoded_inputs->at("input_ids")[i] = pad_token_id_;
    }
  }
  return 1;
}

void BertTokenizer::BatchEncode(
    vector<unordered_map<string, vector<int64_t>>>* batch_encode_inputs,
    const Strings& batch_text,
    const Strings& batch_text_pair /* = vector<string>() */,
    bool is_split_into_words /* = false */,
    const size_t max_seq_len /* = 0 */,
    bool pad_to_max_seq_len /* = false */) const {
  bool has_text_pair = false;
  if (batch_text_pair.size() != 0) {
    has_text_pair = true;
  }

  size_t batch_size = batch_text.size();
#ifdef PADDLE_WITH_MKLML
#pragma omp parallel for
#endif
  for (size_t i = 0; i < batch_size; i++) {
    unordered_map<string, vector<int64_t>> res;
    if (has_text_pair) {
      auto status = Encode(&res,
                           batch_text[i],
                           batch_text_pair[i],
                           is_split_into_words,
                           max_seq_len,
                           pad_to_max_seq_len);
      if (!status) {
        res["input_ids"] =
            std::vector<int64_t>{cls_token_id_, sep_token_id_, cls_token_id_};
        res["token_type_ids"] = std::vector<int64_t>{0, 0, 1};
      }
    } else {
      auto status = Encode(&res,
                           batch_text[i],
                           {},
                           is_split_into_words,
                           max_seq_len,
                           pad_to_max_seq_len);

      if (!status) {
        res["input_ids"] = std::vector<int64_t>{cls_token_id_, sep_token_id_};
        res["token_type_ids"] = std::vector<int64_t>{0, 0};
      }
    }
    batch_encode_inputs->at(i) = std::move(res);
  }
}

template <typename T, typename Context>
void FasterTokenizerKernel(const Context& dev_ctx,
                           const phi::ExtendedTensor& vocab_in,
                           const phi::ExtendedTensor& text_in,
                           const paddle::optional<phi::Strings>& text_pair_in,
                           bool do_lower_case,
                           bool is_split_into_words,
                           int max_seq_len,
                           bool pad_to_max_seq_len,
                           DenseTensor* input_ids,
                           DenseTensor* segment_ids) {
  const auto* vocab = reinterpret_cast<const phi::Vocab*>(&vocab_in);
  const auto* text = reinterpret_cast<const Strings*>(&text_in);
  const auto* text_pair =
      reinterpret_cast<const Strings*>(text_pair_in.get_ptr());
  auto* seg_ids = segment_ids;
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
    tokenizer.BatchEncode(&batch_encode_inputs,
                          *text,
                          *text_pair,
                          is_split_into_words,
                          max_seq_len,
                          pad_to_max_seq_len);
  } else {
    tokenizer.BatchEncode(&batch_encode_inputs,
                          *text,
                          Strings(),
                          is_split_into_words,
                          max_seq_len,
                          pad_to_max_seq_len);
  }

  for (size_t i = 0; i < batch_size; ++i) {
    size_t seq_len = batch_encode_inputs[i]["input_ids"].size();
    if (seq_len > batch_max_seq_len) {
      batch_max_seq_len = seq_len;
    }
  }

  input_ids->Resize(
      common::make_ddim({static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(batch_max_seq_len)}));
  auto* input_ids_data = dev_ctx.template Alloc<T>(input_ids);
  seg_ids->Resize(common::make_ddim({static_cast<int64_t>(batch_size),
                                     static_cast<int64_t>(batch_max_seq_len)}));
  auto* seg_ids_data = dev_ctx.template Alloc<T>(seg_ids);

  auto pad_token_id = tokenizer.GetPadTokenID();
  for (size_t i = 0; i < batch_size; i++) {
    auto& encoder_input_ids = batch_encode_inputs[i]["input_ids"];
    auto& encoder_seg_ids = batch_encode_inputs[i]["token_type_ids"];
    const size_t& seq_len = encoder_input_ids.size();
    // Copy the memory
    std::memcpy(input_ids_data + i * batch_max_seq_len,
                encoder_input_ids.data(),
                seq_len * sizeof(T));
    std::memcpy(seg_ids_data + i * batch_max_seq_len,
                encoder_seg_ids.data(),
                seq_len * sizeof(T));
    std::memset(input_ids_data + i * batch_max_seq_len + seq_len,
                pad_token_id,
                (batch_max_seq_len - seq_len) * sizeof(T));
    std::memset(seg_ids_data + i * batch_max_seq_len + seq_len,
                pad_token_id,
                (batch_max_seq_len - seq_len) * sizeof(T));
  }
}
}  // namespace phi

PD_REGISTER_KERNEL(
    faster_tokenizer, CPU, ALL_LAYOUT, phi::FasterTokenizerKernel, int64_t) {}
