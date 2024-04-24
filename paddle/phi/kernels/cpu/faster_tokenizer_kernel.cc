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

#include <glog/logging.h>
#include <utf8proc.h>

#include <codecvt>
#include <iostream>
#include <locale>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "paddle/phi/common/pstring.h"
#include "paddle/phi/core/dense_tensor.h"
#include "paddle/phi/core/extended_tensor.h"
#include "paddle/phi/core/kernel_registry.h"
#include "paddle/phi/core/string_tensor.h"

namespace phi {

template <typename T>
struct PhiVectorType;

template <typename T>
class PhiVector : public phi::ExtendedTensor,
                  public phi::TypeInfoTraits<phi::TensorBase, PhiVector<T>> {
 public:
  PhiVector() = default;

  explicit PhiVector(const std::vector<T>& init_data) : data_(init_data) {}

  PhiVector(PhiVector&& other) = default;

  PhiVector(const PhiVector& other) = default;

  PhiVector& operator=(const PhiVector& other) = default;

  PhiVector& operator=(const std::vector<T>& other) {
    data_ = other;
    return *this;
  }

  PhiVector& operator=(PhiVector&& other) = default;

  /// \brief Destroy the PhiVector and release exclusive resources.
  virtual ~PhiVector() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return PhiVectorType<T>().type_name; }

  size_t size() const { return data_.size(); }

  bool empty() const { return data_.empty(); }

  const T& back() const { return data_.back(); }

  T& back() { return data_.back(); }

  void resize(size_t size) { data_.resize(size); }

  void clear() { data_.clear(); }

  void emplace_back(const T& feed_data) { data_.emplace_back(feed_data); }

  void pop_back() { data_.pop_back(); }

  const T& operator[](size_t index) const { return data_[index]; }

  T& operator[](size_t index) { return data_[index]; }

  T& at(size_t index) { return data_.at(index); }

  const T& at(size_t index) const { return data_.at(index); }

  typename std::vector<T>::iterator begin() { return data_.begin(); }

  typename std::vector<T>::const_iterator begin() const {
    return data_.begin();
  }

  typename std::vector<T>::iterator end() { return data_.end(); }

  typename std::vector<T>::const_iterator end() const { return data_.end(); }

 private:
  std::vector<T> data_;
};

// Note(YuanRisheng): Vocab is mainly used for faster_tokenizer_op and we don't
// recommend widely use it. Because faster_tokenizer_op may be deleted in the
// future and this class will be deleted.

/*
class Vocab : public phi::ExtendedTensor,
              public phi::TypeInfoTraits<phi::TensorBase, Vocab> {
 public:
  Vocab() = default;

  Vocab(Vocab&& other) = default;

  Vocab(const Vocab& other) = default;

  Vocab& operator=(const Vocab& other) = default;

  Vocab& operator=(Vocab&& other) = default;

  Vocab& operator=(
      const std::unordered_map<std::wstring, std::int32_t>& other) {
    this->data_ = other;
    return *this;
  }

  /// \brief Destroy the Vocab and release exclusive resources.
  virtual ~Vocab() = default;

 public:
  /// \brief Returns the name of the class for type traits.
  /// \return The name of the class.
  static const char* name() { return "Vocab"; }

  size_t size() const { return data_.size(); }

  void clear() { data_.clear(); }

  void emplace(const std::wstring& key, std::int32_t value) {
    data_.emplace(key, value);
  }

  std::int32_t at(const std::wstring& key) { return data_.at(key); }

  std::int32_t at(const std::wstring& key) const { return data_.at(key); }

  std::unordered_map<std::wstring, std::int32_t>::iterator find(
      const std::wstring& key) {
    return data_.find(key);
  }

  std::unordered_map<std::wstring, std::int32_t>::const_iterator find(
      const std::wstring& key) const {
    return data_.find(key);
  }

  std::unordered_map<std::wstring, std::int32_t>::iterator begin() {
    return data_.begin();
  }

  std::unordered_map<std::wstring, std::int32_t>::const_iterator begin() const {
    return data_.begin();
  }

  std::unordered_map<std::wstring, std::int32_t>::iterator end() {
    return data_.end();
  }

  std::unordered_map<std::wstring, std::int32_t>::const_iterator end() const {
    return data_.end();
  }

 private:
  std::unordered_map<std::wstring, std::int32_t> data_;
};
*/

// Note(YuanRisheng): PhiVector is essentially a vector that only used for PHI
// Kernel. It can be used when you define a non-tensor type that needs to be
// stored in a vector as PHI kernel argument.

template <>
struct PhiVectorType<std::string> {
  const char* type_name = "PhiVectorString";
};

using String = std::string;
using Strings = PhiVector<std::string>;

// Convert the std::string type to the std::string type.
bool ConvertStrToWstr(const std::string& src, std::wstring* res);
// Convert the std::wstring type to the std::string type.
void ConvertWstrToStr(const std::wstring& src, std::string* res);
// Normalization Form Canonical Decomposition.
void NFD(const std::string& s, std::string* ret);

// Write the data which is type of
// std::unordered_map<td::string, int32_t> to ostream.
void StringMapToStream(std::ostream& os,
                       const std::unordered_map<std::string, int32_t>& data);

// Read the data which is type of
// std::unordered_map<td::string, int32_t> from istream.
void StringMapFromStream(std::istream& is,
                         std::unordered_map<std::string, int32_t>* data);

std::wstring_convert<std::codecvt_utf8<wchar_t>> kConverter;

// Convert the std::string type to the std::wstring type.
bool ConvertStrToWstr(const std::string& src, std::wstring* res) {
  try {
    *res = kConverter.from_bytes(src);
  } catch (std::range_error& e) {
    VLOG(3) << "The string " << src << " was converted to unicode failedly! ";
    return false;
  }
  return true;
}

// Convert the std::wstring type to the std::string type.
void ConvertWstrToStr(const std::wstring& src, std::string* res) {
  *res = kConverter.to_bytes(src);
}

// Normalization Form Canonical Decomposition.
void NFD(const std::string& s, std::string* ret) {
  *ret = "";
  char* result = reinterpret_cast<char*>(
      utf8proc_NFD(reinterpret_cast<const unsigned char*>(s.c_str())));
  if (result) {
    *ret = std::string(result);
    free(result);  // NOLINT
  }
}

// Write the data which is type of
// std::unordered_map<std::string, int32_t> to ostream.
void StringMapToStream(std::ostream& os,
                       const std::unordered_map<std::string, int32_t>& data) {
  {
    // firstly write the data size.
    size_t t = data.size();
    os.write(reinterpret_cast<const char*>(&t), sizeof(t));
  }
  {
    // then write the data
    for (const auto& item : data) {
      std::string token = item.first;
      int32_t token_id = item.second;
      // write the token
      size_t length = token.size();
      os.write(reinterpret_cast<const char*>(&length), sizeof(length));
      os.write(token.c_str(), length);  // NOLINT
      // write the token_id
      os.write(reinterpret_cast<const char*>(&token_id), sizeof(token_id));
    }
  }
}

// Read the data which is type of
// std::unordered_map<td::string, int32_t> from istream.
void StringMapFromStream(std::istream& is,
                         std::unordered_map<std::string, int32_t>* data) {
  // first read the map size
  size_t map_size = 0;
  is.read(reinterpret_cast<char*>(&map_size), sizeof(map_size));
  data->reserve(map_size);
  // then read the data
  for (size_t i = 0; i < map_size; ++i) {
    // read the token
    size_t token_length = 0;
    is.read(reinterpret_cast<char*>(&token_length), sizeof(token_length));
    char* tmp = new char[token_length];
    is.read(tmp, token_length);  // NOLINT
    std::string token(tmp, tmp + token_length);
    delete[] tmp;
    // read the token_id
    int32_t token_id = 0;
    is.read(reinterpret_cast<char*>(&token_id), sizeof(token_id));

    data->emplace(token, token_id);
  }
}

using std::endl;
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
      const phi::Strings& batch_text,
      const phi::Strings& batch_text_pair = phi::Strings(),
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
    const phi::Strings& batch_text,
    const phi::Strings& batch_text_pair /* = vector<string>() */,
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
void FasterTokenizerKernel(
    const Context& dev_ctx,
    const std::unordered_map<std::wstring, std::int>& vocab,
    const std::vector<std::string>& text,
    const std::vector<std::string>& text_pair,
    bool do_lower_case,
    bool is_split_into_words,
    int max_seq_len,
    bool pad_to_max_seq_len,
    DenseTensor* input_ids,
    DenseTensor* segment_ids) {
  auto* seg_ids = segment_ids;

  if (text_pair && text.size() != text_pair.size()) {
    VLOG(3) << "The input text(list[str]) and text pair (list[str]) must"
            << "be the same number of text sequence. Please check the input!";
    return;
  }

  BertTokenizer tokenizer(vocab, do_lower_case);
  size_t batch_max_seq_len = 0;
  size_t batch_size = text.size();

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
                          phi::Strings(),
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
  auto* input_ids_data = input_ids->mutable_data<T>(ctx.GetPlace());
  seg_ids->Resize(common::make_ddim({static_cast<int64_t>(batch_size),
                                     static_cast<int64_t>(batch_max_seq_len)}));
  auto* seg_ids_data = seg_ids->mutable_data<T>(ctx.GetPlace());

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
