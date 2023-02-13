// Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/platform/rpc_utils.h"

#include <unicode/normlzr.h>
#include <unicode/platform.h>
#include <unicode/uchar.h>
#include <unicode/uconfig.h>
#include <unicode/unistr.h>

#include <algorithm>
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_set>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace platform {

// globals
static std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;

// utils
static inline bool StartsWith(const std::string& str,
                              const std::string& prefix) {
  return str.substr(0, prefix.length()) == prefix;
}

static inline bool EndsWith(const std::string& str, const std::string& suffix) {
  return str.length() >= suffix.length() &&
         str.substr(str.length() - suffix.length()) == suffix;
}

static std::string Replace(const std::string& str,
                           const std::string& old_str,
                           const std::string& new_str) {
  if (old_str == new_str) {
    return str;
  }
  std::stringstream ss;
  size_t start_pos = 0;
  size_t pos = str.find(old_str, start_pos);
  while (pos != std::string::npos) {
    ss << str.substr(start_pos, pos - start_pos) << new_str;
    start_pos = pos + old_str.size();
    pos = str.find(old_str, start_pos);
  }
  ss << str.substr(start_pos);
  return ss.str();
}

static std::vector<std::string> Split(std::string split_text,
                                      std::regex pattern) {
  std::vector<std::string> output;
  std::sregex_token_iterator substrings(
      split_text.begin(), split_text.end(), pattern, -1);
  std::sregex_token_iterator delimiters(
      split_text.begin(), split_text.end(), pattern, 0);
  std::sregex_token_iterator end;
  while (substrings != end && delimiters != end) {
    output.emplace_back(*substrings++);
    output.emplace_back(*delimiters++);
  }
  if (substrings != end) {
    output.emplace_back(*substrings++);
  }
  return output;
}

static void ToLower(std::wstring* input) {
  for (unsigned int i = 0; i < input->length(); ++i) {
    input->at(i) = towlower(input->at(i));
  }
}

static std::wstring SubStr(const std::wstring& input, int a) {
  std::wstring substring;
  for (int i = 0; static_cast<size_t>(i) < input.size() && i < a; i++) {
    substring += input[i];
  }
  return substring;
}

static std::vector<std::wstring> WhitespaceTokenize(const std::wstring& text) {
  std::vector<std::wstring> tokens;
  std::wstringstream ss(text);
  std::wstring token;
  while (ss >> token) {
    tokens.emplace_back(token);
  }
  return tokens;
}

static inline bool IsWhitespace(const wchar_t& c) {
  if (c == L' ' | c == L'\t' || c == L'\n' || c == L'\r') {
    return true;
  }
  int8_t category = u_charType(UChar32(c));
  return (U_MASK(category) & U_GC_ZS_MASK);
}

static inline bool IsControl(const wchar_t& c) {
  if (c == L'\t' || c == L'\n' || c == L'\r') {
    return false;
  }
  int8_t category = u_charType(UChar32(c));
  return (U_MASK(category) & U_GC_C_MASK);
}

static inline bool IsPunct(const wchar_t& c) {
  uint32_t cp = uint32_t(c);
  if ((cp >= 33 && cp <= 47) || (cp >= 58 && cp <= 64) ||
      (cp >= 91 && cp <= 96) || (cp >= 123 && cp <= 126)) {
    return true;
  }
  int8_t category = u_charType(UChar32(c));
  return (U_MASK(category) & U_GC_P_MASK);
}

static inline bool IsChineseChar(wchar_t c) {
  return (c >= 0x4E00 && c <= 0x9FFF) || (c >= 0x3400 && c <= 0x4DBF) ||
         (c >= 0x20000 && c <= 0x2A6DF) || (c >= 0x2A700 && c <= 0x2B73F) ||
         (c >= 0x2B740 && c <= 0x2B81F) || (c >= 0x2B820 && c <= 0x2CEAF) ||
         (c >= 0xF900 && c <= 0xFAFF) || (c >= 0x2F800 && c <= 0x2FA1F);
}

static inline bool IsChinesePunct(wchar_t c) {
  std::unordered_set<wchar_t> puncts = {
      L'！', L'？', L'｡',  L'。', L'＂', L'＃', L'＄', L'％', L'＆', L'＇',
      L'（', L'）', L'＊', L'＋', L'，', L'－', L'／', L'：', L'；', L'＜',
      L'＝', L'＞', L'＠', L'［', L'＼', L'］', L'＾', L'＿', L'｀', L'｛',
      L'｜', L'｝', L'～', L'｟', L'｠', L'｢',  L'｣',  L'､',  L'、', L'〃',
      L'》', L'「', L'」', L'『', L'』', L'【', L'】', L'〔', L'〕', L'〖',
      L'〗', L'〘', L'〙', L'〚', L'〛', L'〜', L'〝', L'〞', L'〟', L'〰',
      L'〾', L'〿',  L'–',  L'—',  L'“',  L'”',  L'‘',  L'’'};
  return puncts.count(c);
}

static inline int GetCharBytes(uint8_t byte) {
  if ((byte & 0x80) == 0) {
    return 1;
  } else if ((byte & 0xE0) == 0xC0) {
    return 2;
  } else if ((byte & 0xF0) == 0xE0) {
    return 3;
  } else if ((byte & 0xF8) == 0xF0) {
    return 4;
  } else {
    return -1;
  }
}

static inline bool IsValidContinuationByte(uint8_t byte) {
  // check if the byte starts with 10
  return (byte & 0xC0) == 0x80;
}

static inline uint8_t GetByteFromHex(const std::string& token) {
  auto num_str = paddle::string::split_string(token, "_")[1];
  num_str = num_str.substr(0, num_str.size() - 1);
  return static_cast<uint8_t>(std::stoi(num_str, nullptr, 16));
}

static inline std::string ByteToHex(uint8_t c) {
  const std::string& hex = "0123456789abcdef";
  int val = static_cast<int>(c);
  std::string res = "0x";
  res += hex[val >> 4];
  res += hex[val & 0x0F];
  return res;
}

// BasicTokenizer
std::wstring StripAccents(const std::wstring& text) {
  UErrorCode err = U_ZERO_ERROR;
  auto* normalizer = icu::Normalizer2::getNFDInstance(err);
  if (U_FAILURE(err)) {
    PADDLE_THROW(phi::errors::Unavailable("Cannot init unicode normalizer."));
  }

  icu::UnicodeString unicode_text =
      icu::UnicodeString::fromUTF8(converter.to_bytes(text));
  unicode_text = normalizer->normalize(unicode_text, err);
  if (U_FAILURE(err)) {
    PADDLE_THROW(phi::errors::InvalidArgument("Cannot normalize the input."));
  }

  icu::UnicodeString normalized_text;
  int sz = unicode_text.length();
  for (int i = 0; i < sz; ++i) {
    normalized_text += unicode_text[i];
  }

  std::string res;
  unicode_text.toUTF8String(res);
  return converter.from_bytes(res);
}

std::wstring SplitOnPunc(const std::wstring& text) {
  std::wstring output;
  std::wstring s;
  int i = 0;
  int n = text.size();
  while (i < n) {
    wchar_t c = text[i];
    if (IsPunct(c)) {
      if (s.size() > 0) {
        output += s;
        output += L' ';
        s.clear();
      }
      if (IsChinesePunct(c)) {
        s += c;
        s += L' ';
      } else if (i + 1 == n) {
        if (n == 1) {
          s += c;
        } else {
          s += L"##";
          s += c;
        }
      } else {
        if (i == 0) {
          s += c;
          s += L"@@ ";
        } else {
          s += L"##";
          s += c;
          s += L"@@ ";
        }
      }
    } else {
      s += c;
    }
    i += 1;
  }
  if (s.size() > 0) {
    output += L' ';
    output += s;
    s.clear();
  }
  return output;
}

std::wstring TokenizeChineseChars(const std::wstring& text) {
  std::wstring output;
  for (const wchar_t& c : text) {
    if (IsChineseChar(c)) {
      output += L' ';
      output += c;
      output += L' ';
    } else {
      output += c;
    }
  }
  return output;
}

std::wstring CleanText(const std::wstring& text) {
  std::wstring output;
  for (const wchar_t& c : text) {
    if (c == 0 || c == 0xfffd || IsControl(c)) {
      continue;
    }
    if (IsWhitespace(c)) {
      output += L' ';
    } else {
      output += c;
    }
  }
  return output;
}

std::vector<std::wstring> BasicTokenizer::Tokenize(const std::wstring& text) {
  auto cleaned_text = CleanText(text);
  auto chinese_tokenized_text = TokenizeChineseChars(cleaned_text);
  auto orig_tokens = WhitespaceTokenize(chinese_tokenized_text);

  std::wstring output_string;
  for (auto& token : orig_tokens) {
    if (do_lower_case_) {
      ToLower(&token);
    }
    auto nfd_token = StripAccents(token);
    if (output_string.size() > 0) {
      output_string += L' ';
    }
    output_string += SplitOnPunc(nfd_token);
  }
  return WhitespaceTokenize(output_string);
}

// WordpieceTokenizer
std::vector<std::wstring> WordpieceTokenizer::Tokenize(
    const std::wstring& text) {
  std::vector<std::wstring> output_tokens;

  for (auto& token : WhitespaceTokenize(text)) {
    int n = token.size();
    if (n > max_chars_per_word_) {
      output_tokens.emplace_back(unk_token_);
      continue;
    }

    std::vector<std::wstring> sub_tokens;
    std::vector<std::wstring> unk_tokens;

    int start = 0;
    while (start < n) {
      bool is_bad = false;
      int end = n;
      std::wstring cur_substr;
      while (start < end) {
        std::wstring substr = token.substr(start, end - start);
        if (start > 0) {
          substr = L"##" + substr;
        }
        if (vocab_.count(substr) > 0) {
          cur_substr = substr;
          break;
        }
        --end;
      }
      // added 2nd search start
      if (cur_substr.empty()) {
        end = n;
        while (start < end) {
          std::wstring substr = token.substr(start, end - start);
          if (vocab_.count(substr) > 0) {
            cur_substr = substr;
            break;
          } else if (end == start + 1) {
            unk_tokens.clear();
            is_bad = true;
            for (auto& sub_byte : converter.to_bytes(substr)) {
              std::string unk_token_str = "[BFB_" + ByteToHex(sub_byte) + "]";
              const std::wstring unk_token =
                  converter.from_bytes(unk_token_str);
              if (vocab_.count(unk_token) > 0) {
                unk_tokens.emplace_back(unk_token);
              }
            }
            break;
          }
          --end;
        }
      }
      // end
      if (is_bad && !unk_tokens.empty()) {
        sub_tokens.insert(
            sub_tokens.end(), unk_tokens.begin(), unk_tokens.end());
      } else {
        sub_tokens.emplace_back(cur_substr);
      }
      start = end;
    }
    output_tokens.insert(
        output_tokens.end(), sub_tokens.begin(), sub_tokens.end());
  }
  return output_tokens;
}

// FullTokenizer
std::vector<std::wstring> FullTokenizer::Tokenize(const std::string& text) {
  std::vector<std::wstring> split_tokens;
  if (text.empty()) {
    return split_tokens;
  }

  std::string processed_text =
      std::regex_replace(text, std::regex("\n"), "[<N>]");
  processed_text =
      std::regex_replace(processed_text, std::regex("\t"), "[<T>]");
  processed_text =
      std::regex_replace(processed_text, std::regex(R"(\s{4})"), "[<T>]");
  processed_text =
      std::regex_replace(processed_text, std::regex(R"(\s{2})"), "[<t>]");

  // cout << processed_text << endl;
  std::regex seperator("\\[<(T|N|t)>\\]");
  std::vector<std::string> text_ls = Split(processed_text, seperator);
  std::string missed_token = "[<S>]";
  std::string unknown_token = "[BFB_";
  auto decoded_missed_token = converter.from_bytes(missed_token);
  auto decoded_unknwon_token = converter.from_bytes(unknown_token);
  for (const std::string& sent : text_ls) {
    if (sent == "[<N>]" || sent == "[<T>]" || sent == "[<t>]") {
      split_tokens.push_back(converter.from_bytes(sent));
    } else {
      auto decoded_sent = converter.from_bytes(sent);
      auto tokens = basic_tokenizer_.Tokenize(decoded_sent);
      for (const std::wstring& token : tokens) {
        auto sub_tokens = wordpiece_tokenizer_.Tokenize(token);

        if (!split_tokens.empty() &&
            SubStr(split_tokens.back(), 5) == decoded_unknwon_token &&
            sub_tokens.front().substr(0, 5) == decoded_unknwon_token) {
          split_tokens.push_back(decoded_missed_token);
        }
        for (const std::wstring& sub_token : sub_tokens) {
          split_tokens.push_back(sub_token);
        }
      }
    }
  }

  return split_tokens;
}

// RpcTokenizer
void RpcTokenizer::Init(const std::string& path) {
  if (path_ == path) {
    return;
  }
  std::ifstream vocab_file(path);
  std::string word;
  int id;
  while (vocab_file >> word >> id) {
    ids_to_words_.emplace(id, word);
    words_to_ids_.emplace(converter.from_bytes(word), id);
  }
  // update tokenizer
  tokenizer_.SetVocab(words_to_ids_);
  // update members
  path_ = path;
}

void RpcTokenizer::Init(
    const std::string& path,
    const std::unordered_map<std::string, std::string>& special_set) {
  if (path_ == path) {
    return;
  }
  Init(path);
  SetSpecialSet(special_set);
}

static inline std::string GetRecoveredToken(const std::vector<uint8_t>& bytes) {
  std::string res;
  int n = bytes.size();
  int i = 0;
  while (i < n) {
    int sz = 0;
    while ((sz = GetCharBytes(bytes[i])) == -1) {
      ++i;
    }
    if (i + sz < n) {
      std::vector<uint8_t> valid_bytes;
      valid_bytes.emplace_back(bytes[i]);
      for (int j = 1; j < sz; ++j) {
        if (!IsValidContinuationByte(bytes[i])) {
          break;
        }
        valid_bytes.emplace_back(bytes[i]);
        ++i;
      }
      if (valid_bytes.size() == static_cast<size_t>(sz)) {
        res += std::string(valid_bytes.begin(), valid_bytes.end());
      }
    }
    ++i;
  }
  return res;
}

std::vector<std::string> RecoverBFBTokens(
    const std::vector<std::string>& tokens) {
  std::vector<std::string> new_tokens;
  std::vector<uint8_t> tmp_bytes;
  for (const auto& token : tokens) {
    if (StartsWith(token, "[BFB")) {
      tmp_bytes.emplace_back(GetByteFromHex(token));
    } else {
      if (!tmp_bytes.empty()) {
        // since there may be illegal bytes, we need this function
        // if all bytes are legal, we can simply use string constructor
        const std::string recovered_token = GetRecoveredToken(tmp_bytes);
        if (!recovered_token.empty()) {
          new_tokens.emplace_back(recovered_token);
        }
      }
      if (token != "[UNK]") {
        new_tokens.emplace_back(token);
      }
      tmp_bytes.clear();
    }
  }
  if (!tmp_bytes.empty()) {
    const std::string recovered_token = GetRecoveredToken(tmp_bytes);
    if (!recovered_token.empty()) {
      new_tokens.emplace_back(recovered_token);
    }
  }
  return new_tokens;
}

std::vector<std::string> PostProcess(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::wstring, int>& vocab,
    bool aggressive_break = false,
    const std::string& stop_token = "[gEND]") {
  std::unordered_set<std::string> break_words;
  if (aggressive_break) {
    break_words = {"[END]", "[gEND]", "[<S>]", "[UNK]", "[CLS]"};
  } else {
    break_words = {"[END]", "[gEND]"};
  }
  static const std::unordered_map<std::string, std::string> replace_words{
      {"[<S>]", " "},
      {"[<N>]", "\n"},
      {"[<T>]", "\t"},
      {"[<t>]", "  "},
  };

  std::vector<std::string> new_text;
  auto words = RecoverBFBTokens(tokens);
  for (auto& word : words) {
    if (break_words.count(word) || word == stop_token) {
      break;
    }
    if (word.empty() || word == "[PAD]") {
      continue;
    }
    if (replace_words.count(word)) {
      new_text.emplace_back(replace_words.at(word));
      continue;
    }

    auto unicode_word = converter.from_bytes(word);
    bool is_chinese_char = IsChineseChar(unicode_word[0]);
    bool is_chinese_punct = IsChinesePunct(unicode_word[0]);

    if (is_chinese_char || is_chinese_punct || vocab.count(unicode_word) == 0) {
      if (!new_text.empty() && EndsWith(new_text.back(), "@@")) {
        auto& last_word = new_text.back();
        last_word = Replace(last_word, "@@", "");
      }
      new_text.emplace_back(word);
    } else if (!StartsWith(word, "##")) {
      if (!new_text.empty() && EndsWith(new_text.back(), "@@")) {
        auto& last_word = new_text.back();
        last_word = Replace(last_word, "@@", "");
        new_text.emplace_back(word);
      } else if (!new_text.empty() && EndsWith(new_text.back(), "\n")) {
        new_text.emplace_back(word);
      } else {
        if (!new_text.empty() && !new_text.back().empty() &&
            IsChineseChar(converter.from_bytes(new_text.back())[0])) {
          new_text.emplace_back(word);
        } else {
          if (!new_text.empty()) {
            new_text.emplace_back(" ");
          }
          new_text.emplace_back(word);
        }
      }
    } else {
      if (!new_text.empty() && EndsWith(new_text.back(), "@@")) {
        auto& last_word = new_text.back();
        last_word = last_word.substr(0, last_word.size() - 2);
      }
      new_text.emplace_back(Replace(word, "##", ""));
    }
  }

  if (!new_text.empty()) {
    auto& last_word = new_text.back();
    last_word = Replace(last_word, "@@", "");
  }

  return new_text;
}

std::string RpcTokenizer::GetWordsFromIds(std::vector<int> ids,
                                          bool aggressive_break,
                                          const std::string& stop_token) {
  std::vector<std::string> tokens;
  for (int id : ids) {
    if (!Contains(id)) {
      continue;
    }
    tokens.emplace_back(GetWordFromId(id));
  }
  return paddle::string::join_strings(
      PostProcess(tokens, words_to_ids_, aggressive_break, stop_token), "");
}

std::vector<int> RpcTokenizer::GetIdsFromText(const std::string& text) {
  std::vector<int> ids;
  auto tokens = tokenizer_.Tokenize(text);
  for (const auto& token : tokens) {
    ids.emplace_back(GetIdFromWord(token));
  }
  return ids;
}

int RpcSend(const std::string& url,
            const std::string& query,
            void (*request_handler)(brpc::Controller*, int, const std::string&),
            void (*response_handler)(brpc::Controller*,
                                     int,
                                     std::shared_ptr<bthread::CountdownEvent>),
            brpc::HttpMethod http_method,
            int timeout_ms,
            int max_retry) {
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.protocol = "http";
  options.timeout_ms = timeout_ms;
  options.max_retry = max_retry;
  PADDLE_ENFORCE_EQ(
      channel.Init(url.c_str(), /*load_balancer*/ "", &options),
      0,
      phi::errors::Unavailable("Rpc send failed: init brpc channel error."));

  auto& rpc_store = RpcRequestStore::Instance();
  int request_id = rpc_store.GetRequestId();

  auto event = std::make_shared<bthread::CountdownEvent>();
  RpcRequestStore::Instance().InsertEvent(request_id, event);

  // if req is async, controller should be on heap to avoid deleting
  auto* ctrl = new brpc::Controller();
  ctrl->http_request().uri() = url.c_str();
  ctrl->http_request().set_method(http_method);
  ctrl->http_request().SetHeader("Content-Type", "application/json");
  request_handler(ctrl, request_id, query);

  channel.CallMethod(
      nullptr,
      ctrl,
      nullptr,
      nullptr,
      brpc::NewCallback(response_handler, ctrl, request_id, event));

  return request_id;
}

}  // namespace platform
}  // namespace paddle
