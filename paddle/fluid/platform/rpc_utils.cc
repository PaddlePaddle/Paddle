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

#include <algorithm>
#include <codecvt>
#include <locale>
#include <sstream>
#include <unordered_set>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"
#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace platform {

// helpers
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

static inline uint8_t GetByte(const std::string& token) {
  auto num_str = paddle::string::split_string(token, "_")[1];
  num_str = num_str.substr(0, num_str.size() - 1);
  return static_cast<uint8_t>(std::stoi(num_str, nullptr, 16));
}

std::vector<std::string> RecoverBfbTokens(
    const std::vector<std::string>& tokens) {
  std::vector<std::string> new_tokens;
  std::vector<uint8_t> tmp_bytes;
  for (const auto& token : tokens) {
    if (StartsWith(token, "[BFB")) {
      tmp_bytes.emplace_back(GetByte(token));
    } else {
      if (!tmp_bytes.empty()) {
        std::string recovered_token(tmp_bytes.begin(), tmp_bytes.end());
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
  return new_tokens;
}

std::vector<std::string> PostProcess(
    const std::vector<std::string>& tokens,
    const std::unordered_map<std::string, int>& vocab,
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
  auto words = RecoverBfbTokens(tokens);
  for (auto& word : words) {
    if (break_words.find(word) != break_words.end() || word == stop_token) {
      break;
    }
    if (word.empty() || word == "[PAD]") {
      continue;
    }
    if (replace_words.count(word)) {
      new_text.push_back(replace_words.at(word));
      continue;
    }

    static std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
    auto unicode_word = converter.from_bytes(word);
    bool is_chinese_char = IsChineseChar(unicode_word[0]);
    bool is_chinese_punct = IsChinesePunct(unicode_word[0]);

    auto& last_word = new_text.back();
    if (is_chinese_char || is_chinese_punct || !vocab.count(word)) {
      // 当前字符为中文或者中文标点，则直接插入
      if (!new_text.empty() && EndsWith(last_word, "@@")) {
        //// 前一个字符以@@结尾，则去掉
        last_word = Replace(last_word, "@@", "");
      }
      new_text.emplace_back(word);
    } else if (!StartsWith(word, "##")) {
      // 当前字符不以##开头
      if (!new_text.empty() && EndsWith(last_word, "@@")) {
        //// 前一个字符以@@结尾
        last_word = Replace(last_word, "@@", "");
        new_text.emplace_back(word);
      } else if (!new_text.empty() && EndsWith(last_word, "\n")) {
        new_text.emplace_back(word);
      } else {
        const auto unicode_last_word = converter.from_bytes(last_word);
        if (!new_text.empty() && !last_word.empty() &&
            IsChineseChar(unicode_last_word[0])) {
          //// 不以@@结尾，但前一个字符为中文，则不加入空格
          new_text.emplace_back(word);
        } else {
          //// 不以@@结尾，则额外加入空格
          if (!new_text.empty()) {
            new_text.emplace_back(" ");
          }
          new_text.emplace_back(word);
        }
      }
    } else {
      if (!new_text.empty() && EndsWith(last_word, "@@")) {
        //// 前一个字符以@@结尾
        last_word = last_word.substr(0, last_word.size() - 2);
      }
      // 以##开头，则直接加入
      new_text.emplace_back(Replace(word, "##", ""));
    }
  }

  if (!new_text.empty()) {
    auto& last_word = new_text.back();
    last_word = Replace(last_word, "@@", "");
  }

  return new_text;
}

void RpcVocabulary::Init(const std::string& path) {
  if (path_ == path) {
    return;
  }
  std::ifstream vocab_file(path);
  std::string word;
  int id;
  while (vocab_file >> word >> id) {
    ids_to_words_.emplace(id, word);
    words_to_ids_.emplace(word, id);
  }
  path_ = path;
}

std::string RpcVocabulary::Get(std::vector<int> ids,
                               bool aggressive_break,
                               const std::string& stop_token) {
  std::vector<std::string> tokens;
  for (int id : ids) {
    if (!Contains(id)) {
      continue;
    }
    tokens.emplace_back(Get(id));
  }
  return paddle::string::join_strings(
      PostProcess(tokens, words_to_ids_, aggressive_break, stop_token), "");
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
