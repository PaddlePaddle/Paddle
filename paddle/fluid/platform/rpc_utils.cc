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
#include <fstream>
#include <regex>
#include <sstream>
#include <unordered_set>

#include "paddle/phi/core/enforce.h"
#include "paddle/phi/core/errors.h"

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

static inline std::string Replace(const std::string& str,
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

static inline std::vector<std::string> Split(std::string split_text,
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

static inline void ToLower(std::wstring* input) {
  for (unsigned int i = 0; i < input->length(); ++i) {
    input->at(i) = towlower(input->at(i));
  }
}

static inline std::wstring SubStr(const std::wstring& input, int a) {
  std::wstring substring;
  for (int i = 0; static_cast<size_t>(i) < input.size() && i < a; i++) {
    substring += input[i];
  }
  return substring;
}

static inline std::vector<std::wstring> WhitespaceTokenize(
    const std::wstring& text) {
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

int RpcCommContext::RpcSend(
    const std::string& url,
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
