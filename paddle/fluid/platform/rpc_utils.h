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

#pragma once

#include <brpc/channel.h>
#include <bthread/countdown_event.h>

#include <atomic>
#include <codecvt>
#include <locale>
#include <memory>
#include <string>
#include <unordered_map>

#include "paddle/utils/string/string_helper.h"

namespace paddle {
namespace platform {

using WordToIdMap = std::unordered_map<std::wstring, int64_t>;
using IdToWordMap = std::unordered_map<int64_t, std::string>;

class RpcTokenizer {
 public:
  static RpcTokenizer &Instance() {
    static RpcTokenizer instance;
    return instance;
  }

  void Init(const std::string &path);

  void Init(const std::string &path,
            const std::unordered_map<std::string, std::string> &special_set);

  void SetSpecialSet(
      const std::unordered_map<std::string, std::string> &special_set) {
    special_set_ = special_set;
  }

  bool Contains(int64_t id) { return ids_to_words_.count(id) > 0; }

  // NOTE: an exception will be raised if id not exist
  std::string GetWordFromId(int64_t id) {
    auto q = ids_to_words_.at(id);
    if (special_set_.count(q) == 1) {
      return special_set_.at(q);
    } else {
      return q;
    }
  }

  template <typename T = int64_t>
  std::string GetWordsFromIds(const std::vector<T> &ids,
                              bool aggressive_break = false,
                              const std::string &stop_token = "[gEND]") {
    std::vector<std::string> tokens;
    for (auto id : ids) {
      if (!Contains(id)) {
        continue;
      }
      tokens.emplace_back(GetWordFromId(id));
    }
    return paddle::string::join_strings(
        PostProcess(tokens, words_to_ids_, aggressive_break, stop_token), "");
  }

  // NOTE: an exception will be raised if word not exist
  int64_t GetIdFromWord(const std::wstring &word) {
    return words_to_ids_.at(word);
  }

 private:
  std::string GetRecoveredToken(const std::vector<uint8_t> &bytes);

  std::vector<std::string> RecoverBFBTokens(
      const std::vector<std::string> &tokens);

  std::vector<std::string> PostProcess(
      const std::vector<std::string> &tokens,
      const WordToIdMap &vocab,
      bool aggressive_break = false,
      const std::string &stop_token = "[gEND]");

 private:
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter_;

  std::string path_;
  IdToWordMap ids_to_words_;
  WordToIdMap words_to_ids_;
  std::unordered_map<std::string, std::string> special_set_;
};

class RpcRequestStore {
 public:
  static RpcRequestStore &Instance() {
    static RpcRequestStore instance;
    return instance;
  }

  int GetRequestId() {
    if (request_id_ == INT32_MAX) {
      request_id_ = 0;
    } else {
      ++request_id_;
    }
    return request_id_;
  }

  std::shared_ptr<bthread::CountdownEvent> GetEvent(int request_id) {
    return id_to_event_map_[request_id];
  }

  int GetErrorCode(int request_id) { return id_to_err_map_[request_id]; }

  std::string GetResponse(int request_id) {
    return id_to_resp_map_[request_id];
  }

  void InsertEvent(int request_id,
                   const std::shared_ptr<bthread::CountdownEvent> &event) {
    if (request_id == 0) {
      LOG(WARNING) << "Total num of requests have exceeded int limits.";
    }
    id_to_event_map_.emplace(request_id, event);
  }

  void InsertErrorCode(int request_id, int error_code) {
    if (request_id == 0) {
      LOG(WARNING) << "Total num of requests have exceeded int limits.";
    }
    id_to_err_map_.emplace(request_id, error_code);
  }

  void InsertResponse(int request_id, const std::string &resp) {
    if (request_id == 0) {
      LOG(WARNING) << "Total num of requests have exceeded int limits.";
    }
    id_to_resp_map_.emplace(request_id, resp);
  }

 private:
  std::atomic<int> request_id_;

  std::unordered_map<int, std::shared_ptr<bthread::CountdownEvent>>
      id_to_event_map_;
  std::unordered_map<int, int> id_to_err_map_;
  std::unordered_map<int, std::string> id_to_resp_map_;
};

struct RpcCommContext {
  static int RpcSend(
      const std::string &url,
      const std::string &query,
      void (*request_handler)(brpc::Controller *, int, const std::string &),
      void (*response_handler)(brpc::Controller *,
                               int,
                               std::shared_ptr<bthread::CountdownEvent>),
      brpc::HttpMethod http_method = brpc::HttpMethod::HTTP_METHOD_POST,
      int timeout_ms = 10000,
      int max_retry = 3);
};

}  // namespace platform
}  // namespace paddle
