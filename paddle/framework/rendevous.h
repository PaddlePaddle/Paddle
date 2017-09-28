/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserve.

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
#include "paddle/framework/eigen.h"
#include "paddle/framework/lod_tensor.h"
#include "paddle/framework/op_registry.h"
#include "paddle/platform/device_context.h"

#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <sstream>
#include <string>
#include <unordered_map>

namespace paddle {
namespace framework {

struct PairKey {
  std::string src_name;
  std::string dst_name;
  // int edge_id;  // a unique id in graph
  int src_id;
  int dst_id;
};

// static std::string CreateKey(std::string src_name, std::string dst_name,
//                              int src_id, int dst_id) {
static std::string CreateKey(const PairKey& key) {
  std::stringstream ss;
  ss << key.src_name << ";" << key.dst_name << ";" << key.src_id << ";"
     << key.dst_id;
  return ss.str();
}
static PairKey ParseKey(const std::string& key) {
  // FIXME(dzhwinter): use piece string for efficiency
  std::vector<std::string> pieces;
  size_t start = -1, next = key.find(';');
  while (next != std::string::npos) {
    pieces.push_back(key.substr(start + 1, next - start));
    start = next;
    next = key.find(';', next + 1);
  }
  pieces.push_back(key.substr(start, key.size() - start + 1));
  for (auto& part : pieces) {
    LOG(INFO) << part << " ";
  }
  PADDLE_ENFORCE(4 == pieces.size(), "need fullkey");
  PairKey ret;
  ret.src_name = pieces[0];
  ret.dst_name = pieces[1];
  ret.src_id = std::stoi(pieces[2]);
  ret.dst_id = std::stoi(pieces[3]);
  return ret;
}

size_t Hash64(const PairKey& key) {
  auto s = CreateKey(key);
  return std::hash<std::string>{}(s);
}

// namespace std {

// template<>
// struct ::std::hash<PairKey> {
//   size_t operator() (const PairKey& key) const {
//     auto s = CreateKey(key);
//     return std::hash<std::string>{}(s);
//   }
// }strin;

// }

class Rendevous {
 public:
  typedef std::function<void(const platform::DeviceContext& src_device,
                             const platform::DeviceContext& dst_device,
                             const Variable& t)>
      DoneCallback;
  struct Msg {
    // Msg() {}
    // Msg(const Variable& v) : var(v) {}
    // Msg(const Variable, const DoneCallback& cb) : var(v), done_cb(cb) {}
    // TODO(dzhwinter): should be any type, namely. Variable
    Tensor var;
    platform::DeviceContext* src_device = nullptr;
    platform::DeviceContext* dst_device = nullptr;
    DoneCallback done_cb = nullptr;
    bool RecvReady() { return done_cb != nullptr && dst_device != nullptr; }
  };

  void Send(const PairKey& key, const platform::DeviceContext& src_device,
            const Variable& t);
  void RecvAsync(const PairKey& key, const platform::DeviceContext& dst_device,
                 const DoneCallback& cb);
  // a wrapper on RecvAsync
  void Recv(const PairKey& key, const platform::DeviceContext& dst_device,
            Variable* t, int timeout_ms);

 private:
  using BlockingChannel = std::deque<std::unique_ptr<Msg>>;
  using Table = std::unordered_map<size_t, BlockingChannel>;
  Table table_;
  std::mutex mu_;
};

}  // namespace framework
}  // namespace paddle
