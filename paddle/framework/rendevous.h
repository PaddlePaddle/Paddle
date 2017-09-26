#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/variable.h"
#include "paddle/platform/device_context.h"

#include <functional>
#include <mutex>
#include <queue>
#include <sstream>
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
  size_t start = 0, next = key.find(';');
  while (next != std::string::npos) {
    pieces.push_back(key.substr(start, next - start + 1));
    start = next;
    next = key.find(';', next + 1);
  }
  pieces.push_back(key.substr(start, key.size() - start + 1));
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
// };

// }

class Rendevous {
 public:
  typedef std::function<void(platform::DeviceContext* src_device,
                             platform::DeviceContext* dst_device, Variable& t)>
      DoneCallback;
  struct Msg {
    Variable var;
    DoneCallback done_cb = nullptr;
    bool RecvReady() { return done_cb != nullptr; }
  };

  void Send(const PairKey& key, platform::DeviceContext& src_device,
            const Variable& t);
  void RecvAsync(const PairKey& key, platform::DeviceContext& dst_device,
                 const DoneCallback& cb);
  // a wrapper on RecvAsync
  void Recv(const PairKey& key, platform::DeviceContext& dst_device,
            Variable* t, int);

 private:
  using BlockingChannel = std::deque<Msg*>;
  using Table = std::unordered_map<size_t, BlockingChannel>;
  Table table_;
  std::mutex mu_;
};

}  // framework
}  // paddle
