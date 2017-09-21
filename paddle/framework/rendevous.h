#pragma once
#include "paddle/framework/eigen.h"
#include "paddle/framework/op_registry.h"
#include "paddle/framework/variable.h"
#include "paddle/platform/device_context.h"

#include <functional>
#include <mutex>
#include <queue>
#include <unordered_map>

namespace paddle {
namespace framework {

struct PairKey {
  std::string src_device;
  std::string dst_device;
  int edge_id;  // a unique id in graph
};

class MarkNode {
 public:
  static MarkNode& Get {
    if (buf_ == nullptr) {
      buf_ = new MarkNode;
    }
    return *buf_;
  }

 private:
  MarkNode* buf_ = nullptr;
}

class Rendevous {
 public:
  typedef std::function<void(DefaultDevice* src_device,
                             DefaultDevice* dst_device, const Variable& t,
                             int* status)>
      DoneCallback;
  struct Msg {
    Variable var;
    DoneCallback done_cb = nullptr;
    bool is_done = false;
  };

  static PairKey CreateKey(std::string src_device, std::string dst_device,
                           int edge_id);

 private:
  typedef std::deque<Msg*> Channel;
  typedef std::unordered_map<int64, Channel> Table;
  Table table_;
  std::mutex mu_;
};
}
}
