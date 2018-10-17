#pragma once

#include <glog/logging.h>
#include <chrono>
#include <limits>
#include <string>
#include <unordered_map>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace platform {

// Timer for timer
// TODO(Superjomn) Clean up the Timers in this project, there are already four
// Timers.
class Timer {
 public:
  std::chrono::high_resolution_clock::time_point start;
  std::chrono::high_resolution_clock::time_point startu;

  void tic() { start = std::chrono::high_resolution_clock::now(); }
  double toc() {
    startu = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> time_span =
        std::chrono::duration_cast<std::chrono::duration<double>>(startu -
                                                                  start);
    double used_time_ms = static_cast<double>(time_span.count()) * 1000.0;
    return used_time_ms;
  }
};

/*
 * A light weight profiler which consumes less performance and memory compared
 * to the Profiler. It allocated the fixed size of memory, so won't OOM. A new
 * timer record's value adds to the existing item directly without storing them.
 *
 * It has no mutex, simple but not thread-safe. One can use it in the
 * concurrency scenerio by creating all the keys first, then it should
 * be safe to use with multiple threads.
 */
class LightTimer {
 public:
  struct Record {
    explicit Record(const std::string &name) : repr(name) {}
    inline void Accumulate(double latency);

    double average() const { return total / count; }

    double cell{0};
    double floor{std::numeric_limits<double>::max()};
    double total{0};
    size_t count{0};
    std::string repr;
  };

  LightTimer(int init_space = 0) { records_.reserve(init_space); }
  ~LightTimer() { LOG(INFO) << "\n" << DebugString(); }

  // Add a new timer record and return record ID.
  size_t NewTimer(const std::string &key) {
    auto it = repr2id_.find(key);
    if (it != std::end(repr2id_)) {
      return it->second;
    }
    records_.emplace_back(key);
    size_t id = records_.size() - 1;
    repr2id_[key] = id;
    return id;
  }

  void AddRecord(size_t id, double latency) {
    PADDLE_ENFORCE_LT(id, records_.size());
    records_[id].Accumulate(latency);
  }

  const Record &GetRecord(size_t id) const {
    PADDLE_ENFORCE_LT(id, records_.size());
    return records_[id];
  }

  static LightTimer &Global() {
    static std::unique_ptr<LightTimer> timer(new LightTimer(10));
    return *timer;
  }

  std::string DebugString() const;

 private:
  std::vector<Record> records_;
  std::unordered_map<std::string, size_t> repr2id_;
};

struct TimerScope {
  TimerScope(LightTimer *root, size_t id) : root_(root), id_(id) {
    timer_.tic();
  }

  ~TimerScope() { root_->AddRecord(id_, timer_.toc()); }

 private:
  LightTimer *root_;
  Timer timer_;
  size_t id_;
};

void LightTimer::Record::Accumulate(double latency) {
  if (cell < latency) {
    cell = latency;
  }
  if (floor > latency) {
    floor = latency;
  }
  total += latency;
  ++count;
}

}  // namespace platform
}  // namespace paddle

#define ADD_ONCE_TIMER(key__)                               \
  ::paddle::platform::TimerScope __timer_scope__##__LINE__( \
      &::paddle::platform::LightTimer::Global(),            \
      ::paddle::platform::LightTimer::Global().NewTimer(key__));

#define ADD_TIMER(key__, timer_ptr__)                       \
  ::paddle::platform::TimerScope __timer_scope__##__LINE__( \
      timer_ptr__, timer_ptr__->NewTimer(key__));
