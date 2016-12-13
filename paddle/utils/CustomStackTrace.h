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

#include <functional>
#include <stack>
#include <thread>
#include <unordered_map>

#include "ThreadLocal.h"

namespace paddle {

/**
 * A ThreadLocal stack for tracing train/test process.
 * (More details of ThreadLocal can be find
 * in the comments of ThreadLocal class.)
 *
 * For example.
 * @code{.cpp}
 *
 * paddle::CustomStackTrace<std::string> stack;
 * for (auto& layer : layers){
 *   stack.push(layer->getName());
 *   layer->forward();
 * }
 *
 * stack.pop("");  // mark under pop stage.
 *
 * for (auto it = layers.rbegin(); it != layers.rend(); ++it){
 *   auto& layer = *it;
 *   layer->backward(passType);
 *   stack.pop(layer->getName());
 * }
 *
 * @endcode
 */
template <typename T>
class CustomStackTrace {
public:
  /**
   * @brief Pop out an item from the top of the stack if item == top.
   *        Else, just set status to popping.
   */
  void pop(const T& item) {
    pushing() = false;
    auto& s = this->stack();
    if (item == s.top()) {
      s.pop();
    }
  }

  /**
   * @brief clear current thread stack.
   */
  void clear() {
    auto& s = stack();
    while (!s.empty()) {
      s.pop();
    }
  }

  /**
   * @brief return true if all thread's stack is empty.
   * @return true if empty
   */
  bool empty() const {
    std::lock_guard<std::mutex> g(this->mtx_);
    for (auto p : this->stackBuffers_) {
      std::stack<T>& s = *p.second;
      if (!s.empty()) {
        return false;
      }
    }
    return true;
  }

  /**
   * @brief DumpCallback Type. It will be invoked many times by dump method.
   *
   * The first parameter is stack thread id.
   * The second parameter is the last action of stack is push or not.
   * The third parameter is the item in stack.
   */
  typedef std::function<void(const std::thread::id& /*threadId*/,
                             bool* /*isPushing*/,
                             const T& /*item*/)>
      DumpCallback;

  /**
   * Dump all thread stack, and all stack will be cleared.
   */
  void dump(const DumpCallback& callback, bool onlyCurrentThread = false) {
    std::lock_guard<std::mutex> g(this->mtx_);
    for (auto p : this->stackBuffers_) {
      std::thread::id tid = p.first;
      if (onlyCurrentThread && tid != std::this_thread::get_id()) {
        continue;
      }
      std::stack<T>& s = *p.second;
      bool* isPush = nullptr;
      auto it = this->pushingBuffers_.find(tid);
      if (it != this->pushingBuffers_.end()) {
        isPush = it->second;
      }

      while (!s.empty()) {
        callback(tid, isPush, s.top());
        s.pop();
      }
    }
  }

  /**
   * @brief Push item to current thread stack.
   */
  void push(const T& item) {
    pushing() = true;
    auto& p = this->stack();
    p.push(item);
  }

private:
  /**
   * Get thread local attribute, and save them into a map (threadId => TYPE*)
   *
   * @tparam TYPE thread local attribute type.
   * @param threadLocal Thread Local object.
   * @param buffers a map from threadId to TYPE*
   */
  template <typename TYPE>
  inline TYPE& getThreadLocal(
      ThreadLocal<TYPE>& threadLocal,
      std::unordered_map<std::thread::id, TYPE*>& buffers) {
    TYPE* retv = threadLocal.get(false);
    if (retv) {
      return *retv;
    } else {
      std::lock_guard<std::mutex> guard(this->mtx_);
      retv = threadLocal.get();
      auto id = std::this_thread::get_id();
      buffers.insert({id, retv});
      return *retv;
    }
  }

  /**
   * @brief Get thread local stack reference.
   */
  std::stack<T>& stack() {
    return this->getThreadLocal(this->logStack_, this->stackBuffers_);
  }

  /**
   * @brief Get thread local pushing flag.
   */
  bool& pushing() {
    return this->getThreadLocal(this->isPushing_, this->pushingBuffers_);
  }

private:
  mutable std::mutex mtx_;

  std::unordered_map<std::thread::id, std::stack<T>*> stackBuffers_;
  std::unordered_map<std::thread::id, bool*> pushingBuffers_;
  ThreadLocal<bool> isPushing_;
  ThreadLocal<std::stack<T>> logStack_;
};

extern CustomStackTrace<std::string> gLayerStackTrace;

/**
 * @brief Install a failure handler to print layer stack when error.
 */
extern void installLayerStackTracer();

}  // namespace paddle
