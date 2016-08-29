/* Copyright (c) 2016 Baidu, Inc. All Rights Reserve.

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

#include <stack>

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
 * PASS_TEST=0;
 * for (auto& layer : layers){
 *   stack.push(layer->getName());
 *   layer->forward(passType);
 * }
 * for (auto& layer : layers){
 *   layer->backward(passType);
 *   stack.pop(layer->getName());
 * }
 * 
 * if(passType == PASS_TEST) {
 *   stack.clear();
 * }
 * else {
 *   stack.dump([](const std::string& layername){
 *     LOG(INFO) << "LayerName: " << layername;
 *   })
 * }
 * 
 *
 * @endcode
 */
template <typename T>
class CustomStackTrace{
public:
  /**
   * @brief Pop out an item from the top of the stack. For safety the item 
   * will be poped should equal to ip.
   */
  void pop(const T& ip) {
    auto& p = *logstack_;
    CHECK_EQ(ip, p.top());
    p.pop();
  }
  /**
   * @brief Empty the stack by sequence from top to button.
   * @param[in] callback A function deal with each item while dumping.
   * It must have and only have a in parameter which is the stack item.
   */
  template <typename Callback>
  void dump(Callback callback) {
    auto& p = *logstack_;
    while (!p.empty()) {
      callback(p.top());
      p.pop();
    }
  }
  /**
   * @brief Only empty the stack.
   */
  void clear() {
    dump([](const T& ip){});
  }
  /**
   * @brief Push item ip to the top of the stack.
   */
  void push(const T& ip) {
    auto& p = *logstack_;
    p.push(ip);
  }

private:
  ThreadLocalD<std::stack<T> > logstack_;
};

extern CustomStackTrace<std::string> gLayerStackTrace;

}  // namespace paddle
