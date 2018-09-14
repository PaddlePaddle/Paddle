/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <memory>
#include <thread>  // NOLINT
#include "glog/logging.h"
namespace paddle {
namespace framework {
namespace details {

template <class T>
class COWPtr {
 public:
  typedef std::shared_ptr<T> RefPtr;

 private:
  RefPtr m_sp;

  void detach() {
    T* tmp = m_sp.get();
    if (!(tmp == nullptr || m_sp.unique())) {
      VLOG(3) << "Detach " << typeid(T).name();
      m_sp = RefPtr(new T(*tmp));
    }
  }

 public:
  COWPtr() : m_sp(nullptr) {}
  explicit COWPtr(T* t) : m_sp(t) {}
  explicit COWPtr(const RefPtr& refptr) : m_sp(refptr) {}

  const T& Data() const { return operator*(); }

  T* MutableData() { return operator->(); }

  const T& operator*() const { return *m_sp; }
  T& operator*() {
    detach();
    return *m_sp;
  }
  const T* operator->() const { return m_sp.operator->(); }
  T* operator->() {
    detach();
    return m_sp.operator->();
  }
};
}  // namespace details
}  // namespace framework
}  // namespace paddle
