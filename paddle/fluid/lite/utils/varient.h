// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
#include <glog/logging.h>
#include <exception>
#include <memory>
#include <type_traits>
#include <typeinfo>

// This is an equivalent implementation of boost::any. We implement this to
// avoid including the whole boost library and keep the inference library small.
// These code references https://gist.github.com/shoooe/9202235

namespace paddle {
namespace lite {

template <size_t arg1, size_t... others>
struct static_max;
template <size_t arg>
struct static_max<arg> {
  static const size_t value = arg;
};
template <size_t arg1, size_t arg2, size_t... others>
struct static_max<arg1, arg2, others...> {
  static const size_t value = arg1 >= arg2 ? static_max<arg1, others...>::value
                                           : static_max<arg2, others...>::value;
};
template <typename... Ts>
struct variant_helper;
template <typename F, typename... Ts>
struct variant_helper<F, Ts...> {
  inline static void destroy(size_t id, void* data) {
    if (id == typeid(F).hash_code())
      reinterpret_cast<F*>(data)->~F();
    else
      variant_helper<Ts...>::destroy(id, data);
  }
  inline static void move(size_t old_t, void* old_v, void* new_v) {
    if (old_t == typeid(F).hash_code())
      new (new_v) F(std::move(*reinterpret_cast<F*>(old_v)));
    else
      variant_helper<Ts...>::move(old_t, old_v, new_v);
  }
  inline static void copy(size_t old_t, const void* old_v, void* new_v) {
    if (old_t == typeid(F).hash_code())
      new (new_v) F(*reinterpret_cast<const F*>(old_v));
    else
      variant_helper<Ts...>::copy(old_t, old_v, new_v);
  }
};
template <>
struct variant_helper<> {
  inline static void destroy(size_t id, void* data) {}
  inline static void move(size_t old_t, void* old_v, void* new_v) {}
  inline static void copy(size_t old_t, const void* old_v, void* new_v) {}
};

template <typename... Ts>
struct variant {
 private:
  static const size_t data_size = static_max<sizeof(Ts)...>::value;
  static const size_t data_align = static_max<alignof(Ts)...>::value;
  using data_t = typename std::aligned_storage<data_size, data_align>::type;
  using helper_t = variant_helper<Ts...>;
  static inline size_t invalid_type() { return typeid(void).hash_code(); }
  size_t type_id;
  data_t data;

 public:
  variant() : type_id(invalid_type()) {}
  variant(const variant<Ts...>& old) : type_id(old.type_id) {
    helper_t::copy(old.type_id, &old.data, &data);
  }
  variant(variant<Ts...>&& old) : type_id(old.type_id) {
    helper_t::move(old.type_id, &old.data, &data);
  }
  // Serves as both the move and the copy asignment operator.
  variant<Ts...>& operator=(variant<Ts...> old) {
    std::swap(type_id, old.type_id);
    std::swap(data, old.data);
    return *this;
  }
  template <typename T>
  bool is() {
    return (type_id == typeid(T).hash_code());
  }

  size_t type() { return type_id; }

  bool valid() { return (type_id != invalid_type()); }

  template <typename T, typename... Args>
  void set(Args&&... args) {
    // First we destroy the current contents
    helper_t::destroy(type_id, &data);
    new (&data) T(std::forward<Args>(args)...);
    type_id = typeid(T).hash_code();
  }
  template <typename T>
  const T& get() const {
    // It is a dynamic_cast-like behaviour
    if (type_id == typeid(T).hash_code())
      return *reinterpret_cast<const T*>(&data);
    else
      LOG(FATAL) << "unmatched type get, should be " << type_id << " but get "
                 << typeid(T).name();
    return *reinterpret_cast<const T*>(&data);
  }

  template <typename T>
  T* get_mutable() {
    // It is a dynamic_cast-like behaviour
    if (type_id == typeid(T).hash_code())
      return reinterpret_cast<T*>(&data);
    else
      LOG(FATAL) << "unmatched type get, should be " << type_id << " but get "
                 << typeid(T).name();
  }
  ~variant() { helper_t::destroy(type_id, &data); }
};

}  // namespace lite
}  // namespace paddle
