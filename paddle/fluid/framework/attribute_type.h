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

#include <string>

// __has_include is currently supported by GCC and Clang. However GCC 4.9 may
// have issues and
// returns 1 for 'defined( __has_include )', while '__has_include' is actually
// not supported:
#if defined(__has_include) && (!defined(BOOST_GCC) || (__GNUC__ + 0) >= 5)
#if __has_include(<cxxabi.h>)
#define PADDLE_FRAMEWORK_HAS_CXXABI_H
#endif
#elif defined(__GLIBCXX__) || defined(__GLIBCPP__)
#define PADDLE_FRAMEWORK_HAS_CXXABI_H
#endif

#if defined(PADDLE_FRAMEWORK_HAS_CXXABI_H)
#include <cxxabi.h>
// For some archtectures (mips, mips64, x86, x86_64) cxxabi.h in Android NDK is
// implemented by gabi++ library
// which does not implement abi::__cxa_demangle(). We detect this implementation
// by checking the include guard here.
#if defined(__GABIXX_CXXABI_H__)
#undef PADDLE_FRAMEWORK_HAS_CXXABI_H
#else
#include <cstddef>
#include <cstdlib>
#endif
#endif

namespace paddle {
namespace framework {

inline char const* demangle_alloc(char const* name);
inline void demangle_free(char const* name);

class scoped_demangled_name {
 private:
  char const* m_p;

 public:
  explicit scoped_demangled_name(char const* name)
      : m_p(demangle_alloc(name)) {}

  ~scoped_demangled_name() { demangle_free(m_p); }

  char const* get() const { return m_p; }

  scoped_demangled_name(scoped_demangled_name const&) = delete;
  scoped_demangled_name& operator=(scoped_demangled_name const&) = delete;
};

#if defined(PADDLE_FRAMEWORK_HAS_CXXABI_H)

inline char const* demangle_alloc(char const* name) {
  int status = 0;
  std::size_t size = 0;
  return abi::__cxa_demangle(name, NULL, &size, &status);
}

inline void demangle_free(char const* name) {
  std::free(const_cast<char*>(name));
}

inline std::string demangle(char const* name) {
  scoped_demangled_name demangled_name(name);
  char const* p = demangled_name.get();
  if (!p) p = name;
  return p;
}

#else

inline char const* demangle_alloc(char const* name) { return name; }

inline void demangle_free(char const*) {}

inline std::string demangle(char const* name) { return name; }

#endif

}  // namespace framework
}  // namespace paddle
