#pragma once

#include <boost/noncopyable.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace paddle {
namespace framework {

//! Using unordered_map as Paddle's default map. It will faster than
//! std::map
template <typename Key, typename Val>
using Map = std::unordered_map<Key, Val>;

//! Using unordered_set as Paddle's default set. It will faster than
//! std::set
template <typename T>
using Set = std::unordered_set<T>;

//! Default Vector is std::vector
template <typename T>
using Vector = std::vector<T>;

//! Default String is std::string
using String = std::string;

//! Default unique ptr is std::unique_ptr
template <typename T>
using UniquePtr = std::unique_ptr<T>;

//! Default shared ptr is std::shared_ptr
template <typename T>
using SharedPtr = std::shared_ptr<T>;

//! Default weak ptr is std::weak_ptr
template <typename T>
using WeakPtr = std::weak_ptr<T>;

//! MakeShared will create std::shared_ptr
template <typename T, typename... ARGS>
inline SharedPtr<T> MakeShared(ARGS... args) {
  return std::make_shared<T>(std::forward<ARGS>(args)...);
}

//! MakeUnique will create std::unique_ptr
template <typename T, typename... ARGS>
inline UniquePtr<T> MakeUnique(ARGS... args) {
  return UniquePtr<T>(new T(std::forward<ARGS>(args)...));
}

//! Using boost::noncopyable as non-copyable markup.
using NonCopyable = boost::noncopyable;

}  // namespace framework
}  // namespace paddle
