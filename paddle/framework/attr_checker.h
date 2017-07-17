#pragma once

#include <boost/variant.hpp>
#include <functional>
#include <string>
#include <unordered_map>
#include <vector>
#include "paddle/framework/enforce.h"

namespace paddle {
namespace framework {

typedef boost::variant<boost::blank, bool, int, float, std::string,
                       std::vector<int>, std::vector<float>,
                       std::vector<std::string>>
    Attribute;
typedef std::unordered_map<std::string, Attribute> AttributeMap;

// check whether a value(attribute) fit a certain limit
template <typename T>
class LargerThanChecker {
 public:
  LargerThanChecker(T lower_bound) : lower_bound_(lower_bound) {}
  void operator()(const T& value) const {
    PADDLE_ENFORCE(value > lower_bound_, "larger_than check fail");
  }

 private:
  T lower_bound_;
};

template <typename T>
class LessThanChecker {
 public:
  LessThanChecker(T upper_bound) : upper_bound_(upper_bound) {}
  void operator()(const T& value) const {
    PADDLE_ENFORCE(value < upper_bound_, "less_than check failed!");
  }

 private:
  T upper_bound_;
};

// we can provide users more common Checker, like 'LessThanChecker',
// 'BetweenChecker'...

template <typename T>
class DefaultValueSetter {
 public:
  DefaultValueSetter(T default_value) : default_value_(default_value) {}
  void operator()(T& value) const { value = default_value_; }

 private:
  T default_value_;
};

// check whether a certain attribute fit its limits
// an attribute can have more than one limits
template <typename T>
class TypedAttrChecker {
  typedef std::function<void(T&)> ValueChecker;

 public:
  TypedAttrChecker(const std::string& attr_name) : attr_name_(attr_name) {}

  TypedAttrChecker& LargerThan(const T& lower_bound) {
    value_checkers_.push_back(LargerThanChecker<T>(lower_bound));
    return *this;
  }

  TypedAttrChecker& LessThan(const T& upper_bound) {
    value_checkers_.push_back(LessThanChecker<T>(upper_bound));
    return *this;
  }

  // we can add more common limits, like LessThan(), Between()...

  TypedAttrChecker& SetDefault(const T& default_value) {
    PADDLE_ENFORCE(default_value_setter_.empty(),
                   "%s can't have more than one default value!", attr_name_);
    default_value_setter_.push_back(DefaultValueSetter<T>(default_value));
    return *this;
  }

  // allow users provide their own checker
  TypedAttrChecker& AddCustomChecker(const ValueChecker& checker) {
    value_checkers_.push_back(checker);
    return *this;
  }

  void operator()(AttributeMap& attr_map) const {
    if (!attr_map.count(attr_name_)) {
      // user do not set this attr
      PADDLE_ENFORCE(!default_value_setter_.empty(),
                     "Attribute '%s' is required!", attr_name_);
      // default_value_setter_ has no more than one element
      T val;
      (default_value_setter_[0])(val);
      attr_map[attr_name_] = val;
    }
    Attribute& attr = attr_map.at(attr_name_);
    T& attr_value = boost::get<T>(attr);
    for (const auto& checker : value_checkers_) {
      checker(attr_value);
    }
  }

 private:
  std::string attr_name_;
  std::vector<ValueChecker> value_checkers_;
  std::vector<ValueChecker> default_value_setter_;
};

// check whether op's all attributes fit their own limits
class OpAttrChecker {
  typedef std::function<void(AttributeMap&)> AttrChecker;

 public:
  template <typename T>
  TypedAttrChecker<T>& AddAttrChecker(const std::string& attr_name) {
    attr_checkers_.push_back(TypedAttrChecker<T>(attr_name));
    AttrChecker& checker = attr_checkers_.back();
    return *(checker.target<TypedAttrChecker<T>>());
  }

  void Check(AttributeMap& attr_map) const {
    for (const auto& checker : attr_checkers_) {
      checker(attr_map);
    }
  }

 private:
  std::vector<AttrChecker> attr_checkers_;
};

}  // namespace framework
}  // namespace paddle
