#pragma once

#include <functional>
#include <string>
#include <vector>
#include "paddle/framework/op_desc.pb.h"

namespace paddle {
namespace framework {

class OpDesc;

// check whether a value(attribute) fit a certain limit
template <typename T>
class LargerThanChecker {
 public:
  LargerThanChecker(T lower_bound) : lower_bound_(lower_bound) {}
  bool operator()(const T& value) const { return value > lower_bound_; }

 private:
  T lower_bound_;
};

// we can provide users more common Checker, like 'LessThanChecker',
// 'BetweenChecker'...

// check whether a certain attribute fit its limits
// an attribute can have more than one limits
template <typename T>
class TypedAttrChecker {
  typedef std::function<bool(const T& value)> ValueChecker;

 public:
  TypedAttrChecker(const std::string& attr_name)
      : attr_name_(attr_name), has_default_(false) {}

  TypedAttrChecker& LargerThan(const T& lower_bound) {
    value_checkers_.push_back(LargerThanChecker<T>(lower_bound));
    return *this;
  }

  // we can add more common limits, like LessThan(), Between()...

  TypedAttrChecker& SetDefault(const T& default_value) {
    default_value_ = default_value;
    has_default_ = true;
    return *this;
  }

  // allow users provide their own checker
  TypedAttrChecker& AddCustomChecker(const ValueChecker& checker) {
    value_checkers_.push_back(checker);
    return *this;
  }

  bool operator()(OpDesc& op_desc) const {
    //========= pseudocode begin ==========//
    // check whether attr need default value;
    if (!op_disc.has_fild(attr_name_)) {
      if (has_default_) {
        op_disc.set_fild_by_name(attr_name_, default_value_);
      } else {
        return false;
      }
    }
    // check whether attr fits all its limits;
    T& attr_value = op_desc.get_field_by_name(attr_name_);
    for (const auto& checker : value_checkers_) {
      if (!checker(attr_value)) {
        return false;
      }
    }
    return true;
    //========= pseudocode end ==========//
  }

 private:
  std::string attr_name_;
  std::vector<ValueChecker> value_checkers_;
  bool has_default_;
  T default_value_;
};

// check whether op's all attributes fit their own limits
class OpAttrChecker {
  typedef std::function<bool<OpDesc & op_desc>> AttrChecker;

 public:
  template <typename T>
  TypedAttrChecker<T>& AddAttrChecker(const std::string& attr_name) {
    attr_checkers_.push_back(TypedAttrChecker<T>(attr_name));
    AttrChecker& checker = attr_checkers_.back();
    return *(checker.target<TypedAttrChecker<T>>())
  }

  bool PassCheck(OpDesc& op_desc) {
    for (const auto& checker : attr_checkers_) {
      if (!checker(op_desc)) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<AttrChecker> attr_checkers_;
};

}  // namespace framework
}  // namespace paddle
