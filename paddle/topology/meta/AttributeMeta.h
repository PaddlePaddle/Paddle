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
#include <glog/logging.h>
#include <paddle/utils/Any.h>
#include <paddle/utils/Error.h>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <unordered_map>
#include <unordered_set>
#include <vector>
#include "AttributeMap.h"
#include "TypeDefs.h"

namespace paddle {
namespace topology {
namespace meta {

/**
 * Constraints are used to check the validity of a attribute and is a part of
 * meta information of Attribute, e.g., the value must be larger than 1.
 *
 *
 * The implementation of constraints is very straight-forward. It just a
 * array of validation functions. Each validation function has type
 * `(T* attr, bool alreadySet) -> Error`. The first argument is a writable
 * pointer, the second argument is a bool flag whether this attribute is set by
 * user or not. If the attribute is not valid, return a reasonable Error
 * message.
 */
template <typename T>
class Constraints {
public:
  /**
   * @param name the binded attribute name, used for readable error.
   */
  explicit Constraints(const std::string& name) : name_(name) {}

  /**
   * @brief Base method to add a constraint function to Constraints
   * @param callback See the documentation of class.
   * @return *this
   */
  Constraints<T>& addConstraint(
      std::function<paddle::Error(T*, bool)> callback) {
    callbacks_.push_back(callback);
    return *this;
  }

  /**
   * @brief mustSet means this attribute must be set by user.
   * @return *this
   */
  Constraints<T>& mustSet() {
    return this->addConstraint([this](T*, bool alreadySet) -> paddle::Error {
      if (!alreadySet) {
        return paddle::Error("Attribute %s must be alreadySet", name_.c_str());
      }
      return paddle::Error();
    });
  }

  /**
   * @brief set the default value of this attribute.
   * @param val default value
   * @return *this
   */
  Constraints<T>& defaultValue(const T& val) {
    return this->addConstraint(
        [this, val](T* attr, bool alreadySet) -> paddle::Error {
          if (!alreadySet) {
            *attr = val;
          }
          return paddle::Error();
        });
  }

  /**
   * @brief check this attribute is valid or not
   */
  paddle::Error validate(T* attr, bool alreadySet) const {
    for (const auto& callback : callbacks_) {
      paddle::Error err = callback(attr, alreadySet);
      if (!err.isOK()) return err;
    }
    return paddle::Error();
  }

  /**
   * @brief value must be in [min, max].
   */
  Constraints<T>& inRange(const T& min, const T& max) {
    return this->addConstraint([this, min, max](T* attr, bool) {
      if (*attr < min || *attr > max) {
        return paddle::Error("%s(%s) must be in range [%s, %s]",
                             name_.c_str(),
                             std::to_string(*attr).c_str(),
                             std::to_string(min).c_str(),
                             std::to_string(max).c_str());
      }
      return paddle::Error();
    });
  }

  /**
   * @brief value must in set s.
   */
  template <typename SetType>
  Constraints<T>& in(const SetType& s) {
    return this->addConstraint([this, s](T* attr, bool) {
      auto it = s.find(*attr);
      if (it == s.end()) {
        return Error("Attribute %s must in a set", name_.c_str());
      }
      return Error();
    });
  }

  /**
   * @brief value must > min;
   */
  Constraints<T>& largerThan(const T& min) {
    return this->addConstraint([this, min](T* attr, bool) {
      if (*attr < min) {
        return paddle::Error("%s(%s) must be larger than %s",
                             name_.c_str(),
                             std::to_string(*attr).c_str(),
                             std::to_string(min).c_str());
      }
      return paddle::Error();
    });
  }

  /**
   * @brief equal value must equal expect
   */
  Constraints<T>& equal(const T& expect) {
    return this->addConstraint([this, expect](T* actual, bool) {
      if (*actual != expect) {
        return paddle::Error("%s must equals to expect value", name_.c_str());
      }
      return paddle::Error();
    });
  }

  /**
   * value's demension must be sz
   */
  template <typename U>
  Constraints<T>& dimsEq(U sz) {
    static_assert(std::is_same<typename T::size_type, U>::value, "");
    return this->addConstraint([this, sz](T* actual, bool) {
      if (sz != actual->size()) {
        return Error("%s's size should be %d, actually %d",
                     name_.c_str(),
                     sz,
                     actual->size());
      }
      return Error();
    });
  }

private:
  std::string name_;
  // (T* attr, bool alreadySet) -> paddle::Error
  std::vector<std::function<paddle::Error(T*, bool)>> callbacks_;
};

/**
 * @brief The meta information of an attribute. It stores attribute type, name,
 * description and contraints.
 */
class AttributeMeta {
private:
  AttributeMeta(const std::string& name,
                const std::type_info& type,
                const std::string& description)
      : name(name), type(type), description(description) {}

public:
  std::string name;
  const std::type_info& type;
  std::string description;

  /**
   * Create a meta information of an attribute.
   *
   * @tparam T: Attribute type.
   */
  template <typename T>
  static std::shared_ptr<AttributeMeta> create(const std::string& name,
                                               const std::string& description) {
    AttributeMeta* retv = new AttributeMeta(name, typeid(T), description);
    retv->constraints_ = Constraints<T>(name);
    return std::shared_ptr<AttributeMeta>(retv);
  }

  /**
   *  Get Constraints of an attribute. nullptr if type mismatched.
   */
  template <typename T>
  Constraints<T>* constraints() {
    return any_cast<Constraints<T>>(&this->constraints_);
  }

  /**
   * validate attribute attr.
   */
  template <typename T>
  paddle::Error validate(T* attr, bool alreadySet) const {
    auto callbackPtr = any_cast<Constraints<T>>(&this->constraints_);
    if (callbackPtr) {
      return callbackPtr->validate(attr, alreadySet);
    } else {
      return paddle::Error("Type mismatched, the input type is %s, need %s",
                           typeid(T).name(),
                           this->type.name());
    }
  }

  /**
   * validate attribute attr, using any type.
   *
   * @note: not all C++ type is supported. See AttributeMeta.cpp for supported
   * types.
   */
  paddle::Error validate(any* attr, bool alreadySet) const;

private:
  paddle::any constraints_;
};
typedef std::shared_ptr<AttributeMeta> AttributeMetaPtr;

class AttributeMetaMap {
private:
  Map<std::string, AttributeMetaPtr> attributeMetas_;
  std::string errTag_;

public:
  /**
   * @brief WithAttributeMeta is a base class for other meta information which
   * contains meta information of attributes.
   * @param errTag: A tag for readable error message.
   */
  explicit AttributeMetaMap(const std::string& errTag) : errTag_(errTag) {}

  const Map<std::string, AttributeMetaPtr>& getAttributes() const {
    return attributeMetas_;
  }

  /**
   * Add an attribute.
   *
   * @note: Die soon because this function should be invoked during initialize
   * Padddle, and the error is caused by programming, should not be handled by
   * user.
   */
  template <typename T>
  Constraints<T>& addAttribute(const std::string& name,
                               const std::string& description) {
    auto metaPtr = AttributeMeta::create<T>(name, description);
    if (metaPtr == nullptr) {
      LOG(FATAL) << "NULL pointer error when create attribute meta";
    }
    auto attrName = metaPtr->name;
    if (this->attributeMetas_.find(attrName) != this->attributeMetas_.end()) {
      LOG(FATAL) << errTag_ << " attribute " << attrName << " has been set.";
    }
    this->attributeMetas_[attrName] = metaPtr;
    return *(metaPtr->template constraints<T>());
  }
};

}  // namespace meta
}  // namespace topology
}  // namespace paddle
