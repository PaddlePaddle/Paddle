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
#include <paddle/utils/Any.h>
#include <paddle/utils/Error.h>
#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <typeinfo>
#include <vector>

namespace paddle {
namespace topology {
namespace meta {

template <typename T>
class Constraints {
public:
  explicit Constraints(const std::string& name) : name_(name) {}

  Constraints<T>& addConstraint(
      std::function<paddle::Error(T*, bool)> callback) {
    callbacks_.push_back(callback);
    return *this;
  }

  Constraints<T>& mustSet() {
    return this->addConstraint([this](T*, bool setted) -> paddle::Error {
      if (!setted) {
        return paddle::Error("Attribute %s must be setted", name_.c_str());
      }
      return paddle::Error();
    });
  }

  Constraints<T>& defaultValue(const T& val) {
    return this->addConstraint(
        [this, val](T* attr, bool setted) -> paddle::Error {
          if (!setted) {
            *attr = val;
          }
          return paddle::Error();
        });
  }

  paddle::Error operator()(T* attr, bool setted) const {
    for (const auto& callback : callbacks_) {
      paddle::Error err = callback(attr, setted);
      if (!err.isOK()) return err;
    }
    return paddle::Error();
  }

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

private:
  std::string name_;
  // (T* attr, bool setted) -> paddle::Error
  std::vector<std::function<paddle::Error(T*, bool)>> callbacks_;
};

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

  template <typename T>
  static std::shared_ptr<AttributeMeta> create(const std::string& name,
                                               const std::string& description) {
    AttributeMeta* retv = new AttributeMeta(name, typeid(T), description);
    retv->constraintBuilder<T>();
    return std::shared_ptr<AttributeMeta>(retv);
  }

  template <typename T>
  Constraints<T>* constraintBuilder() {
    if (typeid(T).hash_code() == type.hash_code()) {
      if (this->constraints.empty()) {
        this->constraints = Constraints<T>(this->name);
      }
      return any_cast<Constraints<T>>(&this->constraints);
    } else {
      return nullptr;
    }
  }

  template <typename T>
  paddle::Error check(T* attr, bool setted) const {
    auto callbackPtr = any_cast<Constraints<T>>(&this->constraints);
    if (callbackPtr) {
      return (*callbackPtr)(attr, setted);
    } else {
      return paddle::Error("Type mismatched, the input type is %s, need %s",
                           typeid(T).name(),
                           this->type.name());
    }
  }

  paddle::Error check(any* attr, bool setted) const;

private:
  paddle::any constraints;
};
typedef std::shared_ptr<AttributeMeta> AttributeMetaPtr;

}  // namespace meta
}  // namespace topology
}  // namespace paddle
