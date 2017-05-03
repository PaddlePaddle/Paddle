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

#include <map>
#include <string>

#include "Util.h"

namespace paddle {

/**
 * This class is used to keep a set of class types. It can register a
 * class by a type name and create an instance of a class by type.
 * Example:
 *   // Declare the registrar
 *   ClassRegistrar<Layer, LayerConfig> registar_;
 *
 *   // Register a class using its constructor
 *   registrar_.registerClass<ConvLayer>("conv");
 *
 *   // Register a class using a creation function
 *   registrar_.registerClass("pool", [](LayerConfig& config){
 *     return PoolLayer::create(config);
 *   });
 *
 *   // create a class instance by type name
 *   Layer* layer = registrar_.createByType("conv", config);
 */
template <class BaseClass, typename... CreateArgs>
class ClassRegistrar {
public:
  typedef std::function<BaseClass*(CreateArgs...)> ClassCreator;

  // Register a class using a creation function.
  // The creation function's arguments are CreateArgs
  void registerClass(const std::string& type, ClassCreator creator) {
    CHECK(creatorMap_.count(type) == 0) << "Duplicated class type: " << type;
    creatorMap_[type] = creator;
  }

  // Register a class using its constructor
  // The constructor's arguments are CreateArgs
  template <class ClassType>
  void registerClass(const std::string& type) {
    registerClass(type,
                  [](CreateArgs... args) { return new ClassType(args...); });
  }

  // Create a class instance of type @type using args
  BaseClass* createByType(const std::string& type, CreateArgs... args) {
    ClassCreator creator;
    CHECK(mapGet(type, creatorMap_, &creator)) << "Unknown class type: "
                                               << type;
    return creator(args...);
  }

  template <typename T>
  inline void forEachType(T callback) {
    for (auto it = creatorMap_.begin(); it != creatorMap_.end(); ++it) {
      callback(it->first);
    }
  }

protected:
  std::map<std::string, ClassCreator> creatorMap_;
};

}  // namespace paddle
