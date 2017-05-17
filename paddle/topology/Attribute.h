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
#include <memory>
#include <string>
#include <unordered_map>
#include "AttributeMap.h"
#include "details/AttributeParser.h"
#include "meta/AttributeMeta.h"
#include "meta/FunctionMeta.h"

namespace paddle {
namespace topology {

/**
 * @brief Base class for Function/Layer attribute.
 *
 * The inherit classes of this class could be registered to Function or Layer,
 * and could be automatically parsed.
 *
 * The example code is:
 *
 * @code{cpp}
 * // Some strong typed attribute used by Function/Layer
 * struct CosSimAttribute : public topology::Attribute {
 *   double scale;
 *
 *   // register function attributes, so we can automatically parsed from
 *   // AttributeMap.
 *   REGISTER_FUNC_ATTRIBUTE() {
 *     regAttr(&CosSimAttribute::scale, "scale", "the scale of cosine operator")
 *       .defaultValue(1.0)
 *       .largerThan(0.0);
 *   }
 * };
 * @endcode{cpp}
 */
class Attribute {
public:
  bool useGPU;

  /**
   * @brief Enable RTTI
   */
  virtual ~Attribute();

protected:
  template <typename T, typename SubClass>
  static meta::Constraints<T>& regAttr(T SubClass::*memPtr,
                                       const std::string& name,
                                       const std::string& description) {
    details::gCurParser->append(name, memPtr);
    return details::gCurFuncMeta->addAttribute<T>(name, description);
  }

  /**
   * Some attribute shared from all attributes.
   */
  static void parentRegAttrs() {
    regAttr(&Attribute::useGPU, "useGPU", "Use GPU or not").mustSet();
  }
};
#define __INIT_SCOPE__()                                         \
  paddle::topology::details::FunctionMetaScope* scope = nullptr; \
  if (withScope)                                                 \
    scope = new paddle::topology::details::FunctionMetaScope(metaPtr.get());

/**
 * Register a function attribute class.
 * See Attribute's document for details
 */
#define REGISTER_FUNC_ATTRIBUTE()                            \
  static void registerFunctionAttribute(                     \
      const paddle::topology::meta::FunctionMetaPtr metaPtr, \
      bool withScope = true) {                               \
    __INIT_SCOPE__()                                         \
    parentRegAttrs();                                        \
    registerFunctionAttribute__impl__();                     \
    delete scope;                                            \
  }                                                          \
  static void registerFunctionAttribute__impl__()

#define REGISTER_FUNC_ATTRIBUTE_EXTENDS(CLASS)               \
  static void registerFunctionAttribute(                     \
      const paddle::topology::meta::FunctionMetaPtr metaPtr, \
      bool withScope = true) {                               \
    __INIT_SCOPE__()                                         \
    CLASS::registerFunctionAttribute(metaPtr, false);        \
    registerFunctionAttribute__impl__();                     \
    delete scope;                                            \
  }                                                          \
  static void registerFunctionAttribute__impl__()

}  // namespace topology

}  // namespace paddle
