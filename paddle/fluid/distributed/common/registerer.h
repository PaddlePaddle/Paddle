// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <iostream>
#include <map>
#include <string>
#include <vector>

namespace paddle {
namespace distributed {

class Any {
 public:
  Any() : content_(NULL) {}

  template <typename ValueType>
  explicit Any(const ValueType &value)
      : content_(new Holder<ValueType>(value)) {}

  Any(const Any &other)
      : content_(other.content_ ? other.content_->clone() : NULL) {}

  ~Any() { delete content_; }

  template <typename ValueType>
  ValueType *any_cast() {
    return content_
               ? &static_cast<Holder<ValueType> *>(content_)->held_  // NOLINT
               : NULL;
  }

 private:
  class PlaceHolder {
   public:
    virtual ~PlaceHolder() {}
    virtual PlaceHolder *clone() const = 0;
  };

  template <typename ValueType>
  class Holder : public PlaceHolder {
   public:
    explicit Holder(const ValueType &value) : held_(value) {}
    virtual PlaceHolder *clone() const { return new Holder(held_); }

    ValueType held_;
  };

  PlaceHolder *content_;
};

class ObjectFactory {
 public:
  ObjectFactory() {}
  virtual ~ObjectFactory() {}
  virtual Any NewInstance() { return Any(); }

 private:
};

typedef std::map<std::string, ObjectFactory *> FactoryMap;
typedef std::map<std::string, FactoryMap> PsCoreClassMap;
#ifdef __cplusplus
extern "C" {
#endif

inline PsCoreClassMap *global_factory_map() {
  static PsCoreClassMap *base_class = new PsCoreClassMap();
  return base_class;
}
#ifdef __cplusplus
}
#endif

inline PsCoreClassMap &global_factory_map_cpp() {
  return *global_factory_map();
}

// typedef pa::Any Any;
// typedef ::FactoryMap FactoryMap;
#define REGISTER_PSCORE_REGISTERER(base_class)                           \
  class base_class##Registerer {                                         \
   public:                                                               \
    static base_class *CreateInstanceByName(const ::std::string &name) { \
      if (global_factory_map_cpp().find(#base_class) ==                  \
          global_factory_map_cpp().end()) {                              \
        LOG(ERROR) << "Can't Find BaseClass For CreateClass with:"       \
                   << #base_class;                                       \
        return NULL;                                                     \
      }                                                                  \
      FactoryMap &map = global_factory_map_cpp()[#base_class];           \
      FactoryMap::iterator iter = map.find(name);                        \
      if (iter == map.end()) {                                           \
        LOG(ERROR) << "Can't Find Class For Create with:" << name;       \
        return NULL;                                                     \
      }                                                                  \
      Any object = iter->second->NewInstance();                          \
      return *(object.any_cast<base_class *>());                         \
    }                                                                    \
  };

#define REGISTER_PSCORE_CLASS(clazz, name)              \
  class ObjectFactory##name : public ObjectFactory {    \
   public:                                              \
    Any NewInstance() { return Any(new name()); }       \
  };                                                    \
  static void register_factory_##name() {               \
    FactoryMap &map = global_factory_map_cpp()[#clazz]; \
    if (map.find(#name) == map.end()) {                 \
      map[#name] = new ObjectFactory##name();           \
    }                                                   \
  }                                                     \
  void register_factory_##name() __attribute__((constructor));

#define CREATE_PSCORE_CLASS(base_class, name) \
  base_class##Registerer::CreateInstanceByName(name);

}  // namespace distributed
}  // namespace paddle
