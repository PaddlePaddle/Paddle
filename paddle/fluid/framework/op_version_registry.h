/* Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.

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
#include <utility>
#include <vector>

#include "paddle/fluid/framework/op_version_proto.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/utils/none.h"

namespace paddle {
namespace framework {
namespace compatible {

namespace pb {
class OpVersionMap;
}  // namespace pb

using OpAttrVariantT =
    boost::variant<bool,                     /* AttrType::BOOL */
                   float,                    /* AttrType::FLOAT */
                   int32_t,                  /* AttrType::INT */
                   int64_t,                  /* AttrType::LONG*/
                   std::string,              /* AttrType::STRING */
                   std::vector<bool>,        /* AttrType::BOOLS */
                   std::vector<float>,       /* AttrType::FLOATS */
                   std::vector<int32_t>,     /* AttrType::INTS */
                   std::vector<int64_t>,     /* AttrType::LONGS */
                   std::vector<std::string>, /* AttrType::STRINGS */
                   paddle::none_t            /* None */
                   >;

struct OpUpdateInfo {
  virtual ~OpUpdateInfo() = default;
};

struct OpAttrInfo : OpUpdateInfo {
  OpAttrInfo(const std::string& name, const std::string& remark,
             const OpAttrVariantT& default_value = paddle::none)
      : name_{name}, default_value_{default_value}, remark_{remark} {}

  const std::string& name() const { return name_; }
  const OpAttrVariantT& default_value() const { return default_value_; }
  const std::string& remark() const { return remark_; }

 private:
  std::string name_;
  OpAttrVariantT default_value_;
  std::string remark_;
};

struct OpInputOutputInfo : OpUpdateInfo {
  OpInputOutputInfo(const std::string& name, const std::string& remark)
      : name_{name}, remark_{remark} {}

  const std::string& name() const { return name_; }
  const std::string& remark() const { return remark_; }

 private:
  std::string name_;
  std::string remark_;
};

struct OpBugfixInfo : OpUpdateInfo {
  explicit OpBugfixInfo(const std::string& remark) : remark_{remark} {}
  const std::string& remark() const { return remark_; }

 private:
  std::string remark_;
};

enum class OpUpdateType {
  kInvalid = 0,
  /* Compatibility upgrade */
  kModifyAttr,
  kNewAttr,
  kNewInput,
  kNewOutput,
  kBugfixWithBehaviorChanged,
  /* Incompatible upgrade, only for existing registration. */
  kDeleteAttr = 100,
  kModifyInput,
  kModifyOutput,
  kDeleteInput,
  kDeleteOutput,
};

class OpUpdateBase {
 public:
  virtual const OpUpdateInfo& info() const = 0;
  virtual OpUpdateType type() const = 0;
  virtual ~OpUpdateBase() = default;
};

template <typename InfoType, OpUpdateType type__>
class OpUpdate : public OpUpdateBase {
 public:
  explicit OpUpdate(const InfoType& info) : info_{info}, type_{type__} {}
  const InfoType& info() const override { return info_; }
  OpUpdateType type() const override { return type_; }

 private:
  InfoType info_;
  OpUpdateType type_;
};

template <OpUpdateType type__, typename InfoType>
OpUpdate<InfoType, type__>* new_update(InfoType&& info) {
  return new OpUpdate<InfoType, type__>(info);
}

template <typename T>
OpAttrVariantT op_attr_wrapper(const T& val) {
  return OpAttrVariantT{val};
}

template <int N>
OpAttrVariantT op_attr_wrapper(const char (&val)[N]) {
  PADDLE_ENFORCE_EQ(
      val[N - 1], 0,
      platform::errors::InvalidArgument(
          "The argument of operator register %c is illegal.", val[N - 1]));
  return OpAttrVariantT{std::string{val}};
}

class OpVersionDesc {
 public:
  /* Compatibility upgrade */
  template <typename T>
  OpVersionDesc&& ModifyAttr(const std::string& name, const std::string& remark,
                             const T& default_value) {
    infos_.emplace_back(new_update<OpUpdateType::kModifyAttr>(
        OpAttrInfo(name, remark, op_attr_wrapper(default_value))));
    return std::move(*this);
  }

  template <typename T>
  OpVersionDesc&& NewAttr(const std::string& name, const std::string& remark,
                          const T& default_value) {
    infos_.emplace_back(new_update<OpUpdateType::kNewAttr>(
        OpAttrInfo(name, remark, op_attr_wrapper(default_value))));
    return std::move(*this);
  }

  OpVersionDesc&& NewInput(const std::string& name, const std::string& remark);
  OpVersionDesc&& NewOutput(const std::string& name, const std::string& remark);
  OpVersionDesc&& BugfixWithBehaviorChanged(const std::string& remark);

  /* Incompatible upgrade, only for existing registration. */
  OpVersionDesc&& DeleteAttr(const std::string& name,
                             const std::string& remark);
  OpVersionDesc&& ModifyInput(const std::string& name,
                              const std::string& remark);
  OpVersionDesc&& ModifyOutput(const std::string& name,
                               const std::string& remark);
  OpVersionDesc&& DeleteInput(const std::string& name,
                              const std::string& remark);
  OpVersionDesc&& DeleteOutput(const std::string& name,
                               const std::string& remark);

 public:
  const std::vector<std::unique_ptr<OpUpdateBase>>& infos() const {
    return infos_;
  }
  OpVersionDesc() = default;
  OpVersionDesc(OpVersionDesc&&) = default;
  OpVersionDesc& operator=(OpVersionDesc&&) = default;

 private:
  std::vector<std::unique_ptr<OpUpdateBase>> infos_;
};

class OpCheckpoint {
 public:
  OpCheckpoint(const std::string& note, OpVersionDesc&& op_version_desc)
      : note_{note},
        op_version_desc_{std::forward<OpVersionDesc>(op_version_desc)} {}
  const std::string& note() const { return note_; }
  const OpVersionDesc& version_desc() { return op_version_desc_; }

  OpCheckpoint() = default;
  OpCheckpoint(OpCheckpoint&&) = default;
  OpCheckpoint& operator=(OpCheckpoint&&) = default;

 private:
  std::string note_;
  OpVersionDesc op_version_desc_;
};

class OpVersion {
 public:
  OpVersion& AddCheckpoint(const std::string& note,
                           OpVersionDesc&& op_version_desc) {
    checkpoints_.emplace_back(OpCheckpoint{note, std::move(op_version_desc)});
    return *this;
  }
  uint32_t version_id() const {
    return static_cast<uint32_t>(checkpoints_.size());
  }
  const std::vector<OpCheckpoint>& checkpoints() const { return checkpoints_; }

  OpVersion() = default;
  OpVersion(OpVersion&&) = default;
  OpVersion& operator=(OpVersion&&) = default;

 private:
  std::vector<OpCheckpoint> checkpoints_;
};

class OpVersionRegistrar {
 public:
  static OpVersionRegistrar& GetInstance() {
    static OpVersionRegistrar instance;
    return instance;
  }
  OpVersion& Register(const std::string& op_type);
  const std::unordered_map<std::string, OpVersion>& GetVersionMap() {
    return op_version_map_;
  }
  bool Has(const std::string& op_type) const {
    return op_version_map_.count(op_type);
  }
  uint32_t version_id(const std::string& op_type) const;

 private:
  std::unordered_map<std::string, OpVersion> op_version_map_;
  OpVersionRegistrar() = default;
  OpVersionRegistrar& operator=(const OpVersionRegistrar&) = delete;
};

inline const std::unordered_map<std::string, OpVersion>& get_op_version_map() {
  return OpVersionRegistrar::GetInstance().GetVersionMap();
}

inline void SaveOpVersions(
    const std::unordered_map<std::string, OpVersion>& src,
    pb::OpVersionMap* dst) {
  for (const auto& pair : src) {
    (*dst)[pair.first].SetVersionID(pair.second.version_id());
  }
}

class OpVersionComparator {
 public:
  virtual bool operator()() = 0;
  virtual ~OpVersionComparator() = default;
};

#define ADD_OP_VERSION_COMPARATOR(cmp_name, cmp_math)                        \
  class OpVersion##cmp_name##Comparator : public OpVersionComparator {       \
   public:                                                                   \
    explicit OpVersion##cmp_name##Comparator(const std::string op_name,      \
                                             uint32_t target_version)        \
        : op_name_(op_name), target_version_(target_version) {}              \
    virtual bool operator()() {                                              \
      uint32_t version_id = 0;                                               \
      if (OpVersionRegistrar::GetInstance().Has(op_name_)) {                 \
        version_id = OpVersionRegistrar::GetInstance().version_id(op_name_); \
      }                                                                      \
      bool check_ok = version_id cmp_math target_version_;                   \
      if (!check_ok) {                                                       \
        LOG(WARNING) << "Check op version in pass failed. op name:"          \
                     << op_name_.c_str() << " op_version:" << version_id     \
                     << "  target_version:" << target_version_;              \
      }                                                                      \
      return check_ok;                                                       \
    }                                                                        \
    virtual ~OpVersion##cmp_name##Comparator() {}                            \
                                                                             \
   private:                                                                  \
    std::string op_name_;                                                    \
    uint32_t target_version_;                                                \
  };

ADD_OP_VERSION_COMPARATOR(LE, <=);
ADD_OP_VERSION_COMPARATOR(EQ, ==);
ADD_OP_VERSION_COMPARATOR(GE, >=);
ADD_OP_VERSION_COMPARATOR(NE, !=);

class OpVersionComparatorCombination {
 public:
  OpVersionComparatorCombination() {}

  OpVersionComparatorCombination& LE(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionLEComparator(op_name, target_version)));
    return *this;
  }
  OpVersionComparatorCombination& EQ(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionEQComparator(op_name, target_version)));
    return *this;
  }
  OpVersionComparatorCombination& GE(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionGEComparator(op_name, target_version)));
    return *this;
  }
  OpVersionComparatorCombination& NE(const std::string& op_name,
                                     int target_version) {
    op_version_comparators_.push_back(std::shared_ptr<OpVersionComparator>(
        new OpVersionNEComparator(op_name, target_version)));
    return *this;
  }

  bool IsMatched() const {
    for (const auto& cmp : op_version_comparators_) {
      if (!(*cmp)()) {
        return false;
      }
    }
    return true;
  }

 private:
  std::vector<std::shared_ptr<OpVersionComparator>> op_version_comparators_;
};

class PassVersionCheckers {
 public:
  PassVersionCheckers& AddCombination(
      const OpVersionComparatorCombination& combinations) {
    pass_version_checkers_.push_back(combinations);
    return *this;
  }
  bool IsPassCompatible() const {
    if (pass_version_checkers_.empty()) {
      return true;
    }
    for (const auto& checker : pass_version_checkers_) {
      if (checker.IsMatched()) {
        return true;
      }
    }
    return false;
  }

 private:
  std::vector<OpVersionComparatorCombination> pass_version_checkers_;
};

class PassVersionCheckerRegistrar {
 public:
  static PassVersionCheckerRegistrar& GetInstance() {
    static PassVersionCheckerRegistrar instance;
    return instance;
  }
  PassVersionCheckers& Register(const std::string& pass_name) {
    PADDLE_ENFORCE_EQ(pass_version_checkers_map_.find(pass_name),
                      pass_version_checkers_map_.end(),
                      platform::errors::AlreadyExists(
                          "PassVersionCheckers(%s) has alredy been registered.",
                          pass_name.c_str()));
    return pass_version_checkers_map_[pass_name];
  }
  bool IsPassCompatible(const std::string& fuse_pass_name) const {
    auto iter = pass_version_checkers_map_.find(fuse_pass_name);
    if (iter == pass_version_checkers_map_.end()) {
      return false;
    }
    return iter->second.IsPassCompatible();
  }

 private:
  std::unordered_map<std::string, PassVersionCheckers>
      pass_version_checkers_map_;

  PassVersionCheckerRegistrar() = default;
  PassVersionCheckerRegistrar& operator=(const PassVersionCheckerRegistrar&) =
      delete;
};

}  // namespace compatible
}  // namespace framework
}  // namespace paddle

#define REGISTER_OP_VERSION(op_type)                                       \
  UNUSED static paddle::framework::compatible::OpVersion&                  \
      RegisterOpVersion__##op_type =                                       \
          paddle::framework::compatible::OpVersionRegistrar::GetInstance() \
              .Register(#op_type)

#define REGISTER_PASS_CAPABILITY(pass_name)                        \
  static auto RegisterOpPassVersionChecker__##pass_name =          \
      paddle::framework::compatible::PassVersionCheckerRegistrar:: \
          GetInstance()                                            \
              .Register(#pass_name)
