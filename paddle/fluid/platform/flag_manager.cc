// Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
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

#include <mutex>  // NOLINT
#include <regex>  // NOLINT
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "gflags/gflags.h"
#include "glog/logging.h"
#include "paddle/fluid/platform/enforce.h"
#include "paddle/fluid/platform/flag_manager.h"

namespace paddle {
namespace platform {
namespace flags {

struct CommandLineFlag {
  std::string value;
  std::string filename;
};

using CommandLineFlagMap = std::unordered_map<std::string, CommandLineFlag>;

class GFlagWrapper {
 public:
  static std::unordered_set<std::string> GetFlagNames() {
    std::unordered_set<std::string> flag_names;
    std::vector<::GFLAGS_NAMESPACE::CommandLineFlagInfo> gflag_infos;
    ::GFLAGS_NAMESPACE::GetAllFlags(&gflag_infos);
    for (const auto& gflag_info : gflag_infos) {
      flag_names.insert(gflag_info.name);
    }
    return flag_names;
  }

  static bool ParseCommandLineFlags(
      const CommandLineFlagMap& flags,
      std::function<uint32_t(int*, char***, bool)> f) {
    FLAGS_logtostderr = true;

    std::vector<std::string> cmd_vector{"dummy"};
    const std::string dash{"--"};
    const std::string assign{"="};
    for (const auto& flag : flags) {
      std::string f{dash + flag.first + assign + flag.second.value};
      cmd_vector.emplace_back(std::move(f));
    }

    std::vector<char*> argv;
    for (auto& cmd : cmd_vector) {
      argv.emplace_back(&cmd[0]);
    }
    int argc = argv.size();
    char** arr = argv.data();
    f(&argc, &arr, false);
    return true;
  }
};

class CommandLineFlags {
 public:
  CommandLineFlags() : internal_gflag_names_{GFlagWrapper::GetFlagNames()} {}

  template <typename T>
  bool Insert(const std::string& flag, T value, const std::string& filename) {
    return Insert(flag, std::to_string(value), filename);
  }

  bool Insert(const std::string& flag, const std::string& value,
              const std::string& filename);
  bool Insert(const std::vector<std::string>& flags);
  bool ParseWithGFlag(uint32_t (*external_func)(int*, char***, bool)) const;

 private:
  mutable std::once_flag flag_;
  mutable std::mutex mutex_;
  std::unordered_set<std::string> internal_gflag_names_;
  CommandLineFlagMap external_flags_;
  CommandLineFlagMap internal_flags_;
};

template <>
bool CommandLineFlags::Insert<std::string>(const std::string& flag,
                                           std::string value,
                                           const std::string& filename) {
  return Insert(flag, std::move(value), filename);
}

bool CommandLineFlags::Insert(const std::string& flag, const std::string& value,
                              const std::string& filename) {
  std::unique_lock<std::mutex> w_lock{mutex_};
  if (internal_gflag_names_.count(flag)) {
    if (!internal_flags_.count(flag)) {
      VLOG(5) << "[Get internal flag] " << flag << " = " << value;
      internal_flags_.insert({flag, CommandLineFlag{value, filename}});
    } else {
      VLOG(3) << "[Warning] The internal flag '" << flag
              << "' has been set to value '" << internal_flags_.at(flag).value
              << " by file '" << filename << "'. It will be replaced by file '"
              << filename << "' with '" << value << "'.\n";
      internal_flags_.at(flag) = CommandLineFlag{value, filename};
    }
  } else {
    if (!external_flags_.count(flag)) {
      VLOG(5) << "[Get external flag] " << flag << " = " << value;
      external_flags_.insert({flag, CommandLineFlag{value, filename}});
    } else {
      VLOG(3) << "[Warning] The external flag '" << flag
              << "' has been set to value '" << external_flags_.at(flag).value
              << " by file '" << filename << "'. It will be replaced by file '"
              << filename << "' with '" << value << "'.\n";
      external_flags_.at(flag) = CommandLineFlag{value, filename};
    }
  }
  return true;
}

bool CommandLineFlags::Insert(const std::vector<std::string>& flags) {
  bool ret = false;
  size_t front_idx = 0;
  if (flags.front() == "dummy") {
    front_idx = 1;
  }
  static const std::regex pattern{R"(--(\w+)=(\d*.*\d*|\w+))"};
  for (size_t i = front_idx; i < flags.size(); ++i) {
    std::smatch matches;
    std::regex_search(flags[i], matches, pattern);
    PADDLE_ENFORCE_EQ(matches.size(), 3UL,
                      platform::errors::InvalidArgument(
                          "The pattern of flags '%s' is illegal. It should be "
                          "like '--flag=value'.",
                          flags[i]));
    ret = Insert(std::string{matches[1]}, std::string{matches[2]}, "") && ret;
  }
  return ret;
}

bool CommandLineFlags::ParseWithGFlag(uint32_t (*external_func)(int*, char***,
                                                                bool)) const {
  bool ret = false;
  std::call_once(flag_, [&ret, external_func, this]() {
    ret = GFlagWrapper::ParseCommandLineFlags(
              internal_flags_, ::GFLAGS_NAMESPACE::ParseCommandLineFlags) &&
          ret;
    if (external_func &&
        external_func != ::GFLAGS_NAMESPACE::ParseCommandLineFlags) {
      ret =
          GFlagWrapper::ParseCommandLineFlags(external_flags_, external_func) &&
          ret;
    }
  });
  return ret;
}
}  // namespace flags

FlagRegistrar::FlagRegistrar() { flags_.reset(new flags::CommandLineFlags); }

FlagRegistrar& FlagRegistrar::Get() {
  static FlagRegistrar instance;
  return instance;
}

template <typename T>
bool FlagRegistrar::Insert(const std::string& flag, T value,
                           const std::string& filename) {
  return flags_->Insert(flag, value, filename);
}

bool FlagRegistrar::Insert(const std::vector<std::string>& flags) {
  return flags_->Insert(flags);
}

bool FlagRegistrar::SyncFlagsOnce(uint32_t (*external_func)(int*, char***,
                                                            bool)) const {
  return flags_->ParseWithGFlag(external_func);
}

template bool FlagRegistrar::Insert<int32_t>(const std::string& flag,
                                             int32_t value,
                                             const std::string& filename);
template bool FlagRegistrar::Insert<int64_t>(const std::string& flag,
                                             int64_t value,
                                             const std::string& filename);
template bool FlagRegistrar::Insert<uint32_t>(const std::string& flag,
                                              uint32_t value,
                                              const std::string& filename);
template bool FlagRegistrar::Insert<uint64_t>(const std::string& flag,
                                              uint64_t value,
                                              const std::string& filename);
template bool FlagRegistrar::Insert<double>(const std::string& flag,
                                            double value,
                                            const std::string& filename);
template bool FlagRegistrar::Insert<bool>(const std::string& flag, bool value,
                                          const std::string& filename);
template bool FlagRegistrar::Insert<std::string>(const std::string& flag,
                                                 std::string value,
                                                 const std::string& filename);

}  // namespace platform
}  // namespace paddle
