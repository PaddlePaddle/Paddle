/* Copyright (c) 2016 PaddlePaddle Authors. All Rights Reserved.

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

#include <istream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace paddle {
namespace operators {
namespace benchmark {

struct OpInputConfig {
  OpInputConfig() {}
  explicit OpInputConfig(std::istream& is);

  void ParseDType(std::istream& is);
  void ParseInitializer(std::istream& is);
  void ParseDims(std::istream& is);
  void ParseLoD(std::istream& is);

  std::string name;
  std::string dtype{"fp32"};  // int32/int, int64/long, fp32/float, fp64/double
  std::string initializer{"random"};  // random, natural, zeros, file
  std::string filename{""};
  std::vector<int64_t> dims;
  std::vector<std::vector<size_t>> lod;
};

struct OpTesterConfig {
  OpTesterConfig() {}
  explicit OpTesterConfig(const std::string& filename);

  bool Init(std::istream& is);

  bool ParseAttrs(std::istream& is);

  const OpInputConfig* GetInput(const std::string& name);

  std::string op_type;
  std::vector<OpInputConfig> inputs;
  std::unordered_map<std::string, std::string> attrs;
  int device_id{-1};  // CPU: -1
  int repeat{1};
  int profile{0};
  int print_debug_string{0};
  double runtime{0.0};
};

static bool Has(const std::vector<std::string>& vec, const std::string& item) {
  for (size_t i = 0; i < vec.size(); ++i) {
    if (vec[i] == item) {
      return true;
    }
  }
  return false;
}

template <typename T>
T StringTo(const std::string& str) {
  std::istringstream is(str);
  T value;
  is >> value;
  return value;
}

static const char kStartSeparator[] = "{";
static const char kEndSeparator[] = "}";
static const char kSepBetweenItems[] = ";";

static bool StartWith(const std::string& str, const std::string& substr) {
  return str.find(substr) == 0;
}

static bool EndWith(const std::string& str, const std::string& substr) {
  return str.rfind(substr) == (str.length() - substr.length());
}

static void EraseEndSep(std::string* str,
                        std::string substr = kSepBetweenItems) {
  if (EndWith(*str, substr)) {
    str->erase(str->length() - substr.length(), str->length());
  }
}

OpInputConfig::OpInputConfig(std::istream& is) {
  std::string sep;
  is >> sep;
  if (sep == kStartSeparator) {
    while (sep != kEndSeparator) {
      is >> sep;
      if (sep == "name" || sep == "name:") {
        is >> name;
        EraseEndSep(&name);
      } else if (sep == "dtype" || sep == "dtype:") {
        ParseDType(is);
      } else if (sep == "initializer" || sep == "initializer:") {
        ParseInitializer(is);
      } else if (sep == "dims" || sep == "dims:") {
        ParseDims(is);
      } else if (sep == "lod" || sep == "lod:") {
        ParseLoD(is);
      } else if (sep == "filename") {
        is >> filename;
        EraseEndSep(&filename);
      }
    }
  }
}

void OpInputConfig::ParseDType(std::istream& is) {
  std::string dtype_str;
  is >> dtype_str;
  EraseEndSep(&dtype_str);

  if (dtype_str == "int32" || dtype_str == "int") {
    dtype = "int32";
  } else if (dtype_str == "int64" || dtype_str == "long") {
    dtype = "int64";
  } else if (dtype_str == "fp32" || dtype_str == "float") {
    dtype = "fp32";
  } else if (dtype_str == "fp64" || dtype_str == "double") {
    dtype = "fp64";
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported dtype %s in OpInputConfig.", dtype_str.c_str()));
  }
  VLOG(4) << "dtype of input " << name << " is: " << dtype;
}

void OpInputConfig::ParseInitializer(std::istream& is) {
  std::string initializer_str;
  is >> initializer_str;
  EraseEndSep(&initializer_str);

  const std::vector<std::string> supported_initializers = {
      "random", "natural", "zeros", "file"};
  if (!Has(supported_initializers, initializer_str)) {
    PADDLE_THROW(common::errors::Unimplemented(
        "Unsupported initializer %s in OpInputConfig.",
        initializer_str.c_str()));
  }

  initializer = initializer_str;
  VLOG(4) << "initializer of input " << name << " is: " << initializer;
}

void OpInputConfig::ParseDims(std::istream& is) {
  std::string dims_str;
  is >> dims_str;

  dims.clear();
  std::string token;
  std::istringstream token_stream(dims_str);
  while (std::getline(token_stream, token, 'x')) {
    dims.push_back(std::stoi(token));
  }
}

void OpInputConfig::ParseLoD(std::istream& is) {
  std::string lod_str;
  std::string start_sep =
      std::string(kStartSeparator) + std::string(kStartSeparator);
  std::string end_sep = std::string(kEndSeparator) + std::string(kEndSeparator);

  std::string sep;
  is >> sep;
  if (StartWith(sep, start_sep)) {
    lod_str += sep;
    while (!EndWith(sep, end_sep)) {
      is >> sep;
      lod_str += sep;
    }
  }
  EraseEndSep(&lod_str);
  PADDLE_ENFORCE_GE(
      lod_str.length(),
      4U,
      common::errors::InvalidArgument(
          "The length of lod string should be "
          "equal to or larger than 4. But length of lod string is %zu.",
          lod_str.length()));
  VLOG(4) << "lod: " << lod_str << ", length: " << lod_str.length();

  // Parse the lod_str
  lod.clear();
  for (size_t i = 1; i < lod_str.length() - 1;) {
    if (lod_str[i] == '{') {
      std::vector<size_t> level;
      while (lod_str[i] != '}') {
        ++i;

        std::string number;
        while (lod_str[i] >= '0' && lod_str[i] <= '9') {
          number += lod_str[i];
          ++i;
        }
        level.push_back(StringTo<size_t>(number));
      }
      lod.push_back(level);
    } else if (lod_str[i] == '}') {
      ++i;
    }
  }
}

OpTesterConfig::OpTesterConfig(const std::string& filename) {
  std::ifstream fin(filename, std::ios::in | std::ios::binary);
  PADDLE_ENFORCE_EQ(
      static_cast<bool>(fin),
      true,
      common::errors::InvalidArgument("OpTesterConfig cannot open file %s.",
                                      filename.c_str()));

  Init(fin);
}

bool OpTesterConfig::Init(std::istream& is) {
  std::string sep;
  is >> sep;
  if (sep == kStartSeparator) {
    while (sep != kEndSeparator) {
      is >> sep;
      if (sep == "op_type" || sep == "op_type:") {
        is >> op_type;
      } else if (sep == "device_id" || sep == "device_id:") {
        is >> device_id;
      } else if (sep == "repeat" || sep == "repeat:") {
        is >> repeat;
      } else if (sep == "profile" || sep == "profile:") {
        is >> profile;
      } else if (sep == "print_debug_string" || sep == "print_debug_string:") {
        is >> print_debug_string;
      } else if (sep == "input" || sep == "input:") {
        OpInputConfig input_config(is);
        inputs.push_back(input_config);
      } else if (sep == "attrs" || sep == "attrs:") {
        ParseAttrs(is);
      } else {
        if (sep != kEndSeparator) {
          return false;
        }
      }
    }
  } else {
    return false;
  }
  return true;
}

bool OpTesterConfig::ParseAttrs(std::istream& is) {
  std::string sep;
  is >> sep;
  if (sep == kStartSeparator) {
    while (true) {
      std::string key;
      is >> key;
      if (key == kEndSeparator) {
        break;
      }

      std::string value;
      is >> value;
      EraseEndSep(&key, ":");
      EraseEndSep(&value);
      VLOG(4) << "attrs: " << key << ", " << value;

      attrs[key] = value;
    }
  }
  return true;
}

const OpInputConfig* OpTesterConfig::GetInput(const std::string& name) {
  for (size_t i = 0; i < inputs.size(); ++i) {
    if (inputs[i].name == name) {
      return &inputs[i];
    }
  }
  return nullptr;
}

}  // namespace benchmark
}  // namespace operators
}  // namespace paddle
