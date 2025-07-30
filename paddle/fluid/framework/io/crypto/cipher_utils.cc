// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/io/crypto/cipher_utils.h"

#include <cryptopp/osrng.h>

#include <sstream>

#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework {

std::string CipherUtils::GenKey(int length) {
  CryptoPP::AutoSeededRandomPool prng;
  int bit_length = length / 8;
  std::string rng;
  rng.resize(bit_length);
  // CryptoPP::byte key[length];
  prng.GenerateBlock(reinterpret_cast<unsigned char*>(&(rng.at(0))),
                     rng.size());
  return rng;
}

std::string CipherUtils::GenKeyToFile(int length, const std::string& filename) {
  CryptoPP::AutoSeededRandomPool prng;
  std::string rng;
  int bit_length = length / 8;
  rng.resize(bit_length);
  // CryptoPP::byte key[length];
  prng.GenerateBlock(reinterpret_cast<unsigned char*>(&(rng.at(0))),
                     rng.size());
  std::ofstream fout(filename, std::ios::binary);
  PADDLE_ENFORCE_EQ(
      fout.is_open(),
      true,
      common::errors::Unavailable("Failed to open file : %s, "
                                  "make sure input filename is available.",
                                  filename));
  fout.write(rng.c_str(), rng.size());  // NOLINT
  fout.close();
  return rng;
}

std::string CipherUtils::ReadKeyFromFile(const std::string& filename) {
  std::ifstream fin(filename, std::ios::binary);
  std::string ret{std::istreambuf_iterator<char>(fin),
                  std::istreambuf_iterator<char>()};
  fin.close();
  return ret;
}

std::unordered_map<std::string, std::string> CipherUtils::LoadConfig(
    const std::string& config_file) {
  std::ifstream fin(config_file);
  PADDLE_ENFORCE_EQ(
      fin.is_open(),
      true,
      common::errors::Unavailable("Failed to open file : %s, "
                                  "make sure input filename is available.",
                                  config_file));
  std::unordered_map<std::string, std::string> ret;
  char c = 0;
  std::string line;
  std::istringstream iss;
  while (std::getline(fin, line)) {
    if (line.at(0) == '#') {
      continue;
    }
    iss.clear();
    iss.str(line);
    std::string key;
    std::string value;
    if (!(iss >> key >> c >> value) && (c == ':')) {
      PADDLE_THROW(common::errors::InvalidArgument(
          "Parse config file error, "
          "check the format of configure in file %s.",
          config_file));
    }
    ret.insert({key, value});
  }
  return ret;
}

template <>
bool CipherUtils::GetValue<bool>(
    const std::unordered_map<std::string, std::string>& config,
    const std::string& key,
    bool* output) {
  auto itr = config.find(key);
  if (itr == config.end()) {
    return false;
  }
  std::istringstream iss(itr->second);
  *output = false;
  iss >> *output;
  if (iss.fail()) {
    iss.clear();
    iss >> std::boolalpha >> *output;
  }
  return true;
}

const int CipherUtils::AES_DEFAULT_IV_SIZE = 128;
const int CipherUtils::AES_DEFAULT_TAG_SIZE = 128;
}  // namespace paddle::framework
