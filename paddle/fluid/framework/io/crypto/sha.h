// Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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

#include <cryptopp/cryptlib.h>
#include <cryptopp/filters.h>
#include <cryptopp/hex.h>
#include <cryptopp/sha.h>
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

std::string GetSha1(std::string msg) {
  std::string digest;
  CryptoPP::SHA1 hash;
  hash.Update(reinterpret_cast<unsigned char*>(&msg.at(0)), msg.size());
  digest.resize(hash.DigestSize());
  hash.Final(reinterpret_cast<unsigned char*>(&digest.at(0)));
  return digest;
}

std::string HexEncoding(std::string bytes) {
  std::string encoded;
  // Everything newed is destroyed when the StringSource is destroyed
  CryptoPP::StringSource ss(
      bytes, true, new CryptoPP::HexEncoder(new CryptoPP::StringSink(encoded)));
  return encoded;
}

}  // namespace framework
}  // namespace paddle
