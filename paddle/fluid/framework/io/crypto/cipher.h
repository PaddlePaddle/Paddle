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

#pragma once

#include <memory>
#include <string>
#include <unordered_map>

namespace paddle {
namespace framework {

class Cipher {
 public:
  Cipher() = default;
  virtual ~Cipher() {}
  // encrypt string
  virtual std::string Encrypt(const std::string& plaintext,
                              const std::string& key) = 0;
  // decrypt string
  virtual std::string Decrypt(const std::string& ciphertext,
                              const std::string& key) = 0;

  // encrypt strings and read them to file,
  virtual void EncryptToFile(const std::string& plaintext,
                             const std::string& key,
                             const std::string& filename) = 0;
  // read from file and decrypt them
  virtual std::string DecryptFromFile(const std::string& key,
                                      const std::string& filename) = 0;
};

class CipherFactory {
 public:
  CipherFactory() = default;
  static std::shared_ptr<Cipher> CreateCipher(const std::string& config_file);
};
}  // namespace framework
}  // namespace paddle
