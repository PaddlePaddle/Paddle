/* Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.

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

namespace paddle {
namespace framework {

std::string ConvertHexString(const char* buf, int len);

class Cryption {
 public:
  Cryption(const Cryption&) = delete;
  Cryption& operator=(const Cryption&) = delete;

  ~Cryption() {}

  static Cryption* GetCryptorInstance();

  std::string EncryptInMemory(const char* inputStr, const size_t strLen);
  std::string DecryptInMemory(const char* encryptStr, const size_t strLen);

  void EncryptInFile(const std::string& inputFilePath,
                     const std::string& encryptFilePath);

  void DecryptInFile(const std::string& encryptFilePath,
                     const std::string& decryptFilePath);

  const char* GetEncryptKeyPath() { return encrypt_key_path_; }
  const char* GetDecryptKeyPath() { return decrypt_key_path_; }

 private:
  Cryption();

  void CreateKeyInFile();

 private:
  // used to generate encryption key and decryption key
  const char* key_string_ = "0123456789abcdef";

  // a import factor for file encryption/decryption
  //   the encryption/decryption block size should be equal for a file
  const int block_size = 4096;

  // the variables used by encryption/decryption in file
  const char* encrypt_key_path_ = "./encrypt.key";
  const char* decrypt_key_path_ = "./decrypt.key";

  std::unique_ptr<char> encrypt_text_;
  std::unique_ptr<char> decrypt_text_;
};

}  // namespace framework
}  // namespace paddle
