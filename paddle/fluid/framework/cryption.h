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
  Cryption();
  // Cryption(const Cryption&) = delete;

  ~Cryption() {}

  static Cryption* GetCryptionInstance();

  char* EncryptMemoryWithKeyInMemory(const char* inputStr);
  char* DecryptMemoryWithKeyInMemory(const char* encryptStr);

  void EncryptFileWithKeyInFile(const std::string& inputFilePath,
                                const std::string& encryptFilePath);
  void DecryptFileWithKeyInFile(const std::string& encryptFilePath,
                                const std::string& decryptTFilePath);

  char* GetEncryptKey() { return encrypt_key; }
  char* GetDecryptKey() { return decrypt_key; }

 private:
  void CreateKeyInMemory();
  void FreeKeyInMemory();

  void CreateKeyInFile();

 private:
  const char* key_string = "0123456789abcdef";
  const int block_size = 4096;

  char* encrypt_key;
  char* decrypt_key;
  int encrypt_key_length;
  int decrypt_key_length;

  const char* encrypt_key_path = "./encrypt.key";
  const char* decrypt_key_path = "./decrypt.key";

  int original_str_len;

  std::unique_ptr<char> encrypt_text;
  std::unique_ptr<char> decrypt_text;
};

}  // namespace framework
}  // namespace paddle
