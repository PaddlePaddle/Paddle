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

#include <string>

namespace paddle {
namespace framework {

class Cryption {
 public:
  Cryption();
  ~Cryption();

  bool CreateKeyInMemory();

  void EncryptInMemory(char* origStr, int strLen);

  void DecryptInMemory(char* origStr, int strLen);

  char* GetEncryptKey() { return encrypt_key; }

  char* GetDecryptKey() { return decrypt_key; }

 private:
  const char* key_string = "0123456789abcdef";
  mutable char* encrypt_key;
  mutable char* decrypt_key;
  mutable int encrypt_key_length;
  mutable int decrypt_key_length;
};

}  // namespace framework
}  // namespace paddle
