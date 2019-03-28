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

#include <memory>
#include <string>

#include "paddle/fluid/framework/cryption.h"
#include "paddle/fluid/platform/dynload/wbaes.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle {
namespace framework {

Cryption::Cryption() {
  // Init
  CreateKeyInFile();
}

void Cryption::CreateKeyInFile() {
  int result = WBAES_OK;

  result = wbaes_create_key_in_file(key_string_, encrypt_key_path_,
                                    decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES create key on disk failed.");
}

Cryption* Cryption::GetCryptorInstance() {
  static Cryption cryptor;
  return &cryptor;
}

std::string Cryption::EncryptInMemory(const char* inputStr,
                                      const size_t strLen) {
  int result = WBAES_OK;

  // Input Check
  PADDLE_ENFORCE_NOT_NULL(inputStr, "The input string is null.");
  PADDLE_ENFORCE(0 != strLen,
                 "The length of the input string should not be equal to 0.");
  PADDLE_ENFORCE(
      0 == strLen % 16,
      "Only support input data whose length can be divisible by 16.");

  // Alloc memory
  encrypt_text_.reset(new char[strLen + 1]);
  PADDLE_ENFORCE_NOT_NULL(encrypt_text_.get(),
                          "Encrypt memory allocate failed.");

  // Encrypt
  result = wbaes_init(encrypt_key_path_, NULL);
  PADDLE_ENFORCE(WBAES_OK == result,
                 "WBAES init encryption environment failed.");

  result = wbaes_encrypt(inputStr, encrypt_text_.get(), strLen);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt in memory failed.");

  return std::string(reinterpret_cast<const char*>(encrypt_text_.get()),
                     strLen);
}

std::string Cryption::DecryptInMemory(const char* encryptStr,
                                      const size_t strLen) {
  int result = WBAES_OK;

  // Input Check
  PADDLE_ENFORCE_NOT_NULL(encryptStr, "The encrypt string is null.");
  PADDLE_ENFORCE(0 != strLen,
                 "The length of the input string should not be equal to 0.");
  PADDLE_ENFORCE(
      0 == strLen % 16,
      "Only support input data whose length can be divisible by 16.");

  // Alloc memory
  decrypt_text_.reset(new char[strLen + 1]);
  PADDLE_ENFORCE_NOT_NULL(encrypt_text_.get(),
                          "Encrypt memory allocate failed.");

  // Decrypt
  result = wbaes_init(NULL, decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result,
                 "WBAES init decryption environment failed. Makesure the "
                 "decrypt.key exists.");

  result = wbaes_decrypt(encryptStr, decrypt_text_.get(), strLen);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES decrypt in memmory failed.");

  return std::string(reinterpret_cast<const char*>(decrypt_text_.get()),
                     strLen);
}

void Cryption::EncryptInFile(const std::string& inputFilePath,
                             const std::string& encryptFilePath) {
  int result = WBAES_OK;

  // Encrypt
  result = wbaes_init(encrypt_key_path_, NULL);
  PADDLE_ENFORCE(WBAES_OK == result,
                 "WBAES init encryption environment failed.");

  result = wbaes_encrypt_file(inputFilePath.data(), encryptFilePath.data(),
                              block_size);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt on disk failed.");
}

void Cryption::DecryptInFile(const std::string& encryptFilePath,
                             const std::string& decryptFilePath) {
  int result = WBAES_OK;

  // Decrypt
  result = wbaes_init(NULL, decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result,
                 "WBAES init decryption encironment failed. Makesure the "
                 "decrypt.key exists.");

  result = wbaes_decrypt_file(encryptFilePath.data(), decryptFilePath.data(),
                              block_size);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES decrypt on disk failed.");
}

}  // namespace framework
}  // namespace paddle
