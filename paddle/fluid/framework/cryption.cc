/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

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
#include <mutex>  // NOLINT
#include <string>

#include "glog/logging.h"
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
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES create key in file failed.");

  return;
}

Cryption* Cryption::GetCryptorInstance() {
  static Cryption cryptor;
  return &cryptor;
}

std::string Cryption::EncryptInMemory(const char* inputStr,
                                      const size_t strLen) {
  int result = WBAES_OK;

  // Input Check
  PADDLE_ENFORCE_NOT_NULL(inputStr, "Input string is null.");

  // Alloc memory
  encrypt_text_.reset(new char[strLen + 1]);

  // Encrypt
  result = wbaes_init(encrypt_key_path_, NULL);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = wbaes_encrypt(inputStr, encrypt_text_.get(), strLen);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt failed.");

  return std::string(reinterpret_cast<const char*>(encrypt_text_.get()),
                     strLen);
}

std::string Cryption::DecryptInMemory(const char* encryptStr,
                                      const size_t strLen) {
  int result = WBAES_OK;

  // Input Check
  PADDLE_ENFORCE_NOT_NULL(encryptStr, "Encrypt string is null.");

  // Alloc memory
  decrypt_text_.reset(new char[strLen + 1]);

  // Decrypt
  result = wbaes_init(NULL, decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = wbaes_decrypt(encryptStr, decrypt_text_.get(), strLen);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES decrypt failed.");

  return std::string(reinterpret_cast<const char*>(decrypt_text_.get()),
                     strLen);
}

void Cryption::EncryptInFile(const std::string& inputFilePath,
                             const std::string& encryptFilePath) {
  int result = WBAES_OK;

  // Encrypt
  result = wbaes_init(encrypt_key_path_, NULL);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = wbaes_encrypt_file(inputFilePath.data(), encryptFilePath.data(),
                              block_size);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt file failed.");

  return;
}

void Cryption::DecryptInFile(const std::string& encryptFilePath,
                             const std::string& decryptFilePath) {
  int result = WBAES_OK;

  // Decrypt
  result = wbaes_init(NULL, decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = wbaes_decrypt_file(encryptFilePath.data(), decryptFilePath.data(),
                              block_size);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt file failed.");

  return;
}

std::string ConvertHexString(const char* buf, int len) {
  int i = 0, j = 0;
  static char str_buf[1024];
  for (; (i < 1024 - 2) && (j < len); i += 2, ++j) {
    snprintf(&str_buf[i], 3, "%02x", (unsigned char)buf[j]);  // NOLINT
  }
  return std::string(str_buf);
}

}  // namespace framework
}  // namespace paddle
