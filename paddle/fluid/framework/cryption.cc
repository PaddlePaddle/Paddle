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

#include "WBAESLib.h"

namespace paddle {
namespace framework {

Cryption::Cryption() {
  // Init
  CreateKeyInFile();
}

void Cryption::CreateKeyInFile() {
  int result = WBAES_OK;
  result = WBAES_CREATE_KEY_IN_FILE(key_string_, encrypt_key_path_,
                                    decrypt_key_path_);
  // result = platform::dynload::WBAES_CREATE_KEY_IN_FILE(key_string_,
  // encrypt_key_path_,
  //                                  decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES create key in file failed.");

  return;
}

Cryption* Cryption::GetCryptorInstance() {
  static Cryption cryptor;
  return &cryptor;
}

std::string Cryption::EncryptInMemory(const char* inputStr,
                                      size_t* encryptLen) {
  int result = WBAES_OK;
  size_t strLen = std::char_traits<char>::length(inputStr);
  *encryptLen = ((strLen + 15) / 16) * 16;

  // Input Check
  PADDLE_ENFORCE_NOT_NULL(inputStr, "Input string is null.");

  // Alloc memory
  // LOG(INFO) << "encrypt length: " << *encryptLen;
  encrypt_text_.reset(new char[*encryptLen + 1]);

  // Encrypt
  WBAES_INIT(encrypt_key_path_, NULL);
  // result = platform::dynload::WBAES_INIT(encrypt_key_path_, NULL);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = WBAES_ENCRYPT(inputStr, encrypt_text_.get(), *encryptLen);
  // result = platform::dynload::WBAES_ENCRYPT(inputStr, encrypt_text_.get(),
  // *encryptLen);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt failed.");

  // LOG(INFO) << "input string: " << ConvertHexString(inputStr, *encryptLen);
  // LOG(INFO) << "encrypt string: "
  //           << ConvertHexString(encrypt_text_.get(), *encryptLen);

  encrypt_text_.get()[*encryptLen] = '\0';

  return std::string(reinterpret_cast<const char*>(encrypt_text_.get()),
                     *encryptLen);
}

char* Cryption::DecryptInMemory(const char* encryptStr, const size_t& strLen) {
  int result = WBAES_OK;

  // Input Check
  PADDLE_ENFORCE_NOT_NULL(encryptStr, "Encrypt string is null.");

  // Alloc memory
  // LOG(INFO) << "encrypt length: " << strLen;
  decrypt_text_.reset(new char[strLen + 1]);

  // Decrypt
  WBAES_INIT(NULL, decrypt_key_path_);
  // result = platform::dynload::WBAES_INIT(NULL, decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  WBAES_DECRYPT(encryptStr, decrypt_text_.get(), strLen);
  // result = platform::dynload::WBAES_DECRYPT(encryptStr, decrypt_text_.get(),
  // strLen);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES decrypt failed.");

  // LOG(INFO) << "encrypt string: " << ConvertHexString(encryptStr, strLen);
  // LOG(INFO) << "decrypt string: "
  //           << ConvertHexString(decrypt_text_.get(), strLen);

  // TODO(chenwhql): the last byte will change in cryption
  decrypt_text_.get()[strLen] = '\0';

  return decrypt_text_.get();
}

void Cryption::EncryptInFile(const std::string& inputFilePath,
                             const std::string& encryptFilePath) {
  int result = WBAES_OK;

  // Encrypt
  WBAES_INIT(encrypt_key_path_, NULL);
  // result = platform::dynload::WBAES_INIT(encrypt_key_path_, NULL);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = WBAES_ENCRYPT_FILE(inputFilePath.data(), encryptFilePath.data(),
                              block_size);
  // result = platform::dynload::WBAES_ENCRYPT_FILE(inputFilePath.data(),
  // encryptFilePath.data(),
  //                            block_size);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt file failed.");

  return;
}

void Cryption::DecryptInFile(const std::string& encryptFilePath,
                             const std::string& decryptFilePath) {
  int result = WBAES_OK;

  // Decrypt
  result = WBAES_INIT(NULL, decrypt_key_path_);
  // result = platform::dynload::WBAES_INIT(NULL, decrypt_key_path_);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init file failed.");

  result = WBAES_DECRYPT_FILE(encryptFilePath.data(), decryptFilePath.data(),
                              block_size);
  // result = platform::dynload::WBAES_DECRYPT_FILE(encryptFilePath.data(),
  // decryptFilePath.data(),
  //                            block_size);
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
