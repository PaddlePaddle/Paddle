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
#include "paddle/fluid/platform/enforce.h"

#include "WBAESLib.h"

namespace paddle {
namespace framework {

static std::once_flag init_cryption;
Cryption* cryption_instance;

static std::shared_ptr<char> encryptKey;
static std::shared_ptr<char> decryptKey;

Cryption::Cryption() {
  // Init variables
  encrypt_key = encryptKey.get();
  decrypt_key = encryptKey.get();
  encrypt_key_length = 0;
  decrypt_key_length = 0;
  original_str_len = 0;
  encrypt_text = nullptr;
  decrypt_text = nullptr;
  // Init keys
  CreateKeyInMemory();
}

void Cryption::CreateKeyInMemory() {
  int result = WBAES_OK;

  result =
      WBAES_CREATE_KEY_IN_MEMORY(key_string, &encrypt_key, &encrypt_key_length,
                                 &decrypt_key, &decrypt_key_length);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES create key in memory failed.");

  // LOG(INFO) << "encrypt_key: " << ConvertHexString(encrypt_key,
  // encrypt_key_length);
  // LOG(INFO) << "decrypt_key: " << ConvertHexString(decrypt_key,
  // decrypt_key_length);
}

Cryption* Cryption::GetCryptionInstance() {
  std::call_once(init_cryption, []() { cryption_instance = new Cryption(); });
  return cryption_instance;
}

char* Cryption::EncryptInMemory(const char* inputStr) {
  int result = WBAES_OK;
  original_str_len = strlen(inputStr);

  // Alloc memory
  encrypt_text.reset(new char[((original_str_len + 15) / 16) * 16]);
  // encrypt_text.reset(new char[original_str_len]);

  // Encrypt
  result = WBAES_INIT_MEMORY(encrypt_key, NULL);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init memory failed.");
  result = WBAES_ENCRYPT(inputStr, encrypt_text.get(), original_str_len);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt failed.");

  LOG(INFO) << "input string: " << ConvertHexString(inputStr, original_str_len);
  LOG(INFO) << "encrypt string: "
            << ConvertHexString(encrypt_text.get(), original_str_len);

  return encrypt_text.get();
}

char* Cryption::DecryptInMemory(const char* encryptStr) {
  int result = WBAES_OK;

  // Alloc memory
  decrypt_text.reset(new char[original_str_len]);

  // Decrypt
  result = WBAES_INIT_MEMORY(NULL, decrypt_key);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init memory failed.");
  result = WBAES_DECRYPT(encryptStr, decrypt_text.get(), original_str_len);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES decrypt failed.");

  LOG(INFO) << "encrypt string: "
            << ConvertHexString(encryptStr, original_str_len);
  LOG(INFO) << "decrypt string: "
            << ConvertHexString(decrypt_text.get(), original_str_len);

  // TODO(chenwhql): the last byte will change in cryption
  // decrypt_text.get()[original_str_len] = '\0';

  return decrypt_text.get();
}

void Cryption::EncryptInFile(const std::string& inputFilePath,
                             const std::string& encryptFilePath) {
  int result = WBAES_OK;

  // Encrypt
  result = WBAES_INIT_MEMORY(encrypt_key, NULL);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init memory failed.");
  result = WBAES_ENCRYPT_FILE(inputFilePath.data(), encryptFilePath.data(),
                              block_size);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES encrypt file failed.");

  return;
}

void Cryption::DecryptInFile(const std::string& encryptFilePath,
                             const std::string& decryptFilePath) {
  int result = WBAES_OK;

  // Decrypt
  result = WBAES_INIT_MEMORY(NULL, decrypt_key);
  PADDLE_ENFORCE(WBAES_OK == result, "WBAES init memory failed.");
  result = WBAES_ENCRYPT_FILE(encryptFilePath.data(), decryptFilePath.data(),
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
