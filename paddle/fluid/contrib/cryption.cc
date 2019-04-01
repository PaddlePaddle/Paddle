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

#include "paddle/fluid/contrib/cryption.h"
#include "paddle/fluid/platform/enforce.h"

#include <WBAESLib.h>

namespace paddle {
namespace contrib {

Cryption::Cryption() {
  // Init
  CreateKeyInFile();
}

void Cryption::CreateKeyInFile() {
  int result = WBAES_CREATE_KEY_IN_FILE(key_string_, encrypt_key_path_,
                                        decrypt_key_path_);
  PADDLE_ENFORCE_EQ(0, result, "WBAES create key on disk failed.");
}

Cryption* Cryption::GetCryptorInstance() {
  static Cryption cryptor;
  return &cryptor;
}

std::string Cryption::EncryptInMemory(const char* input_str,
                                      const size_t str_len) {
  // Input Check
  PADDLE_ENFORCE_NOT_NULL(input_str, "The input string is null.");
  PADDLE_ENFORCE_NE(0, str_len,
                    "The length of the input string should not be equal to 0.");
  PADDLE_ENFORCE_EQ(
      0, str_len % 16,
      "Only support input data whose length can be divisible by 16.");

  // Alloc memory
  // Using std::string here will cause an error in WBAES API.
  encrypt_text_.reset(new char[str_len + 1]);
  PADDLE_ENFORCE_NOT_NULL(encrypt_text_.get(),
                          "Encrypt memory allocate failed.");

  // Encrypt
  int result = WBAES_INIT(encrypt_key_path_, NULL);
  PADDLE_ENFORCE_EQ(0, result, "WBAES init encryption environment failed.");

  result = WBAES_ENCRYPT(input_str, encrypt_text_.get(), str_len);
  PADDLE_ENFORCE_EQ(0, result, "WBAES encrypt in memory failed.");

  return std::string(reinterpret_cast<const char*>(encrypt_text_.get()),
                     str_len);
}

std::string Cryption::DecryptInMemory(const char* encrypt_str,
                                      const size_t str_len) {
  // Input Check
  PADDLE_ENFORCE_NOT_NULL(encrypt_str, "The encrypt string is null.");
  PADDLE_ENFORCE_NE(0, str_len,
                    "The length of the input string should not be equal to 0.");
  PADDLE_ENFORCE_EQ(
      0, str_len % 16,
      "Only support input data whose length can be divisible by 16.");

  // Alloc memory
  // Using std::string here will cause an error in WBAES API.
  decrypt_text_.reset(new char[str_len + 1]);
  PADDLE_ENFORCE_NOT_NULL(encrypt_text_.get(),
                          "Encrypt memory allocate failed.");

  // Decrypt
  int result = WBAES_INIT(NULL, decrypt_key_path_);
  PADDLE_ENFORCE_EQ(0, result,
                    "WBAES init decryption environment failed. Makesure the "
                    "decrypt.key exists.");

  result = WBAES_DECRYPT(encrypt_str, decrypt_text_.get(), str_len);
  PADDLE_ENFORCE_EQ(0, result, "WBAES decrypt in memmory failed.");

  return std::string(reinterpret_cast<const char*>(decrypt_text_.get()),
                     str_len);
}

void Cryption::EncryptInFile(const std::string& input_file_path,
                             const std::string& encrypt_file_path) {
  // Input check
  PADDLE_ENFORCE(!input_file_path.empty(),
                 "The input file path cannot be empty.");
  PADDLE_ENFORCE(!encrypt_file_path.empty(),
                 "The encrypted file path cannot be empty.");

  // Encrypt
  int result = WBAES_INIT(encrypt_key_path_, NULL);
  PADDLE_ENFORCE_EQ(0, result, "WBAES init encryption environment failed.");

  result = WBAES_ENCRYPT_FILE(input_file_path.data(), encrypt_file_path.data(),
                              block_size);
  PADDLE_ENFORCE_EQ(0, result, "WBAES encrypt on disk failed.");
}

void Cryption::DecryptInFile(const std::string& encrypt_file_path,
                             const std::string& decrypt_file_path) {
  // Input check
  PADDLE_ENFORCE(!encrypt_file_path.empty(),
                 "The encrypted file path cannot be empty.");
  PADDLE_ENFORCE(!decrypt_file_path.empty(),
                 "The decrypted file path cannot be empty.");

  // Decrypt
  int result = WBAES_INIT(NULL, decrypt_key_path_);
  PADDLE_ENFORCE_EQ(0, result,
                    "WBAES init decryption encironment failed. Makesure the "
                    "decrypt.key exists.");

  result = WBAES_DECRYPT_FILE(encrypt_file_path.data(),
                              decrypt_file_path.data(), block_size);
  PADDLE_ENFORCE_EQ(0, result, "WBAES decrypt on disk failed.");
}

}  // namespace contrib
}  // namespace paddle
