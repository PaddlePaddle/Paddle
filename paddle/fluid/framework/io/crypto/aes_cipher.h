// Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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

#include <string>

#include "paddle/fluid/framework/io/crypto/cipher.h"

namespace CryptoPP {

class StreamTransformationFilter;
class SymmetricCipher;
class AuthenticatedSymmetricCipher;
class AuthenticatedDecryptionFilter;
class AuthenticatedEncryptionFilter;
template <class CryptoppCipher>
class member_ptr;

}  // namespace CryptoPP
namespace paddle {
namespace framework {

class AESCipher : public Cipher {
 public:
  AESCipher() = default;
  ~AESCipher() {}

  std::string Encrypt(const std::string& input,
                      const std::string& key) override;
  std::string Decrypt(const std::string& input,
                      const std::string& key) override;

  void EncryptToFile(const std::string& input,
                     const std::string& key,
                     const std::string& filename) override;
  std::string DecryptFromFile(const std::string& key,
                              const std::string& filename) override;

  void Init(const std::string& cipher_name,
            const int& iv_size,
            const int& tag_size);

 private:
  std::string EncryptInternal(const std::string& plaintext,
                              const std::string& key);
  std::string DecryptInternal(const std::string& ciphertext,
                              const std::string& key);

  std::string AuthenticatedEncryptInternal(const std::string& plaintext,
                                           const std::string& key);
  std::string AuthenticatedDecryptInternal(const std::string& ciphertext,
                                           const std::string& key);

  void BuildCipher(
      bool for_encrypt,
      bool* need_iv,
      CryptoPP::member_ptr<CryptoPP::SymmetricCipher>* m_cipher,
      CryptoPP::member_ptr<CryptoPP::StreamTransformationFilter>* m_filter);

  void BuildAuthEncCipher(
      bool* need_iv,
      CryptoPP::member_ptr<CryptoPP::AuthenticatedSymmetricCipher>* m_cipher,
      CryptoPP::member_ptr<CryptoPP::AuthenticatedEncryptionFilter>* m_filter);

  void BuildAuthDecCipher(
      bool* need_iv,
      CryptoPP::member_ptr<CryptoPP::AuthenticatedSymmetricCipher>* m_cipher,
      CryptoPP::member_ptr<CryptoPP::AuthenticatedDecryptionFilter>* m_filter);

  std::string aes_cipher_name_;
  int iv_size_;
  int tag_size_;
  std::string iv_;
  bool is_authenticated_cipher_{false};
};

}  // namespace framework
}  // namespace paddle
