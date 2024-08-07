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

#include "paddle/fluid/framework/io/crypto/aes_cipher.h"

#include <cryptopp/aes.h>
#include <cryptopp/ccm.h>
#include <cryptopp/cryptlib.h>
#include <cryptopp/filters.h>
#include <cryptopp/gcm.h>
#include <cryptopp/modes.h>
#include <cryptopp/smartptr.h>

#include <set>
#include <string>

#include "paddle/fluid/framework/io/crypto/cipher_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework {

void AESCipher::Init(const std::string& cipher_name,
                     const int& iv_size,
                     const int& tag_size) {
  aes_cipher_name_ = cipher_name;
  iv_size_ = iv_size;
  tag_size_ = tag_size;
  std::set<std::string> authented_cipher_set{"AES_GCM_NoPadding"};
  if (authented_cipher_set.find(cipher_name) != authented_cipher_set.end()) {
    is_authenticated_cipher_ = true;
  }
}

std::string AESCipher::EncryptInternal(const std::string& plaintext,
                                       const std::string& key) {
  CryptoPP::member_ptr<CryptoPP::SymmetricCipher> m_cipher;
  CryptoPP::member_ptr<CryptoPP::StreamTransformationFilter> m_filter;
  bool need_iv = false;
  const unsigned char* key_char =
      reinterpret_cast<const unsigned char*>(&(key.at(0)));
  BuildCipher(true, &need_iv, &m_cipher, &m_filter);
  if (need_iv) {
    iv_ = CipherUtils::GenKey(iv_size_);
    m_cipher->SetKeyWithIV(key_char,
                           key.size(),
                           reinterpret_cast<const unsigned char*>(&(iv_.at(0))),
                           iv_.size());
  } else {
    m_cipher->SetKey(key_char, key.size());
  }

  std::string ciphertext;
  m_filter->Attach(new CryptoPP::StringSink(ciphertext));
  CryptoPP::Redirector* filter_redirector = new CryptoPP::Redirector(*m_filter);
  CryptoPP::StringSource ss(plaintext, true, filter_redirector);
  if (need_iv) {
    return iv_ + ciphertext;
  }

  return ciphertext;
}

std::string AESCipher::DecryptInternal(const std::string& ciphertext,
                                       const std::string& key) {
  CryptoPP::member_ptr<CryptoPP::SymmetricCipher> m_cipher;
  CryptoPP::member_ptr<CryptoPP::StreamTransformationFilter> m_filter;
  bool need_iv = false;
  const unsigned char* key_char =
      reinterpret_cast<const unsigned char*>(&(key.at(0)));
  BuildCipher(false, &need_iv, &m_cipher, &m_filter);
  int ciphertext_beg = 0;
  if (need_iv) {
    iv_ = ciphertext.substr(0, iv_size_ / 8);
    ciphertext_beg = iv_size_ / 8;
    m_cipher->SetKeyWithIV(key_char,
                           key.size(),
                           reinterpret_cast<const unsigned char*>(&(iv_.at(0))),
                           iv_.size());
  } else {
    m_cipher->SetKey(key_char, key.size());
  }
  std::string plaintext;
  m_filter->Attach(new CryptoPP::StringSink(plaintext));
  CryptoPP::Redirector* filter_redirector = new CryptoPP::Redirector(*m_filter);
  CryptoPP::StringSource ss(
      ciphertext.substr(ciphertext_beg), true, filter_redirector);

  return plaintext;
}

std::string AESCipher::AuthenticatedEncryptInternal(
    const std::string& plaintext, const std::string& key) {
  CryptoPP::member_ptr<CryptoPP::AuthenticatedSymmetricCipher> m_cipher;
  CryptoPP::member_ptr<CryptoPP::AuthenticatedEncryptionFilter> m_filter;
  bool need_iv = false;
  const unsigned char* key_char =
      reinterpret_cast<const unsigned char*>(&(key.at(0)));
  BuildAuthEncCipher(&need_iv, &m_cipher, &m_filter);
  if (need_iv) {
    iv_ = CipherUtils::GenKey(iv_size_);
    m_cipher->SetKeyWithIV(key_char,
                           key.size(),
                           reinterpret_cast<const unsigned char*>(&(iv_.at(0))),
                           iv_.size());
  } else {
    m_cipher->SetKey(key_char, key.size());
  }

  std::string ciphertext;
  m_filter->Attach(new CryptoPP::StringSink(ciphertext));
  CryptoPP::Redirector* filter_redirector = new CryptoPP::Redirector(*m_filter);
  CryptoPP::StringSource ss(plaintext, true, filter_redirector);
  if (need_iv) {
    ciphertext = iv_.append(ciphertext);
  }

  return ciphertext;
}

std::string AESCipher::AuthenticatedDecryptInternal(
    const std::string& ciphertext, const std::string& key) {
  CryptoPP::member_ptr<CryptoPP::AuthenticatedSymmetricCipher> m_cipher;
  CryptoPP::member_ptr<CryptoPP::AuthenticatedDecryptionFilter> m_filter;
  bool need_iv = false;
  const unsigned char* key_char =
      reinterpret_cast<const unsigned char*>(&(key.at(0)));
  BuildAuthDecCipher(&need_iv, &m_cipher, &m_filter);
  int ciphertext_beg = 0;
  if (need_iv) {
    iv_ = ciphertext.substr(0, iv_size_ / 8);
    ciphertext_beg = iv_size_ / 8;
    m_cipher->SetKeyWithIV(key_char,
                           key.size(),
                           reinterpret_cast<const unsigned char*>(&(iv_.at(0))),
                           iv_.size());
  } else {
    m_cipher->SetKey(key_char, key.size());
  }
  std::string plaintext;
  m_filter->Attach(new CryptoPP::StringSink(plaintext));
  CryptoPP::Redirector* filter_redirector = new CryptoPP::Redirector(*m_filter);
  CryptoPP::StringSource ss(
      ciphertext.substr(ciphertext_beg), true, filter_redirector);
  PADDLE_ENFORCE_EQ(
      m_filter->GetLastResult(),
      true,
      common::errors::InvalidArgument("Integrity check failed. "
                                      "Invalid ciphertext input."));
  return plaintext;
}

void AESCipher::BuildCipher(
    bool for_encrypt,
    bool* need_iv,
    CryptoPP::member_ptr<CryptoPP::SymmetricCipher>* m_cipher,
    CryptoPP::member_ptr<CryptoPP::StreamTransformationFilter>* m_filter) {
  if (aes_cipher_name_ == "AES_ECB_PKCSPadding" && for_encrypt) {
    m_cipher->reset(new CryptoPP::ECB_Mode<CryptoPP::AES>::Encryption);
    m_filter->reset(new CryptoPP::StreamTransformationFilter(
        **m_cipher, nullptr, CryptoPP::BlockPaddingSchemeDef::PKCS_PADDING));
  } else if (aes_cipher_name_ == "AES_ECB_PKCSPadding" && !for_encrypt) {
    m_cipher->reset(new CryptoPP::ECB_Mode<CryptoPP::AES>::Decryption);
    m_filter->reset(new CryptoPP::StreamTransformationFilter(
        **m_cipher, nullptr, CryptoPP::BlockPaddingSchemeDef::PKCS_PADDING));
  } else if (aes_cipher_name_ == "AES_CBC_PKCSPadding" && for_encrypt) {
    m_cipher->reset(new CryptoPP::CBC_Mode<CryptoPP::AES>::Encryption);
    *need_iv = true;
    m_filter->reset(new CryptoPP::StreamTransformationFilter(
        **m_cipher, nullptr, CryptoPP::BlockPaddingSchemeDef::PKCS_PADDING));
  } else if (aes_cipher_name_ == "AES_CBC_PKCSPadding" && !for_encrypt) {
    m_cipher->reset(new CryptoPP::CBC_Mode<CryptoPP::AES>::Decryption);
    *need_iv = true;
    m_filter->reset(new CryptoPP::StreamTransformationFilter(
        **m_cipher, nullptr, CryptoPP::BlockPaddingSchemeDef::PKCS_PADDING));
  } else if (aes_cipher_name_ == "AES_CTR_NoPadding" && for_encrypt) {
    m_cipher->reset(new CryptoPP::CTR_Mode<CryptoPP::AES>::Encryption);
    *need_iv = true;
    m_filter->reset(new CryptoPP::StreamTransformationFilter(
        **m_cipher, nullptr, CryptoPP::BlockPaddingSchemeDef::NO_PADDING));
  } else if (aes_cipher_name_ == "AES_CTR_NoPadding" && !for_encrypt) {
    m_cipher->reset(new CryptoPP::CTR_Mode<CryptoPP::AES>::Decryption);
    *need_iv = true;
    m_filter->reset(new CryptoPP::StreamTransformationFilter(
        **m_cipher, nullptr, CryptoPP::BlockPaddingSchemeDef::NO_PADDING));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Create cipher error. "
        "Cipher name %s is error, or has not been implemented.",
        aes_cipher_name_));
  }
}

void AESCipher::BuildAuthEncCipher(
    bool* need_iv,
    CryptoPP::member_ptr<CryptoPP::AuthenticatedSymmetricCipher>* m_cipher,
    CryptoPP::member_ptr<CryptoPP::AuthenticatedEncryptionFilter>* m_filter) {
  if (aes_cipher_name_ == "AES_GCM_NoPadding") {
    m_cipher->reset(new CryptoPP::GCM<CryptoPP::AES>::Encryption);
    *need_iv = true;
    m_filter->reset(new CryptoPP::AuthenticatedEncryptionFilter(
        **m_cipher,
        nullptr,
        false,
        tag_size_ / 8,
        CryptoPP::DEFAULT_CHANNEL,
        CryptoPP::BlockPaddingSchemeDef::NO_PADDING));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Create cipher error. "
        "Cipher name %s is error, or has not been implemented.",
        aes_cipher_name_));
  }
}

void AESCipher::BuildAuthDecCipher(
    bool* need_iv,
    CryptoPP::member_ptr<CryptoPP::AuthenticatedSymmetricCipher>* m_cipher,
    CryptoPP::member_ptr<CryptoPP::AuthenticatedDecryptionFilter>* m_filter) {
  if (aes_cipher_name_ == "AES_GCM_NoPadding") {
    m_cipher->reset(new CryptoPP::GCM<CryptoPP::AES>::Decryption);
    *need_iv = true;
    m_filter->reset(new CryptoPP::AuthenticatedDecryptionFilter(
        **m_cipher,
        nullptr,
        CryptoPP::AuthenticatedDecryptionFilter::DEFAULT_FLAGS,
        tag_size_ / 8,
        CryptoPP::BlockPaddingSchemeDef::NO_PADDING));
  } else {
    PADDLE_THROW(common::errors::Unimplemented(
        "Create cipher error. "
        "Cipher name %s is error, or has not been implemented.",
        aes_cipher_name_));
  }
}

std::string AESCipher::Encrypt(const std::string& plaintext,
                               const std::string& key) {
  return is_authenticated_cipher_ ? AuthenticatedEncryptInternal(plaintext, key)
                                  : EncryptInternal(plaintext, key);
}

std::string AESCipher::Decrypt(const std::string& ciphertext,
                               const std::string& key) {
  return is_authenticated_cipher_
             ? AuthenticatedDecryptInternal(ciphertext, key)
             : DecryptInternal(ciphertext, key);
}

void AESCipher::EncryptToFile(const std::string& plaintext,
                              const std::string& key,
                              const std::string& filename) {
  std::ofstream fout(filename, std::ios::binary);
  std::string ciphertext = this->Encrypt(plaintext, key);
  fout.write(ciphertext.data(), ciphertext.size());  // NOLINT
  fout.close();
}

std::string AESCipher::DecryptFromFile(const std::string& key,
                                       const std::string& filename) {
  std::ifstream fin(filename, std::ios::binary);
  std::string ciphertext{std::istreambuf_iterator<char>(fin),
                         std::istreambuf_iterator<char>()};
  fin.close();
  return Decrypt(ciphertext, key);
}

}  // namespace paddle::framework
