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

#include "paddle/fluid/framework/io/crypto/aes_cipher.h"

#include <cryptopp/cryptlib.h>
#include <glog/logging.h>
#include <gtest/gtest.h>

#include <fstream>
#include <string>

#include "paddle/fluid/framework/io/crypto/cipher_utils.h"

namespace paddle {
namespace framework {

class AESTest : public ::testing::Test {
 public:
  std::string key;

  void SetUp() override { key = CipherUtils::GenKey(256); }
  static void GenConfigFile(const std::string& cipher_name);
};

void AESTest::GenConfigFile(const std::string& cipher_name) {
  std::ofstream fout("aes_test.conf");
  fout << "cipher_name : " << cipher_name << std::endl;
  fout.close();
}

TEST_F(AESTest, security_string) {
  std::vector<std::string> name_list({"AES_CTR_NoPadding",
                                      "AES_CBC_PKCSPadding",
                                      "AES_ECB_PKCSPadding",
                                      "AES_GCM_NoPadding"});
  const std::string plaintext("hello world.");
  bool is_throw = false;
  for (auto& i : name_list) {
    AESTest::GenConfigFile(i);
    try {
      auto cipher = CipherFactory::CreateCipher("aes_test.conf");
      std::string ciphertext = cipher->Encrypt(plaintext, AESTest::key);

      std::string plaintext1 = cipher->Decrypt(ciphertext, AESTest::key);
      EXPECT_EQ(plaintext, plaintext1);
    } catch (CryptoPP::Exception& e) {
      is_throw = true;
      LOG(ERROR) << e.what();
    }
    EXPECT_FALSE(is_throw);
  }
}

TEST_F(AESTest, security_vector) {
  std::vector<std::string> name_list({"AES_CTR_NoPadding",
                                      "AES_CBC_PKCSPadding",
                                      "AES_ECB_PKCSPadding",
                                      "AES_GCM_NoPadding"});
  std::vector<int> input{1, 2, 3, 4};
  bool is_throw = false;
  for (auto& i : name_list) {
    AESTest::GenConfigFile(i);
    try {
      auto cipher = CipherFactory::CreateCipher("aes_test.conf");
      for (auto& i : input) {
        std::string ciphertext =
            cipher->Encrypt(std::to_string(i), AESTest::key);

        std::string plaintext = cipher->Decrypt(ciphertext, AESTest::key);

        int output = std::stoi(plaintext);

        EXPECT_EQ(i, output);
      }
    } catch (CryptoPP::Exception& e) {
      is_throw = true;
      LOG(ERROR) << e.what();
    }
    EXPECT_FALSE(is_throw);
  }
}

TEST_F(AESTest, encrypt_to_file) {
  std::vector<std::string> name_list({"AES_CTR_NoPadding",
                                      "AES_CBC_PKCSPadding",
                                      "AES_ECB_PKCSPadding",
                                      "AES_GCM_NoPadding"});
  const std::string plaintext("hello world.");
  std::string filename("aes_test.ciphertext");
  bool is_throw = false;
  for (auto& i : name_list) {
    AESTest::GenConfigFile(i);
    try {
      auto cipher = CipherFactory::CreateCipher("aes_test.conf");
      cipher->EncryptToFile(plaintext, AESTest::key, filename);
      std::string plaintext1 = cipher->DecryptFromFile(AESTest::key, filename);
      EXPECT_EQ(plaintext, plaintext1);
    } catch (CryptoPP::Exception& e) {
      is_throw = true;
      LOG(ERROR) << e.what();
    }
    EXPECT_FALSE(is_throw);
  }
}

}  // namespace framework
}  // namespace paddle
