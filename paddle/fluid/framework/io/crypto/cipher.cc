// Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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

#include "paddle/fluid/framework/io/crypto/cipher.h"

#include "paddle/fluid/framework/io/crypto/aes_cipher.h"
#include "paddle/fluid/framework/io/crypto/cipher_utils.h"
#include "paddle/fluid/platform/enforce.h"

namespace paddle::framework {

std::shared_ptr<Cipher> CipherFactory::CreateCipher(
    const std::string& config_file) {
  std::string cipher_name;
  int iv_size = 0;
  int tag_size = 0;
  std::unordered_map<std::string, std::string> config;
  if (!config_file.empty()) {
    config = CipherUtils::LoadConfig(config_file);
    CipherUtils::GetValue<std::string>(config, "cipher_name", &cipher_name);
  } else {
    // set default cipher name
    cipher_name = "AES_CTR_NoPadding";
  }
  if (cipher_name.find("AES") != cipher_name.npos) {
    auto ret = std::make_shared<AESCipher>();
    // if not set iv_size, set default value
    if (config_file.empty() ||
        !CipherUtils::GetValue<int>(config, "iv_size", &iv_size)) {
      iv_size = CipherUtils::AES_DEFAULT_IV_SIZE;
    }
    // if not set tag_size, set default value
    if (config_file.empty() ||
        !CipherUtils::GetValue<int>(config, "tag_size", &tag_size)) {
      tag_size = CipherUtils::AES_DEFAULT_IV_SIZE;
    }
    ret->Init(cipher_name, iv_size, tag_size);
    return ret;
  } else {
    PADDLE_THROW(common::errors::InvalidArgument(
        "Invalid cipher name is specified. "
        "Please check you have specified valid cipher"
        " name in CryptoProperties."));
  }
  return nullptr;
}

}  // namespace paddle::framework
