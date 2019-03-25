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

#include <fstream>
#include <iostream>
#include <string>

#include "glog/logging.h"
#include "gtest/gtest.h"
#include "paddle/fluid/framework/cryption.h"

#define TEST_BUF_SIZE 32

using paddle::framework::Cryption;

void readFile(const std::string& path, char* buf, size_t len) {
  std::ifstream fin;

  fin.open(path, std::ios::binary);
  fin.read(buf, len);

  buf[len] = '\0';

  fin.close();
  return;
}

void writeFile(const std::string& path, const char* buf, size_t len) {
  std::ofstream fout;

  fout.open(path, std::ios::binary);
  fout.write(buf, len);

  fout.close();
  return;
}

TEST(Cryption, OnceInitTest) {
  Cryption& c1 = *Cryption::GetCryptorInstance();
  Cryption& c2 = *Cryption::GetCryptorInstance();
  char buf1[TEST_BUF_SIZE];
  char buf2[TEST_BUF_SIZE];

  readFile(c1.GetEncryptKeyPath(), buf1, TEST_BUF_SIZE);
  readFile(c2.GetEncryptKeyPath(), buf2, TEST_BUF_SIZE);

  std::string key1 = paddle::framework::ConvertHexString(buf1, TEST_BUF_SIZE);
  std::string key2 = paddle::framework::ConvertHexString(buf2, TEST_BUF_SIZE);

  EXPECT_EQ(key1, key2);
}

TEST(Cryption, ShortStringDivisible16Test) {
  Cryption& c = *Cryption::GetCryptorInstance();
  std::string inputStr("0123456789abcdef0123456789abcdef");

  // get encrypt len
  size_t cryptLen = ((inputStr.size() + 15) / 16) * 16;
  size_t padLen = cryptLen - inputStr.size();
  if (padLen > 0) inputStr.append(std::string(padLen, '0'));

  std::string encryptStr = c.EncryptInMemory(inputStr.data(), cryptLen);
  std::string decryptStr = c.DecryptInMemory(encryptStr.data(), cryptLen);

  EXPECT_EQ(inputStr, decryptStr);
}

TEST(Cryption, ShortStringUndivisible16Test) {
  Cryption& c = *Cryption::GetCryptorInstance();
  std::string inputStr("0123456789abcdef0123456789abc");

  // get encrypt len
  size_t cryptLen = ((inputStr.size() + 15) / 16) * 16;
  size_t padLen = cryptLen - inputStr.size();
  if (padLen > 0) inputStr.append(std::string(padLen, '0'));

  std::string encryptStr = c.EncryptInMemory(inputStr.data(), cryptLen);
  std::string decryptStr = c.DecryptInMemory(encryptStr.data(), cryptLen);

  EXPECT_EQ(inputStr, decryptStr);
}

TEST(Cryption, LongStringDivisible16Test) {
  Cryption& c = *Cryption::GetCryptorInstance();
  std::string inputStr =
      "d5273cb3bb524ada0612bd6241f08810d5273cb3bb524ada0612bd6241f088102eaf1f02"
      "c0234c9ae15c7f4df66147e82fb188675c2eb39cc229c6868f84705c139df128f83df158"
      "9be04e68a7c9a668d3a62aaa6c486acb096341b122e793ee8a3c30cc3dbfe54ab3994802"
      "ef69989c26307bf93a091c7bce5c0115a2895bc65ad0e5a592fc3553aba94a6f7ea54389"
      "d6a25c7921c5051f5a334bd1b4cc5a683c5d13e2c649306efcba9a2987495d39618e2e93"
      "13be64f1d694ab6144a77d6384d0a1b78302994bfc2b1df8de9e4162b177d11daf199afa"
      "4bd1d1e631bc89b55950b6a2e4caa3d24870767ed2fbb9615337926aebcdfe04457d1edf"
      "4b1e0828a14fc6a9277a12dddeaeb58b15c9a65bf6415f0941e582f8ad77371b5d1c0832"
      "9df4945452a7a62bcdf3143b55fea27b80d4d776b75f8662ca9d2d22aa9956c869f94f49"
      "125a17ebbc12733e280dbd0a1e05f03eb38bd390ec164223723cdd92214cbf2335028496"
      "74e14d46903ee02c29ee56a8fdc0a511aad8833b55e0206d67ec0b9fc703227c3352c100"
      "9cba86f264a35f0036ce4ba03e1184a3a63eac73ae74e772e13560761cdbf717b6c756b5"
      "250fb72a28ddfea28907b510606b702590adb8589c105932f8a17d489cb60a913f78b814"
      "fabe538da023afc376eded5983dc3d57f024c9cbc2606b7a2f5b29d7ffbce662aee304ca"
      "fa746ca8bca4cbid";

  // get encrypt len
  size_t cryptLen = ((inputStr.size() + 15) / 16) * 16;
  size_t padLen = cryptLen - inputStr.size();
  if (padLen > 0) inputStr.append(std::string(padLen, '0'));

  std::string encryptStr = c.EncryptInMemory(inputStr.data(), cryptLen);
  std::string decryptStr = c.DecryptInMemory(encryptStr.data(), cryptLen);

  EXPECT_EQ(inputStr, decryptStr);
}

TEST(Cryption, LongStringiUndivisible16Test) {
  Cryption& c = *Cryption::GetCryptorInstance();
  std::string inputStr =
      "d5273cb3bb524ada0612bd6241f08810d5273cb3bb524ada0612bd6241f088102eaf1f02"
      "c0234c9ae15c7f4df66147e82fb188675c2eb39cc229c6868f84705c139df128f83df158"
      "9be04e68a7c9a668d3a62aaa6c486acb096341b122e793ee8a3c30cc3dbfe54ab3994802"
      "ef69989c26307bf93a091c7bce5c0115a2895bc65ad0e5a592fc3553aba94a6f7ea54389"
      "d6a25c7921c5051f5a334bd1b4cc5a683c5d13e2c649306efcba9a2987495d39618e2e93"
      "13be64f1d694ab6144a77d6384d0a1b78302994bfc2b1df8de9e4162b177d11daf199afa"
      "4bd1d1e631bc89b55950b6a2e4caa3d24870767ed2fbb9615337926aebcdfe04457d1edf"
      "4b1e0828a14fc6a9277a12dddeaeb58b15c9a65bf6415f0941e582f8ad77371b5d1c0832"
      "9df4945452a7a62bcdf3143b55fea27b80d4d776b75f8662ca9d2d22aa9956c869f94f49"
      "125a17ebbc12733e280dbd0a1e05f03eb38bd390ec164223723cdd92214cbf2335028496"
      "74e14d46903ee02c29ee56a8fdc0a511aad8833b55e0206d67ec0b9fc703227c3352c100"
      "9cba86f264a35f0036ce4ba03e1184a3a63eac73ae74e772e13560761cdbf717b6c756b5"
      "250fb72a28ddfea28907b510606b702590adb8589c105932f8a17d489cb60a913f78b814"
      "fabe538da023afc376eded5983dc3d57f024c9cbc2606b7a2f5b29d7ffbce662aee304ca"
      "fa746ca8bca4cb";

  // get encrypt len
  size_t cryptLen = ((inputStr.size() + 15) / 16) * 16;
  size_t padLen = cryptLen - inputStr.size();
  if (padLen > 0) inputStr.append(std::string(padLen, '0'));

  std::string encryptStr = c.EncryptInMemory(inputStr.data(), cryptLen);
  std::string decryptStr = c.DecryptInMemory(encryptStr.data(), cryptLen);

  EXPECT_EQ(inputStr, decryptStr);
}

TEST(Cryption, CryptWithFile) {
  Cryption& c = *Cryption::GetCryptorInstance();
  const char* inputStr = "0123456789abcdef0123456789abcdef";
  int strLen = strlen(inputStr);
  char* inputStrCopy = new char[strLen];
  char* encryptStr = new char[strLen];
  char* decryptStr = new char[strLen];

  std::string inputPath("./__input_str__");
  std::string encryptPath("./__encrypt_str__");
  std::string decryptPath("./__decrypt_str__");

  if (inputStrCopy == nullptr || encryptStr == nullptr ||
      decryptStr == nullptr) {
    LOG(ERROR) << "alloc error";
    return;
  }

  writeFile(inputPath, inputStr, strlen(inputStr));
  readFile(inputPath, inputStrCopy, strLen);

  c.EncryptInFile(inputPath, encryptPath);
  readFile(encryptPath, encryptStr, strLen);

  c.DecryptInFile(encryptPath, decryptPath);
  readFile(decryptPath, decryptStr, strLen);

  EXPECT_STREQ(inputStr, decryptStr);

  delete[] inputStrCopy;
  delete[] encryptStr;
  delete[] decryptStr;
}

TEST(Cryption, CryptWithLongFileDivisible16) {
  Cryption& c = *Cryption::GetCryptorInstance();
  const char* inputStr =
      "d5273cb3bb524ada0612bd6241f08810d5273cb3bb524ada0612bd6241f088102eaf1f02"
      "c0234c9ae15c7f4df66147e82fb188675c2eb39cc229c6868f84705c139df128f83df158"
      "9be04e68a7c9a668d3a62aaa6c486acb096341b122e793ee8a3c30cc3dbfe54ab3994802"
      "ef69989c26307bf93a091c7bce5c0115a2895bc65ad0e5a592fc3553aba94a6f7ea54389"
      "d6a25c7921c5051f5a334bd1b4cc5a683c5d13e2c649306efcba9a2987495d39618e2e93"
      "13be64f1d694ab6144a77d6384d0a1b78302994bfc2b1df8de9e4162b177d11daf199afa"
      "4bd1d1e631bc89b55950b6a2e4caa3d24870767ed2fbb9615337926aebcdfe04457d1edf"
      "4b1e0828a14fc6a9277a12dddeaeb58b15c9a65bf6415f0941e582f8ad77371b5d1c0832"
      "9df4945452a7a62bcdf3143b55fea27b80d4d776b75f8662ca9d2d22aa9956c869f94f49"
      "125a17ebbc12733e280dbd0a1e05f03eb38bd390ec164223723cdd92214cbf2335028496"
      "74e14d46903ee02c29ee56a8fdc0a511aad8833b55e0206d67ec0b9fc703227c3352c100"
      "9cba86f264a35f0036ce4ba03e1184a3a63eac73ae74e772e13560761cdbf717b6c756b5"
      "250fb72a28ddfea28907b510606b702590adb8589c105932f8a17d489cb60a913f78b814"
      "fabe538da023afc376eded5983dc3d57f024c9cbc2606b7a2f5b29d7ffbce662aee304ca"
      "fa746ca8bca4cbde";
  int strLen = strlen(inputStr);
  char* inputStrCopy = new char[strLen];
  char* encryptStr = new char[strLen];
  char* decryptStr = new char[strLen];

  std::string inputPath("./__input_str1__");
  std::string encryptPath("./__encrypt_str1__");
  std::string decryptPath("./__decrypt_str1__");

  if (inputStrCopy == nullptr || encryptStr == nullptr ||
      decryptStr == nullptr) {
    LOG(ERROR) << "alloc error";
    return;
  }

  writeFile(inputPath, inputStr, strlen(inputStr));
  readFile(inputPath, inputStrCopy, strLen);

  c.EncryptInFile(inputPath, encryptPath);
  readFile(encryptPath, encryptStr, strLen);

  c.DecryptInFile(encryptPath, decryptPath);
  readFile(decryptPath, decryptStr, strLen);

  EXPECT_STREQ(inputStr, decryptStr);

  delete[] inputStrCopy;
  delete[] encryptStr;
  delete[] decryptStr;
}

TEST(Cryption, CryptWithLongFileUndivisible16) {
  Cryption& c = *Cryption::GetCryptorInstance();
  const char* inputStr =
      "d5273cb3bb524ada0612bd6241f08810d5273cb3bb524ada0612bd6241f088102eaf1f02"
      "c0234c9ae15c7f4df66147e82fb188675c2eb39cc229c6868f84705c139df128f83df158"
      "9be04e68a7c9a668d3a62aaa6c486acb096341b122e793ee8a3c30cc3dbfe54ab3994802"
      "ef69989c26307bf93a091c7bce5c0115a2895bc65ad0e5a592fc3553aba94a6f7ea54389"
      "d6a25c7921c5051f5a334bd1b4cc5a683c5d13e2c649306efcba9a2987495d39618e2e93"
      "13be64f1d694ab6144a77d6384d0a1b78302994bfc2b1df8de9e4162b177d11daf199afa"
      "4bd1d1e631bc89b55950b6a2e4caa3d24870767ed2fbb9615337926aebcdfe04457d1edf"
      "4b1e0828a14fc6a9277a12dddeaeb58b15c9a65bf6415f0941e582f8ad77371b5d1c0832"
      "9df4945452a7a62bcdf3143b55fea27b80d4d776b75f8662ca9d2d22aa9956c869f94f49"
      "125a17ebbc12733e280dbd0a1e05f03eb38bd390ec164223723cdd92214cbf2335028496"
      "74e14d46903ee02c29ee56a8fdc0a511aad8833b55e0206d67ec0b9fc703227c3352c100"
      "9cba86f264a35f0036ce4ba03e1184a3a63eac73ae74e772e13560761cdbf717b6c756b5"
      "250fb72a28ddfea28907b510606b702590adb8589c105932f8a17d489cb60a913f78b814"
      "fabe538da023afc376eded5983dc3d57f024c9cbc2606b7a2f5b29d7ffbce662aee304ca"
      "fa746ca8bca4";
  int strLen = strlen(inputStr);
  char* inputStrCopy = new char[strLen];
  char* encryptStr = new char[strLen];
  char* decryptStr = new char[strLen];

  std::string inputPath("./__input_str1__");
  std::string encryptPath("./__encrypt_str1__");
  std::string decryptPath("./__decrypt_str1__");

  if (inputStrCopy == nullptr || encryptStr == nullptr ||
      decryptStr == nullptr) {
    LOG(ERROR) << "alloc error";
    return;
  }

  writeFile(inputPath, inputStr, strlen(inputStr));
  readFile(inputPath, inputStrCopy, strLen);
  LOG(INFO) << "inputFile: " << std::string(inputStrCopy);

  c.EncryptInFile(inputPath, encryptPath);
  readFile(encryptPath, encryptStr, strLen);
  LOG(INFO) << "encryptFile: " << std::string(encryptStr);

  c.DecryptInFile(encryptPath, decryptPath);
  readFile(decryptPath, decryptStr, strLen);
  LOG(INFO) << "decryptFile: " << std::string(decryptStr);

  EXPECT_STREQ(inputStr, decryptStr);

  delete[] inputStrCopy;
  delete[] encryptStr;
  delete[] decryptStr;
}
