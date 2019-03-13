#   Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import unittest
import paddle.fluid as fluid
from paddle.fluid import core


class TestCryption(unittest.TestCase):
    def test_cryption(self):
        input_str = "0123456789abcdef0123456789abcdef"
        cryptor = core.Cryption.getCryptor()
        encrypt_str = cryptor.encryptInMemory(input_str)
        decrypt_str = cryptor.decryptInMemory(encrypt_str)
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_with_file(self):
        input_str = "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e"
        file_path = "./__str__"
        # encrypt
        encryptor = core.Cryption.getCryptor()
        encrypt_str = encryptor.encryptInMemory(input_str)
        with open(file_path, "wb") as f:
            f.write(encrypt_str)

        # decrypt
        decryptor = core.Cryption.getCryptor()
        with open(file_path, "rb") as f:
            load_str = f.read()
        decrypt_str = decryptor.decryptInMemory(load_str)

        # compare
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_with_file2(self):
        input_str = "d5273cb3bb524ada0612bd6241f08810d5273cb3bb524ada0612bd6241f088102eaf1f02c0234c9ae15c7f4df66147e82fb188675c2eb39cc229c6868f84705c139df128f83df1589be04e68a7c9a668d3a62aaa6c486acb096341b122e793ee8a3c30cc3dbfe54ab3994802ef69989c26307bf93a091c7bce5c0115a2895bc65ad0e5a592fc3553aba94a6f7ea54389d6a25c7921c5051f5a334bd1b4cc5a683c5d13e2c649306efcba9a2987495d39618e2e9313be64f1d694ab6144a77d6384d0a1b78302994bfc2b1df8de9e4162b177d11daf199afa4bd1d1e631bc89b55950b6a2e4caa3d24870767ed2fbb9615337926aebcdfe04457d1edf4b1e0828a14fc6a9277a12dddeaeb58b15c9a65bf6415f0941e582f8ad77371b5d1c08329df4945452a7a62bcdf3143b55fea27b80d4d776b75f8662ca9d2d22aa9956c869f94f49125a17ebbc12733e280dbd0a1e05f03eb38bd390ec164223723cdd92214cbf233502849674e14d46903ee02c29ee56a8fdc0a511aad8833b55e0206d67ec0b9fc703227c3352c100"
        file_path = "./__str__"
        # encrypt
        encryptor = core.Cryption.getCryptor()
        encrypt_str = encryptor.encryptInMemory(input_str)
        with open(file_path, "wb") as f:
            f.write(encrypt_str)

        # decrypt
        decryptor = core.Cryption.getCryptor()
        with open(file_path, "rb") as f:
            load_str = f.read()
        decrypt_str = decryptor.decryptInMemory(load_str)

        # compare
        self.assertEqual(input_str, decrypt_str)


if __name__ == '__main__':
    unittest.main()
