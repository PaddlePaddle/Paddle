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
        input_str = str.encode("0123456789abcdef0123456789abcdef")
        cryptor = core.Cryption.getCryptor()
        encrypt_str, encrypt_len = cryptor.encrypt_memory_with_key_in_memory(
            input_str)
        decrypt_str = cryptor.decrypt_memory_with_key_in_memory(encrypt_str,
                                                                encrypt_len)
        self.assertEqual(input_str, decrypt_str)

    # TODO(chenwhql): long string test failed, 
    #                 maybe caused by the type conversion between pyhind and C++.
    #                 bytes to char*, but the bytes variable include '\0'.
    def test_cryption_with_longstr(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e"
        )
        cryptor = core.Cryption.getCryptor()
        encrypt_str, encrypt_len = cryptor.encrypt_memory_with_key_in_memory(
            input_str)
        decrypt_str = cryptor.decrypt_memory_with_key_in_memory(encrypt_str,
                                                                encrypt_len)
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_with_file(self):
        input_str = str.encode("0123456789abcdef0123456789abcdef")
        input_path = "./__str__"
        encrypt_path = "./__encrypt__"
        decrypt_path = "./__decrypt__"

        with open(input_path, "wb") as f:
            f.write(input_str)

        # encrypt
        encryptor = core.Cryption.getCryptor()
        encryptor.encrypt_file_with_key_in_file(input_path, encrypt_path)

        # decrypt
        decryptor = core.Cryption.getCryptor()
        decryptor.decrypt_file_with_key_in_file(encrypt_path, decrypt_path)

        with open(decrypt_path, "rb") as f:
            decrypt_str = f.read()

        self.assertEqual(input_str, decrypt_str)

    def test_cryption_with_long_file(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e"
        )
        input_path = "./__str__"
        encrypt_path = "./__encrypt__"
        decrypt_path = "./__decrypt__"

        with open(input_path, "wb") as f:
            f.write(input_str)

        # encrypt
        encryptor = core.Cryption.getCryptor()
        encryptor.encrypt_file_with_key_in_file(input_path, encrypt_path)

        # decrypt
        decryptor = core.Cryption.getCryptor()
        decryptor.decrypt_file_with_key_in_file(encrypt_path, decrypt_path)

        with open(decrypt_path, "rb") as f:
            decrypt_str = f.read()

        # compare
        self.assertEqual(input_str, decrypt_str)


if __name__ == '__main__':
    unittest.main()
