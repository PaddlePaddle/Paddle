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
    def _align_bytes_to_16(self, byte_str):
        result_len = ((len(byte_str) + 15) // 16) * 16
        pad_len = result_len - len(byte_str)
        if pad_len > 0:
            byte_str = bytearray(byte_str)
            while pad_len > 0:
                byte_str.append(0)
                pad_len = pad_len - 1
            byte_str = bytes(byte_str)
        return byte_str, result_len

    def test_cryption_in_memory_divisible_16(self):
        input_str = str.encode("0123456789abcdef0123456789abcdef")

        # get encrypt len
        input_str, crypt_len = self._align_bytes_to_16(input_str)

        cryptor = core.Cryption.get_cryptor()
        encrypt_str = cryptor.encrypt_in_memory(input_str, crypt_len)
        decrypt_str = cryptor.decrypt_in_memory(encrypt_str, crypt_len)
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_memory_undivisible_16(self):
        input_str = str.encode("0123456789abcdef0123456789abc")

        # get encrypt len
        input_str, crypt_len = self._align_bytes_to_16(input_str)

        cryptor = core.Cryption.get_cryptor()
        encrypt_str = cryptor.encrypt_in_memory(input_str, crypt_len)
        decrypt_str = cryptor.decrypt_in_memory(encrypt_str, crypt_len)
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_memory_with_longstr_divisible_16(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e13"
        )

        input_str, crypt_len = self._align_bytes_to_16(input_str)

        cryptor = core.Cryption.get_cryptor()
        encrypt_str = cryptor.encrypt_in_memory(input_str, crypt_len)
        decrypt_str = cryptor.decrypt_in_memory(encrypt_str, crypt_len)
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_memory_with_longstr_undivisible_16(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6"
        )

        input_str, crypt_len = self._align_bytes_to_16(input_str)

        cryptor = core.Cryption.get_cryptor()
        encrypt_str = cryptor.encrypt_in_memory(input_str, crypt_len)
        decrypt_str = cryptor.decrypt_in_memory(encrypt_str, crypt_len)
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_memory_with_file(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e"
        )
        encrypt_path = "./__encrypt__"

        input_str, crypt_len = self._align_bytes_to_16(input_str)

        # encrypt
        encryptor = core.Cryption.get_cryptor()
        encrypt_str = encryptor.encrypt_in_memory(input_str, crypt_len)

        with open(encrypt_path, "wb") as f:
            f.write(encrypt_str)

        with open(encrypt_path, "rb") as f:
            load_encrypt_str = f.read()

        # decrypt
        decryptor = core.Cryption.get_cryptor()
        decrypt_str = decryptor.decrypt_in_memory(load_encrypt_str, crypt_len)

        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_file(self):
        input_str = str.encode("0123456789abcdef0123456789abcdef")
        input_path = "./__str__"
        encrypt_path = "./__encrypt__"
        decrypt_path = "./__decrypt__"

        with open(input_path, "wb") as f:
            f.write(input_str)

        # encrypt
        encryptor = core.Cryption.get_cryptor()
        encryptor.encrypt_in_file(input_path, encrypt_path)

        # decrypt
        decryptor = core.Cryption.get_cryptor()
        decryptor.decrypt_in_file(encrypt_path, decrypt_path)

        with open(decrypt_path, "rb") as f:
            decrypt_str = f.read()

        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_file_with_longstr(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e23"
        )
        input_path = "./__str__"
        encrypt_path = "./__encrypt__"
        decrypt_path = "./__decrypt__"

        with open(input_path, "wb") as f:
            f.write(input_str)

        # encrypt
        encryptor = core.Cryption.get_cryptor()
        encryptor.encrypt_in_file(input_path, encrypt_path)

        # decrypt
        decryptor = core.Cryption.get_cryptor()
        decryptor.decrypt_in_file(encrypt_path, decrypt_path)

        with open(decrypt_path, "rb") as f:
            decrypt_str = f.read()

        # compare
        self.assertEqual(input_str, decrypt_str)

    def test_cryption_in_file_with_longstr_undivisible_16(self):
        input_str = str.encode(
            "0123456789abcdef0123456789abcdef0abf4f080010ffffffffffffffffff011a0d0a0566657463681202080a18011a0c0a04666565641202080918011a1a0a0866635f302e775f30120c08071a080a060805100d100118011a270a0a66635f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a1e0a0178121708071a130a0f080510ffffffffffffffffff01100d100018001a2a0a0d7363616c655f302e746d705f30121708071a130a0f080510ffffffffffffffffff011001100018001a180a0866635f302e625f30120a08071a060a040805100118011a270a0a66635f302e746d705f31121708071a130a0f080510ffffffffffffffffff0110011000180022ec0e0a090a015812046665656412080a034f75741201781a046665656422090a03636f6c10001800220d0a076f705f726f6c6510001800220f0a0b6f705f726f6c655f7661721005228e0e0a0c6f705f63616c6c737461636b1005427d202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e2f706164646c652f666c7569642f6672616d65776f726b2e7079222c206c696e6520313339332c20696e205f70726570656e645f6f700a2020202061747472733d6b77617267732e67657428226174747273222c204e6f6e6529290a426b202046696c6520222f776f726b2f706164646c652f6275696c642f707974686f6e"
        )
        input_path = "./__str__"
        encrypt_path = "./__encrypt__"
        decrypt_path = "./__decrypt__"

        with open(input_path, "wb") as f:
            f.write(input_str)

        # encrypt
        encryptor = core.Cryption.get_cryptor()
        encryptor.encrypt_in_file(input_path, encrypt_path)

        # decrypt
        decryptor = core.Cryption.get_cryptor()
        decryptor.decrypt_in_file(encrypt_path, decrypt_path)

        with open(decrypt_path, "rb") as f:
            decrypt_str = f.read()

        # compare
        self.assertEqual(input_str, decrypt_str)


if __name__ == '__main__':
    unittest.main()
