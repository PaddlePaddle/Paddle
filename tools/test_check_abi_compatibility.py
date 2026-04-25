#!/usr/bin/env python

# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
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

import unittest

try:
    from check_abi_compatibility import (
        DynamicSymbol,
        MissingLibrary,
        RemovedSymbol,
        compare_library_symbols,
        is_protected_paddle_abi_symbol,
        parse_readelf_dynamic_symbols,
    )
except ModuleNotFoundError:
    from tools.check_abi_compatibility import (
        DynamicSymbol,
        MissingLibrary,
        RemovedSymbol,
        compare_library_symbols,
        is_protected_paddle_abi_symbol,
        parse_readelf_dynamic_symbols,
    )


def make_symbol(name, demangled_name=None, bind="GLOBAL", section="12"):
    return DynamicSymbol(
        name=name,
        symbol_type="FUNC",
        bind=bind,
        section=section,
        demangled_name=demangled_name or name,
    )


class TestParseReadelfDynamicSymbols(unittest.TestCase):
    def test_ignores_weak_undefined_and_local_symbols(self):
        readelf_output = """
Symbol table '.dynsym' contains 5 entries:
   Num:    Value          Size Type    Bind   Vis      Ndx Name
     1: 0000000000001000    42 FUNC    GLOBAL DEFAULT   12 _ZN3c1017get_default_dtypeEv
     2: 0000000000001010    42 FUNC    WEAK   DEFAULT   12 _ZN3c104weakEv
     3: 0000000000000000     0 FUNC    GLOBAL DEFAULT  UND _ZN3c107missingEv
     4: 0000000000001020    42 FUNC    LOCAL  DEFAULT   12 _ZN3c105localEv
     5: 0000000000001030     8 OBJECT  GLOBAL DEFAULT   13 _ZN3phi3barE
"""
        symbols = parse_readelf_dynamic_symbols(readelf_output)
        self.assertEqual(
            [symbol.name for symbol in symbols],
            ["_ZN3c1017get_default_dtypeEv", "_ZN3phi3barE"],
        )


class TestProtectedSymbols(unittest.TestCase):
    def test_detects_protected_cxx_namespaces(self):
        self.assertTrue(
            is_protected_paddle_abi_symbol(
                make_symbol(
                    "_ZN3c1017get_default_dtypeEv",
                    "c10::get_default_dtype()",
                )
            )
        )
        self.assertTrue(
            is_protected_paddle_abi_symbol(
                make_symbol("_ZN3phi3barEv", "phi::bar()")
            )
        )
        self.assertTrue(
            is_protected_paddle_abi_symbol(
                make_symbol("_ZN5torch4cuda11synchronizeEv")
            )
        )

    def test_detects_relevant_c_and_python_entrypoints(self):
        self.assertTrue(
            is_protected_paddle_abi_symbol(make_symbol("PyInit_libpaddle"))
        )
        self.assertTrue(
            is_protected_paddle_abi_symbol(make_symbol("PD_ConfigCreate"))
        )

    def test_ignores_third_party_symbols(self):
        self.assertFalse(
            is_protected_paddle_abi_symbol(make_symbol("XXH32", "XXH32"))
        )
        self.assertFalse(
            is_protected_paddle_abi_symbol(
                make_symbol("_ZN4YAML7EmitterC1Ev", "YAML::Emitter::Emitter()")
            )
        )


class TestCompareLibrarySymbols(unittest.TestCase):
    def test_added_symbols_do_not_fail(self):
        base_symbols = [
            make_symbol(
                "_ZN3c1017get_default_dtypeEv", "c10::get_default_dtype()"
            )
        ]
        pr_symbols = [
            *base_symbols,
            make_symbol(
                "_ZN3c1017set_default_dtypeEv", "c10::set_default_dtype()"
            ),
        ]

        issues = compare_library_symbols(
            "paddle/libs/libphi_core.so", base_symbols, pr_symbols
        )

        self.assertEqual(issues, [])

    def test_removed_protected_symbol_fails(self):
        base_symbols = [
            make_symbol(
                "_ZN3c1017get_default_dtypeEv", "c10::get_default_dtype()"
            )
        ]

        issues = compare_library_symbols(
            "paddle/libs/libphi_core.so", base_symbols, []
        )

        self.assertEqual(
            issues,
            [
                RemovedSymbol(
                    library="paddle/libs/libphi_core.so",
                    name="_ZN3c1017get_default_dtypeEv",
                    demangled_name="c10::get_default_dtype()",
                )
            ],
        )

    def test_removed_third_party_symbol_does_not_fail(self):
        base_symbols = [make_symbol("XXH32", "XXH32")]

        issues = compare_library_symbols(
            "paddle/base/libpaddle.so", base_symbols, []
        )

        self.assertEqual(issues, [])

    def test_missing_pr_library_fails_when_base_has_library(self):
        base_symbols = [
            make_symbol(
                "_ZN3c1017get_default_dtypeEv", "c10::get_default_dtype()"
            )
        ]

        issues = compare_library_symbols(
            "paddle/libs/libphi_core.so", base_symbols, None
        )

        self.assertEqual(
            issues, [MissingLibrary(library="paddle/libs/libphi_core.so")]
        )

    def test_missing_base_library_does_not_fail(self):
        pr_symbols = [
            make_symbol(
                "_ZN3c1017get_default_dtypeEv", "c10::get_default_dtype()"
            )
        ]

        issues = compare_library_symbols(
            "paddle/libs/libphi_core.so", None, pr_symbols
        )

        self.assertEqual(issues, [])


if __name__ == "__main__":
    unittest.main()
