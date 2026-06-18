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

from check_abi_compatibility import (
    DynamicSymbol,
    MissingLibrary,
    RemovedSymbol,
    check_abi_issues_approval,
    check_abi_removal_approval,
    compare_library_symbols,
    find_required_abi_approver,
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
    def test_detects_protected_compat_cxx_namespaces(self):
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
                make_symbol("_ZN2at6Tensor3dimEv", "at::Tensor::dim()")
            )
        )
        self.assertTrue(
            is_protected_paddle_abi_symbol(
                make_symbol("_ZN5torch4cuda11synchronizeEv")
            )
        )
        self.assertTrue(
            is_protected_paddle_abi_symbol(
                make_symbol(
                    "_ZN6caffe28TypeMeta12toScalarTypeEv",
                    "caffe2::TypeMeta::toScalarType()",
                )
            )
        )

    def test_ignores_non_compat_paddle_entrypoints(self):
        self.assertFalse(
            is_protected_paddle_abi_symbol(
                make_symbol(
                    "_ZN3phi12is_cpu_placeERKNS_5PlaceE",
                    "phi::is_cpu_place(phi::Place const&)",
                )
            )
        )
        self.assertFalse(
            is_protected_paddle_abi_symbol(
                make_symbol("_ZN6paddle3fooEv", "paddle::foo()")
            )
        )
        self.assertFalse(
            is_protected_paddle_abi_symbol(make_symbol("PyInit_libpaddle"))
        )
        self.assertFalse(
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

    def test_removed_non_compat_phi_symbol_does_not_fail(self):
        base_symbols = [
            make_symbol(
                "_ZN3phi12is_cpu_placeERKNS_5PlaceE",
                "phi::is_cpu_place(phi::Place const&)",
            )
        ]

        issues = compare_library_symbols(
            "paddle/libs/libphi_core.so", base_symbols, []
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


class TestAbiRemovalApproval(unittest.TestCase):
    def test_no_abi_issues_do_not_require_approval(self):
        def fetch_reviews(_pr_id, _token, _repository):
            self.fail("reviews should not be fetched without ABI issues")

        approval = check_abi_issues_approval(
            [], env={}, fetch_reviews=fetch_reviews
        )

        self.assertTrue(approval.approved)

    def test_removed_symbol_without_approval_fails(self):
        issues = [
            RemovedSymbol(
                library="paddle/libs/libphi_core.so",
                name="_ZN3c1017get_default_dtypeEv",
                demangled_name="c10::get_default_dtype()",
            )
        ]

        approval = check_abi_issues_approval(
            issues,
            env={"GIT_PR_ID": "78831", "GITHUB_API_TOKEN": "token"},
            fetch_reviews=lambda _pr_id, _token, _repository: [],
        )

        self.assertFalse(approval.approved)
        self.assertIn("no APPROVED review", approval.reason)

    def test_removed_symbol_with_required_approval_passes(self):
        for reviewer in ("SigureMo", "BingooYang"):
            with self.subTest(reviewer=reviewer):
                approval = check_abi_removal_approval(
                    env={"GIT_PR_ID": "78831", "GITHUB_API_TOKEN": "token"},
                    fetch_reviews=lambda _pr_id, _token, _repository: [
                        {"state": "APPROVED", "user": {"login": reviewer}}
                    ],
                )

                self.assertTrue(approval.approved)
                self.assertEqual(approval.reviewer, reviewer)

    def test_other_reviewer_approval_does_not_pass(self):
        reviews = [
            {"state": "APPROVED", "user": {"login": "someone-else"}},
            {"state": "COMMENTED", "user": {"login": "SigureMo"}},
        ]

        self.assertIsNone(find_required_abi_approver(reviews))

    def test_missing_review_context_fails_closed_when_abi_issue_exists(self):
        issues = [
            RemovedSymbol(
                library="paddle/libs/libphi_core.so",
                name="_ZN3c1017get_default_dtypeEv",
                demangled_name="c10::get_default_dtype()",
            )
        ]

        approval = check_abi_issues_approval(
            issues,
            env={},
            fetch_reviews=lambda _pr_id, _token, _repository: [],
        )

        self.assertFalse(approval.approved)
        self.assertIn("GIT_PR_ID or PR_ID is not set", approval.reason)

    def test_review_fetch_error_fails_closed(self):
        def fetch_reviews(_pr_id, _token, _repository):
            raise RuntimeError("GitHub API unavailable")

        approval = check_abi_removal_approval(
            env={"PR_ID": "78831", "GITHUB_TOKEN": "token"},
            fetch_reviews=fetch_reviews,
        )

        self.assertFalse(approval.approved)
        self.assertIn("GitHub API unavailable", approval.reason)


if __name__ == "__main__":
    unittest.main()
