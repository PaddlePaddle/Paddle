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

import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from check_aten_ops_signature import (
    changed_aten_ops_headers,
    discover_torch_include_dir,
    main,
    parse_paddle_header,
    parse_paddle_tensor_base,
    parse_paddle_tensor_body,
    run_check,
    tensor_base_changed,
    tensor_body_changed,
)


class TestAtenOpsSignatureCheck(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.paddle_root = self.root / "Paddle"
        self.torch_include = self.root / "torch" / "include"
        self.write_torch(
            "ATen/core/TensorBody.h",
            "namespace at { class Tensor {}; }",
        )
        (self.torch_include / "ATen/ops").mkdir(parents=True, exist_ok=True)

    def tearDown(self):
        self.tmp.cleanup()

    def write_paddle_op(self, name, content):
        path = (
            self.paddle_root / "paddle/phi/api/include/compat/ATen/ops" / name
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_paddle_tensor_body(self, content):
        path = (
            self.paddle_root
            / "paddle/phi/api/include/compat/ATen/core/TensorBody.h"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_paddle_tensor_base(self, content):
        path = (
            self.paddle_root
            / "paddle/phi/api/include/compat/ATen/core/TensorBase.h"
        )
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def write_torch(self, relpath, content):
        path = self.torch_include / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        return path

    def check(self, header):
        return run_check(
            self.paddle_root,
            self.torch_include,
            [header],
        )

    def check_changed_headers_only(self, header):
        return run_check(
            self.paddle_root,
            self.torch_include,
            [header],
            check_tensor_body=False,
            check_tensor_base=False,
        )

    def test_free_function_signature_match(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline std::vector<at::Tensor> foo(
                const at::Tensor& self,
                ::std::optional<at::Layout>,
                int64_t dim = 0) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            inline ::std::vector<Tensor> foo(
                const Tensor & input,
                std::optional<Layout> layout,
                int64_t axis=0) {
              return {};
            }
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_torch_api_free_function_declaration_match(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self, int64_t dim = 0) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            TORCH_API Tensor foo(const Tensor& input, int64_t axis = 0);
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_free_function_braced_default_value_match(self):
        header = self.write_paddle_op(
            "empty_strided.h",
            """
            namespace at {
            inline Tensor empty_strided(IntArrayRef size,
                                        IntArrayRef stride,
                                        TensorOptions options = {}) {
              return Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/empty_strided.h",
            """
            namespace at {
            inline Tensor empty_strided(IntArrayRef size,
                                        IntArrayRef stride,
                                        TensorOptions options = {}) {
              return Tensor();
            }
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_member_declaration_braced_default_value_match(self):
        header = self.write_paddle_op(
            "new_empty.h",
            """
            namespace at {
            inline Tensor Tensor::new_empty(IntArrayRef size,
                                            TensorOptions options) const {
              return Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              Tensor new_empty(IntArrayRef size,
                               TensorOptions options = {}) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              Tensor new_empty(IntArrayRef size,
                               TensorOptions options = {}) const;
            };
            }  // namespace at
            """,
        )
        declarations = parse_paddle_tensor_body(
            self.paddle_root
            / "paddle/phi/api/include/compat/ATen/core/TensorBody.h"
        )
        self.assertTrue(
            any(
                "TensorOptions={}" in sig.canonical
                for sig in declarations.values()
            )
        )
        self.assertEqual(self.check(header), [])

    def test_multiple_defaults_after_braced_default_value_match(self):
        header = self.write_paddle_op(
            "sparse_coo_tensor.h",
            """
            namespace at {
            inline Tensor sparse_coo_tensor(
                const Tensor& indices,
                const Tensor& values,
                IntArrayRef size,
                TensorOptions options = {},
                std::optional<bool> is_coalesced = std::nullopt) {
              return Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/sparse_coo_tensor.h",
            """
            namespace at {
            inline Tensor sparse_coo_tensor(
                const Tensor& indices,
                const Tensor& values,
                IntArrayRef size,
                TensorOptions options = {},
                std::optional<bool> is_coalesced = std::nullopt) {
              return Tensor();
            }
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_from_blob_overloads_with_braced_defaults_are_collected(self):
        header = self.write_paddle_op(
            "from_blob.h",
            """
            namespace at {
            inline TensorMaker for_blob(void* data, IntArrayRef sizes) noexcept {
              return TensorMaker();
            }
            inline Tensor from_blob(void* data,
                                    IntArrayRef sizes,
                                    IntArrayRef strides,
                                    const TensorOptions& options = {}) {
              return Tensor();
            }
            inline Tensor from_blob(void* data,
                                    IntArrayRef sizes,
                                    const TensorOptions& options = {}) {
              return Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/from_blob.h",
            header.read_text(),
        )
        signatures, errors = parse_paddle_header(header)
        self.assertEqual(errors, [])
        self.assertEqual(
            sum(sig.name == "from_blob" for sig in signatures),
            2,
        )
        self.assertEqual(self.check(header), [])

    def test_symint_free_function_signature_match(self):
        header = self.write_paddle_op(
            "view.h",
            """
            namespace at {
            inline at::Tensor view(const at::Tensor& self,
                                   at::IntArrayRef size) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/view.h",
            """
            namespace at {
            namespace symint {
            template <typename T,
                      typename = std::enable_if_t<std::is_same_v<T, int64_t>>>
            at::Tensor view(const at::Tensor& self, at::IntArrayRef size) {
              return self;
            }
            }  // namespace symint
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_missing_torch_overload_not_present_in_compat_is_ignored(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            inline Tensor foo(const Tensor& input) {
              return input;
            }
            inline Tensor foo(const Tensor& input, int64_t dim) {
              return input;
            }
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_free_function_does_not_match_tensor_member_function(self):
        header = self.write_paddle_op(
            "view.h",
            """
            namespace at {
            inline at::Tensor view(const at::Tensor& self,
                                   at::ScalarType dtype) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor view(at::ScalarType dtype) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor view(at::ScalarType dtype) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/view.h",
            """
            namespace at {}
            """,
        )
        errors = self.check(header)
        self.assertTrue(errors)
        self.assertTrue(
            any("signature does not match" in error.message for error in errors)
        )

    def test_inline_member_declaration_is_collected_from_tensor_body(self):
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              Tensor std(int dim) const { return Tensor(); }
            };
            }  // namespace at
            """,
        )
        declarations = parse_paddle_tensor_body(
            self.paddle_root
            / "paddle/phi/api/include/compat/ATen/core/TensorBody.h"
        )
        self.assertTrue(
            any(
                signature.canonical == "Tensor Tensor::std(int)const"
                for signature in declarations.values()
            )
        )

    def test_member_function_signature_match_from_tensor_body_declaration(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {}
            namespace at {
            inline at::Tensor Tensor::foo(int64_t dim) const {
              return at::Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim=0) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim=0) const;
            };
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_template_member_signature_match_from_tensor_body_declaration(self):
        header = self.write_paddle_op(
            "item.h",
            """
            namespace at {}
            namespace at {
            template <typename T>
            T Tensor::item() const {
              return T();
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              template <typename T>
              T item() const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              template <typename T>
              T item() const;
            };
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_free_function_default_value_mismatch_fails(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self, int64_t dim = 1) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            inline Tensor foo(const Tensor & input, int64_t axis=0) {
              return input;
            }
            }  // namespace at
            """,
        )
        errors = self.check(header)
        self.assertTrue(errors)
        self.assertTrue(
            any("signature does not match" in error.message for error in errors)
        )

    def test_free_function_string_default_value_mismatch_fails(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self,
                                  std::string mode = "sum") {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            TORCH_API Tensor foo(const Tensor& self,
                                 std::string mode = "mean");
            }  // namespace at
            """,
        )
        errors = self.check(header)
        self.assertTrue(errors)
        self.assertTrue(
            any("signature does not match" in error.message for error in errors)
        )

    def test_tensor_body_inline_member_mismatch_fails(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim = 1) const {
                return at::Tensor();
              }
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim = 0) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )
        errors = self.check(header)
        self.assertTrue(
            any(
                "TensorBody member signature" in error.message
                for error in errors
            )
        )

    def test_changed_ops_header_does_not_check_unchanged_tensor_body(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor stale_member(int64_t dim = 1) const {
                return at::Tensor();
              }
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor stale_member(int64_t dim = 0) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )

        self.assertEqual(self.check_changed_headers_only(header), [])

    def test_tensor_body_check_still_fails_when_requested(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor stale_member(int64_t dim = 1) const {
                return at::Tensor();
              }
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor stale_member(int64_t dim = 0) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/ops/foo.h",
            """
            namespace at {
            inline at::Tensor foo(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )

        errors = self.check(header)

        self.assertTrue(
            any(
                "TensorBody member signature" in error.message
                for error in errors
            )
        )

    def test_tensor_base_public_member_mismatch_fails_when_requested(self):
        self.write_paddle_tensor_base(
            """
            namespace at {
            class TensorBase {
             public:
              bool use_count() const noexcept;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBase.h",
            """
            namespace at {
            class TensorBase {
             public:
              size_t use_count() const noexcept;
            };
            }  // namespace at
            """,
        )

        errors = run_check(
            self.paddle_root,
            self.torch_include,
            [],
            check_tensor_body=False,
            check_tensor_base=True,
        )

        self.assertTrue(
            any(
                "TensorBase member signature" in error.message
                for error in errors
            )
        )

    def test_tensor_base_private_helper_is_not_checked(self):
        self.write_paddle_tensor_base(
            """
            namespace at {
            class TensorBase {
             public:
              size_t use_count() const noexcept;
             private:
              void MaybeResetHolder();
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBase.h",
            """
            namespace at {
            class TensorBase {
             public:
              size_t use_count() const noexcept;
            };
            }  // namespace at
            """,
        )

        self.assertEqual(
            run_check(
                self.paddle_root,
                self.torch_include,
                [],
                check_tensor_body=False,
                check_tensor_base=True,
            ),
            [],
        )

    def test_tensor_base_owner_is_preserved(self):
        self.write_paddle_tensor_base(
            """
            namespace at {
            class TensorBase {
             public:
              const void* const_data_ptr() const;
            };
            }  // namespace at
            """,
        )

        declarations = parse_paddle_tensor_base(
            self.paddle_root
            / "paddle/phi/api/include/compat/ATen/core/TensorBase.h"
        )

        self.assertIn(
            "const void* TensorBase::const_data_ptr()const",
            declarations,
        )

    def test_tensor_body_does_not_match_inherited_tensor_base_member(self):
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              const void* const_data_ptr() const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {};
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBase.h",
            """
            namespace at {
            class TensorBase {
             public:
              const void* const_data_ptr() const;
            };
            }  // namespace at
            """,
        )

        errors = run_check(
            self.paddle_root,
            self.torch_include,
            [],
            check_tensor_body=True,
            check_tensor_base=False,
        )

        self.assertTrue(
            any(
                "TensorBody member signature" in error.message
                for error in errors
            )
        )

    def test_tensor_base_member_still_matches_tensor_base(self):
        self.write_paddle_tensor_base(
            """
            namespace at {
            class TensorBase {
             public:
              const void* const_data_ptr() const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBase.h",
            """
            namespace at {
            class TensorBase {
             public:
              const void* const_data_ptr() const;
            };
            }  // namespace at
            """,
        )

        self.assertEqual(
            run_check(
                self.paddle_root,
                self.torch_include,
                [],
                check_tensor_body=False,
                check_tensor_base=True,
            ),
            [],
        )

    def test_deprecated_attribute_and_macro_are_equivalent(self):
        header = self.write_paddle_op("packed_accessor.h", "namespace at {}")
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              template <typename T,
                        size_t N,
                        template <typename U> class PtrTraits = DefaultPtrTraits,
                        typename index_t = int64_t>
              [[deprecated("packed_accessor is deprecated,use packed_accessor32 "
                           "or packed_accessor64 instead")]]
              GenericPackedTensorAccessor<T, N, PtrTraits, index_t>
              packed_accessor() && = delete;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              template <typename T,
                        size_t N,
                        template <typename U> class PtrTraits = DefaultPtrTraits,
                        typename index_t = int64_t>
              C10_DEPRECATED_MESSAGE(
                  "packed_accessor is deprecated,use packed_accessor32 or "
                  "packed_accessor64 instead")
              GenericPackedTensorAccessor<T, N, PtrTraits, index_t>
              packed_accessor() && = delete;
            };
            }  // namespace at
            """,
        )
        self.assertEqual(self.check(header), [])

    def test_member_default_value_mismatch_fails(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor Tensor::foo(bool upper) const {
              return at::Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(bool upper=true) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(bool upper=false) const;
            };
            }  // namespace at
            """,
        )
        errors = self.check(header)
        self.assertTrue(errors)
        self.assertTrue(
            any("signature does not match" in error.message for error in errors)
        )

    def test_member_return_type_mismatch_fails(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Scalar Tensor::foo(int64_t dim) const {
              return at::Scalar();
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Scalar foo(int64_t dim) const;
            };
            }  // namespace at
            """,
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim) const;
            };
            }  // namespace at
            """,
        )
        errors = self.check(header)
        self.assertTrue(errors)
        self.assertTrue(
            any("signature does not match" in error.message for error in errors)
        )

    def test_missing_torch_function_fails(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor bar(const at::Tensor& self) {
              return self;
            }
            }  // namespace at
            """,
        )
        self.write_torch("ATen/ops/foo.h", "namespace at {}")
        errors = self.check(header)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].candidates, ())

    def test_helper_function_in_namespace_at_is_checked(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline std::vector<at::Tensor> convert_indices_list(
                const c10::List<std::optional<at::Tensor>>& indices) {
              return {};
            }
            }  // namespace at
            """,
        )
        self.write_torch("ATen/ops/foo.h", "namespace at {}")
        errors = self.check(header)
        self.assertEqual(len(errors), 1)
        self.assertEqual(errors[0].signature.name, "convert_indices_list")

    def test_namespace_count_must_be_one_or_two(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {}
            namespace at {}
            namespace at {}
            """,
        )
        signatures, errors = parse_paddle_header(header)
        self.assertEqual(signatures, [])
        self.assertEqual(len(errors), 1)
        self.assertIn("expected one or two", errors[0])

    def test_discover_torch_include_dir_argument(self):
        self.assertEqual(
            discover_torch_include_dir(str(self.torch_include)),
            self.torch_include,
        )

    def test_discover_torch_include_dir_from_python_helper(self):
        bad = self.root / "bad"
        with mock.patch(
            "check_aten_ops_signature.torch_include_dirs_from_python",
            return_value=[bad, self.torch_include],
        ):
            self.assertEqual(
                discover_torch_include_dir(None), self.torch_include
            )

    def test_changed_aten_ops_headers_includes_modified_headers(self):
        ops_dir = self.paddle_root / "paddle/phi/api/include/compat/ATen/ops"
        ops_dir.mkdir(parents=True, exist_ok=True)
        foo = ops_dir / "foo.h"
        bar = ops_dir / "bar.h"
        foo.write_text("namespace at {}\n")
        subprocess.run(["git", "init"], cwd=self.paddle_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.paddle_root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "test"],
            cwd=self.paddle_root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.paddle_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "base"],
            cwd=self.paddle_root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "branch", "develop"], cwd=self.paddle_root, check=True
        )
        foo.write_text(
            "namespace at { inline Tensor foo() { return Tensor(); } }\n"
        )
        bar.write_text("namespace at {}\n")
        subprocess.run(
            ["git", "add", str(foo), str(bar)],
            cwd=self.paddle_root,
            check=True,
        )

        headers = changed_aten_ops_headers(self.paddle_root, "develop")

        self.assertEqual(set(headers), {foo, bar})

    def test_tensor_body_changed_detects_modified_tensor_body(self):
        tensor_body = self.write_paddle_tensor_body(
            "namespace at { class Tensor {}; }\n"
        )
        subprocess.run(["git", "init"], cwd=self.paddle_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.paddle_root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "test"],
            cwd=self.paddle_root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.paddle_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "base"],
            cwd=self.paddle_root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "branch", "develop"], cwd=self.paddle_root, check=True
        )
        tensor_body.write_text(
            "namespace at { class Tensor { public: Tensor foo() const; }; }\n"
        )

        self.assertTrue(tensor_body_changed(self.paddle_root, "develop"))

    def test_tensor_base_changed_detects_modified_tensor_base(self):
        tensor_base = self.write_paddle_tensor_base(
            "namespace at { class TensorBase {}; }\n"
        )
        subprocess.run(["git", "init"], cwd=self.paddle_root, check=True)
        subprocess.run(
            ["git", "config", "user.email", "test@example.com"],
            cwd=self.paddle_root,
            check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "test"],
            cwd=self.paddle_root,
            check=True,
        )
        subprocess.run(["git", "add", "."], cwd=self.paddle_root, check=True)
        subprocess.run(
            ["git", "commit", "-m", "base"],
            cwd=self.paddle_root,
            check=True,
            stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            ["git", "branch", "develop"], cwd=self.paddle_root, check=True
        )
        tensor_base.write_text(
            "namespace at { class TensorBase { public: size_t use_count() const; }; }\n"
        )

        self.assertTrue(tensor_base_changed(self.paddle_root, "develop"))

    def test_tensor_base_only_change_runs_tensor_base_check(self):
        with (
            mock.patch(
                "check_aten_ops_signature.changed_aten_ops_headers",
                return_value=[],
            ),
            mock.patch(
                "check_aten_ops_signature.tensor_body_changed",
                return_value=False,
            ),
            mock.patch(
                "check_aten_ops_signature.tensor_base_changed",
                return_value=True,
            ),
            mock.patch(
                "check_aten_ops_signature.discover_torch_include_dir",
                return_value=self.torch_include,
            ),
            mock.patch(
                "check_aten_ops_signature.run_check",
                return_value=[],
            ) as run_check_mock,
            mock.patch(
                "sys.argv",
                [
                    "check_aten_ops_signature.py",
                    "--paddle-root",
                    str(self.paddle_root),
                ],
            ),
        ):
            self.assertEqual(main(), 0)
        run_check_mock.assert_called_once()
        self.assertFalse(run_check_mock.call_args.kwargs["check_tensor_body"])
        self.assertTrue(run_check_mock.call_args.kwargs["check_tensor_base"])

    def test_tensor_body_change_runs_all_ops_headers(self):
        foo = self.write_paddle_op("foo.h", "namespace at {}\n")
        bar = self.write_paddle_op("bar.h", "namespace at {}\n")
        with (
            mock.patch(
                "check_aten_ops_signature.tensor_body_changed",
                return_value=True,
            ),
            mock.patch(
                "check_aten_ops_signature.tensor_base_changed",
                return_value=False,
            ),
            mock.patch(
                "check_aten_ops_signature.changed_aten_ops_headers",
                side_effect=AssertionError(
                    "TensorBody changes must scan all ops headers"
                ),
            ),
            mock.patch(
                "check_aten_ops_signature.all_aten_ops_headers",
                return_value=[foo, bar],
            ),
            mock.patch(
                "check_aten_ops_signature.discover_torch_include_dir",
                return_value=self.torch_include,
            ),
            mock.patch(
                "check_aten_ops_signature.run_check",
                return_value=[],
            ) as run_check_mock,
            mock.patch(
                "sys.argv",
                [
                    "check_aten_ops_signature.py",
                    "--paddle-root",
                    str(self.paddle_root),
                ],
            ),
        ):
            self.assertEqual(main(), 0)
        run_check_mock.assert_called_once()
        self.assertEqual(run_check_mock.call_args.args[2], [foo, bar])
        self.assertTrue(run_check_mock.call_args.kwargs["check_tensor_body"])
        self.assertFalse(run_check_mock.call_args.kwargs["check_tensor_base"])

    def test_existing_ops_member_mismatch_fails_when_scanned(self):
        header = self.write_paddle_op(
            "foo.h",
            """
            namespace at {
            inline at::Tensor Tensor::foo(bool dim) const {
              return at::Tensor();
            }
            }  // namespace at
            """,
        )
        self.write_paddle_tensor_body(
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim) const;
            };
            }  // namespace at
            """
        )
        self.write_torch(
            "ATen/core/TensorBody.h",
            """
            namespace at {
            class Tensor {
             public:
              at::Tensor foo(int64_t dim) const;
            };
            }  // namespace at
            """,
        )

        errors = run_check(
            self.paddle_root,
            self.torch_include,
            [header],
            check_tensor_body=True,
            check_tensor_base=False,
        )

        self.assertTrue(errors)
        self.assertTrue(
            any(
                "member function implementation is missing" in error.message
                for error in errors
            )
        )

    def test_no_changed_signature_inputs_skips_torch_discovery(self):
        with (
            mock.patch(
                "check_aten_ops_signature.changed_aten_ops_headers",
                return_value=[],
            ),
            mock.patch(
                "check_aten_ops_signature.tensor_body_changed",
                return_value=False,
            ),
            mock.patch(
                "check_aten_ops_signature.tensor_base_changed",
                return_value=False,
            ),
            mock.patch(
                "check_aten_ops_signature.discover_torch_include_dir",
                side_effect=AssertionError("should not discover torch"),
            ),
            mock.patch(
                "sys.argv",
                [
                    "check_aten_ops_signature.py",
                    "--paddle-root",
                    str(self.paddle_root),
                ],
            ),
        ):
            self.assertEqual(main(), 0)


if __name__ == "__main__":
    unittest.main()
