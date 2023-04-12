# Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import os
import os.path as osp
import re
import subprocess
import warnings

import pandas as pd


def preprocess_macro(file_content, processed_file_path):
    if file_content is None:
        return file_content
    # comment out external macro
    file_content = re.sub(r'#(include|pragma)', r'// \g<0>', file_content)
    with open(processed_file_path, "w") as f:
        f.write(file_content)
    # expand macro and correct format
    subprocess.run(
        ['g++', '-E', processed_file_path, '-o', processed_file_path]
    )
    subprocess.run(['clang-format', '-i', processed_file_path])
    file_content = open(processed_file_path, "r").read()
    return file_content


def search_pattern(pattern, file_content):
    if file_content is not None:
        match_result = re.search(pattern, file_content)
        if match_result is not None:
            return match_result.group(1)
    return None


class KernelSignatureSearcher:
    kernel_sig_pattern = (
        r'(template <typename [\w\s,]+>[\s\n]*void (\w+Kernel)\([^\)]+\))'
    )
    kernel_reg_pattern = r'PD_REGISTER_(GENERAL_)?KERNEL\([\s\n]*(\w+),[\s\n]*(\w+),[\s\n]*(\w+),[\s\n]*([\w:<>]+),[^\)]*\)'
    macro_kernel_reg_pattern = (
        r'#define \w+\([^\)]*\)[\s\n\\]*PD_REGISTER_(GENERAL_)?KERNEL'
    )

    srcs_dir = ['cpu', 'gpu', 'xpu', 'onednn', 'gpudnn', 'kps']
    filter = {"kernel_name": []}

    def __init__(self, base_path, build_path):
        self.base_path = base_path
        self.build_path = build_path
        self.kernel_func_map = {}
        self.func_signature_map = {}

    @classmethod
    def search(cls, base_path, build_path):
        searcher = cls(base_path, build_path)
        searcher.search_kernel_signature()
        searcher.search_kernel_registration(base_path)
        kernel_func_df = pd.DataFrame(
            list(searcher.kernel_func_map.items()),
            columns=['kernel_name', 'kernel_func'],
        )
        func_signature_df = pd.DataFrame(
            list(searcher.func_signature_map.items()),
            columns=['kernel_func', 'kernel_signature'],
        )
        return pd.merge(
            kernel_func_df, func_signature_df, on='kernel_func', how='left'
        )

    def search_kernel_signature(self):
        for file in os.listdir(self.base_path):
            if file.endswith("_kernel.h"):
                file_content = open(osp.join(self.base_path, file), 'r').read()
                match_results = re.findall(
                    self.kernel_sig_pattern, file_content
                )
                for match_result in match_results:
                    self.func_signature_map[match_result[1]] = match_result[0]

    def search_kernel_registration(self, path):
        self.processed_file_path = osp.join(
            self.build_path, '.processed_file.cc'
        )
        for file in os.listdir(path):
            file_path = osp.join(path, file)
            if file in self.srcs_dir:
                self.search_kernel_registration(file_path)
            if osp.isdir(file_path):
                continue
            if re.match(r'\w+_kernel\.(cc|cu)', file):
                self._search_kernel_registration(file_path, file)
        if osp.exists(self.processed_file_path):
            os.remove(self.processed_file_path)

    def _search_kernel_registration(self, file_path, file):
        file_content = open(file_path, 'r').read()
        self.file_preprocessed = False
        # if some kernel registration is in macro, preprocess macro first
        if re.search(self.macro_kernel_reg_pattern, file_content):
            file_content = preprocess_macro(
                file_content, self.processed_file_path
            )
            self.file_preprocessed = True
        match_results = re.findall(self.kernel_reg_pattern, file_content)
        self.header_context = None
        for match_result in match_results:
            kernel_name = match_result[1]
            if (
                kernel_name not in self.filter["kernel_name"]
                and kernel_name not in self.kernel_func_map
            ):
                kernel_func = match_result[-1].split("<")[0].split("::")[-1]
                self.kernel_func_map[kernel_name] = kernel_func
                if kernel_func not in self.func_signature_map:
                    kernel_signature = self.search_target_kernel_signature(
                        kernel_func, file, file_content
                    )
                    self.func_signature_map[kernel_func] = kernel_signature
                    if kernel_signature is None:
                        warnings.warn(
                            "Can't find kernel signature for kernel: "
                            + kernel_func
                            + ", which is registered in file: "
                            + file_path
                            + "."
                        )

    def search_target_kernel_signature(self, kernel_func, file, file_content):
        target_kernel_signature_pattern = self.kernel_sig_pattern.replace(
            r'(\w+Kernel)', kernel_func
        )
        # search kernel signature in current kernel registration file
        kernel_signature = search_pattern(
            target_kernel_signature_pattern, file_content
        )
        if kernel_signature is not None:
            return kernel_signature
        # expand macro and search again
        if not self.file_preprocessed:
            file_content = preprocess_macro(
                file_content, self.processed_file_path
            )
            kernel_signature = search_pattern(
                target_kernel_signature_pattern, file_content
            )
            if kernel_signature is not None:
                return kernel_signature
        # expand macro in according kernel header file and search again
        if self.header_context is None:
            header_path = osp.join(self.base_path, file.split('.')[0] + '.h')
            if osp.exists(header_path):
                self.header_context = open(header_path, 'r').read()
        if self.header_context is not None:
            self.header_context = preprocess_macro(
                self.header_context, self.processed_file_path
            )
            kernel_signature = search_pattern(
                target_kernel_signature_pattern, self.header_context
            )
            if kernel_signature is not None:
                return kernel_signature
        return None


if __name__ == "__main__":
    Paddle_path = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
    build_path = osp.join(Paddle_path, 'build')
    os.makedirs(build_path, exist_ok=True)
    base_path = osp.join(Paddle_path, 'paddle/phi/kernels')

    kernel_signature_df = KernelSignatureSearcher.search(base_path, build_path)
    indenpendent_subdir = [
        'fusion',
        'legacy',
        'selected_rows',
        'sparse',
        'strings',
    ]
    for subdir in indenpendent_subdir:
        sub_path = osp.join(base_path, subdir)
        sub_df = KernelSignatureSearcher.search(sub_path, build_path)
        kernel_signature_df = pd.concat(
            [kernel_signature_df, sub_df], ignore_index=True
        )
    output_csv_path = osp.join(build_path, 'kernel_signatures.csv')
    kernel_signature_df.to_csv(output_csv_path, index=False)
    print(kernel_signature_df)
