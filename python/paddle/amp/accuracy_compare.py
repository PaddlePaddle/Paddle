#   Copyright (c) 2023 PaddlePaddle Authors. All Rights Reserved.
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

import numpy as np


# Judge whether the value is within the range indicated by fp16
def is_infinite(value, dtype=np.float16):
    # return value > np.finfo(np.float16).max or value < np.finfo(np.float16).min
    array = np.array([value]).astype(dtype)
    return np.isinf(array) or np.isnan(array)


# Judge whether the value of fp32 is equal to that of fp16
def is_allclose(actual, expected, atol=1e-2, rtol=1e-2):
    return np.allclose(
        np.array([actual]), np.array([expected]), atol=atol, rtol=rtol
    )


class TensorInfo:
    def __init__(self):
        self.device = None
        self.op_type = None
        self.tensor_name = None
        self.dtype = None
        self.numel = None
        self.max_value = None
        self.min_value = None
        self.mean_value = None
        self.has_inf = None
        self.has_nan = None
        self.num_zero = None

    def __str__(self):
        return "[TensorInfo] device={}, op_type={}, tensor_name={}, dtype={}, numel={}, num_inf={}, num_nan={}, num_zero={}, max_value={:.6f}, min_value={:.6f}, mean_value={:.6f}".format(
            self.device,
            self.op_type,
            self.tensor_name,
            self.dtype,
            self.numel,
            self.has_inf,
            self.has_nan,
            self.num_zero,
            self.max_value,
            self.min_value,
            self.mean_value,
        )

    def key(
        self,
    ):
        return self.op_type + "/" + self.tensor_name

    def init_from_string(self, line):
        try:
            line_frags = line.strip().split(" ")
            for frag in line_frags:
                word_str = (
                    frag.replace("[", "").replace("]", "").replace(",", "")
                )
                words = word_str.split("=")
                if words[0] == "op":
                    self.op_type = words[1]
                elif words[0] == "device":
                    self.device = words[1]
                elif words[0] == "tensor":
                    self.tensor_name = words[1]
                elif words[0] == "dtype":
                    self.dtype = words[1]
                elif words[0] == "numel":
                    self.numel = np.int64(words[1])
                elif words[0] == "max":
                    self.max_value = np.float32(words[1])
                elif words[0] == "min":
                    self.min_value = np.float32(words[1])
                elif words[0] == "mean":
                    self.mean_value = np.float32(words[1])
                elif words[0] == "num_inf":
                    self.has_inf = int(words[1])
                elif words[0] == "num_nan":
                    self.has_nan = int(words[1])
                elif words[0] == "num_zero":
                    self.num_zero = np.int64(words[1])
        except Exception as e:
            print(f"!! Error parsing {line}")
        return self


class MixedPrecisionTensorInfo:
    def __init__(
        self, fp32_tensor_info, fp16_tensor_info, fp32_idx=0, grad_scale=1.0
    ):
        self.is_normal = True
        self.fp32_idx = fp32_idx

        self.fp32_tensor_name = None
        self.fp32_dtype = None
        self.fp32_max_value = None
        self.fp32_min_value = None
        self.fp32_mean_value = None
        self.fp32_num_zero = None
        self.scaled_fp32_max_value = None
        self.scaled_fp32_min_value = None

        self.fp16_tensor_name = None
        self.fp16_dtype = None
        self.fp16_max_value = None
        self.fp16_min_value = None
        self.fp16_mean_value = None
        self.fp16_num_zero = None
        self.fp16_has_inf = None
        self.fp16_has_nan = None

        self.fp32_div_fp16_max_value = None
        self.fp32_div_fp16_min_value = None
        self.fp32_div_fp16_mean_value = None

        if fp32_tensor_info is not None:
            self.op_type = fp32_tensor_info.op_type
            self.numel = fp32_tensor_info.numel
            self.fp32_num_zero = fp32_tensor_info.num_zero
            self.fp32_tensor_name = fp32_tensor_info.tensor_name
            self.fp32_dtype = fp32_tensor_info.dtype
            self.fp32_max_value = fp32_tensor_info.max_value
            self.fp32_min_value = fp32_tensor_info.min_value
            self.fp32_mean_value = fp32_tensor_info.mean_value
            if "GRAD" in self.fp32_tensor_name:
                self.scaled_fp32_max_value = (
                    grad_scale * fp32_tensor_info.max_value
                )
                self.scaled_fp32_min_value = (
                    grad_scale * fp32_tensor_info.min_value
                )

        if fp16_tensor_info is not None:
            self.op_type = fp16_tensor_info.op_type
            self.numel = fp16_tensor_info.numel
            self.fp16_num_zero = fp16_tensor_info.num_zero
            self.fp16_tensor_name = fp16_tensor_info.tensor_name
            self.fp16_dtype = fp16_tensor_info.dtype
            self.fp16_max_value = fp16_tensor_info.max_value
            self.fp16_min_value = fp16_tensor_info.min_value
            self.fp16_mean_value = fp16_tensor_info.mean_value
            self.fp16_has_inf = fp16_tensor_info.has_inf
            self.fp16_has_nan = fp16_tensor_info.has_nan

        if fp32_tensor_info is not None and fp16_tensor_info is not None:
            # Check whether the op name and data are equal
            assert fp32_tensor_info.op_type == fp16_tensor_info.op_type
            assert (
                fp32_tensor_info.numel == fp16_tensor_info.numel
            ), "Error:\n\tFP32 Tensor Info:{}\n\tFP16 Tensor Info:{}".format(
                fp32_tensor_info, fp16_tensor_info
            )
            # Fp16 divided by fp32
            self.fp32_div_fp16_max_value = self._div(
                self.fp16_max_value, self.fp32_max_value
            )
            self.fp32_div_fp16_min_value = self._div(
                self.fp16_min_value, self.fp32_min_value
            )
            self.fp32_div_fp16_mean_value = self._div(
                self.fp16_mean_value, self.fp32_mean_value
            )

        self._check_normal()

    def __str__(self):
        def _float_str(value):
            return f"{value:.6f}" if value is not None else value

        debug_str = "[MixedPrecisionTensorInfo] op_type={}, numel={}".format(
            self.op_type, self.numel
        )
        debug_str += "\n  FP32: tensor_name={}, dtype={}, max_value={}, min_value={}, mean_value={}".format(
            self.fp32_tensor_name,
            self.fp32_dtype,
            _float_str(self.fp32_max_value),
            _float_str(self.fp32_min_value),
            _float_str(self.fp32_mean_value),
        )
        debug_str += "\n  FP16: tensor_name={}, dtype={}, max_value={}, min_value={}, mean_value={}, has_inf={}, has_nan={}".format(
            self.fp16_tensor_name,
            self.fp16_dtype,
            _float_str(self.fp16_max_value),
            _float_str(self.fp16_min_value),
            _float_str(self.fp16_mean_value),
            self.fp16_has_inf,
            self.fp16_has_nan,
        )
        return debug_str

    def _div(self, a, b):
        if a is not None and b is not None:
            return a / b if b != 0 else 1
        return None

    def get_tensor_name(self):
        if self.fp32_tensor_name is None:
            return self.fp16_tensor_name  # + "#" + str(self.idx)
        elif self.fp16_tensor_name is None:
            return self.fp32_tensor_name + "#" + str(self.fp32_idx)
        else:
            return (
                self.fp16_tensor_name.replace(".cast_fp16", "/.cast_fp16/")
                + "#"
                + str(self.fp32_idx)
            )

    def _check_normal(self):
        # When the OP meets the following conditions, it is abnormal data, and use --skip_normal_tensors to retain the data in Excel:
        # 1. The number of OP outputs exceeds the indication range of int32
        # 2. The output data exceeds the representation range of fp16
        # 3. Nan or inf appears in fp16 output data
        # 4. The maximum value of fp32 is not equal to the maximum value of fp16
        # 5. The minimum value of fp32 is not equal to the minimum value of fp16
        if self.numel is not None and self.numel > np.iinfo(np.int32).max:
            self.is_normal = False
            return

        check_list = [
            self.fp32_max_value,
            self.fp32_min_value,
            self.scaled_fp32_max_value,
            self.scaled_fp32_min_value,
            self.fp16_max_value,
            self.fp16_min_value,
        ]

        for value in check_list:
            if value is not None and is_infinite(value):
                self.is_normal = False
                return

        if self.fp16_has_inf is not None and self.fp16_has_inf:
            self.is_normal = False
            return
        if self.fp16_has_nan is not None and self.fp16_has_nan:
            self.is_normal = False
            return

        if (
            self.scaled_fp32_max_value is not None
            and self.fp16_max_value is not None
            and not is_allclose(self.fp16_max_value, self.scaled_fp32_max_value)
        ):
            self.is_normal = False
            return
        if (
            self.scaled_fp32_min_value is not None
            and self.fp16_min_value is not None
            and not is_allclose(self.fp16_min_value, self.scaled_fp32_min_value)
        ):
            self.is_normal = False
            return


class ExcelWriter:
    def __init__(self, log_fp32_dir, log_fp16_dir, output_path):
        self.log_fp32_dir = log_fp32_dir
        self.log_fp16_dir = log_fp16_dir

        try:
            import xlsxwriter as xlw
        except ImportError:
            print(
                "import xlsxwriter failed. please run 'pip install xlsxwriter==3.0.9' to install it"
            )

        self.workbook = xlw.Workbook(output_path)
        self.title_format = self.workbook.add_format(
            {
                'bold': True,
                'border': 1,
                'font_color': 'black',
                'bg_color': '#6495ED',
                'align': 'center',
            }
        )
        self.tensor_name_format = self.workbook.add_format(
            {'bold': True, 'bg_color': '#F5F5F5'}
        )
        self.red_bg_cell_format = self.workbook.add_format(
            {'bold': True, 'bg_color': 'red'}
        )
        self.yellow_bg_cell_format = self.workbook.add_format(
            {'bold': True, 'bg_color': 'yellow'}
        )
        self.orange_bg_cell_format = self.workbook.add_format(
            {'bold': True, 'bg_color': 'orange'}
        )

    def close(self):
        self.workbook.close()
        self.workbook = None

    def _write_dtype(self, worksheet, value, row, col):
        if value is None:
            worksheet.write(row, col, "--")
        else:
            if value == "fp16":
                worksheet.write(row, col, value, self.yellow_bg_cell_format)
            else:
                worksheet.write(row, col, value)

    def _write_tensor_name(self, worksheet, mp_tensor_info, row, col):
        tensor_name = mp_tensor_info.get_tensor_name()
        if (
            mp_tensor_info.fp32_tensor_name is not None
            and mp_tensor_info.fp16_tensor_name
        ):
            worksheet.write(row, col, tensor_name, self.tensor_name_format)
        else:
            worksheet.write(row, col, tensor_name)

    def _write_maxmin_value(
        self, worksheet, value, row, col, check_finite=True
    ):
        if value is None:
            worksheet.write(row, col, "--")
        else:
            if abs(value) < 1e-5:
                value_str = f"{value:.6E}"
            else:
                value_str = f"{value:.6f}"
            if check_finite and is_infinite(value, np.float16):
                worksheet.write(row, col, value_str, self.red_bg_cell_format)
            else:
                worksheet.write(row, col, value_str)

    def _write_tensor_num_zero(
        self, worksheet, value, row, col, check_finite=True
    ):
        if value is None:
            worksheet.write(row, col, "--")
        else:
            value_str = f"{value:>10d}"
            worksheet.write(row, col, value_str)

    def _write_infinite_status(self, worksheet, value, row, col):
        if value is None:
            worksheet.write(row, col, "--")
        else:
            if value == 1:
                worksheet.write(row, col, value, self.red_bg_cell_format)
            else:
                worksheet.write(row, col, value)

    def _write_fp32divfp16_value(self, worksheet, value, row, col, loss_scale):
        def _in_range(value, scale=1):
            return value > scale * 0.95 and value < scale * 1.05

        if value is None:
            worksheet.write(row, col, "--")
        else:
            value_str = f"{value:.6f}"
            if _in_range(value, scale=1) or _in_range(value, loss_scale):
                worksheet.write(row, col, value_str)
            else:
                worksheet.write(row, col, value_str, self.orange_bg_cell_format)

    def _write_titles(self, worksheet, loss_scale, row):
        column_width_dict = {
            "op_type": 24,
            "tensor_name": 60,
            "numel": 10,
            "num_zero": 10,
            "infinite": 8,
            "dtype": 8,
            "max_value": 16,
            "min_value": 16,
            "mean_value": 16,
            "num_inf": 8,
            "num_nan": 8,
        }
        title_names = ["op_type", "tensor_name", "numel", "infinite"]
        if self.log_fp16_dir is None:
            # only fp32 values
            worksheet.merge_range("E1:H1", "fp32", self.title_format)
            worksheet.merge_range(
                "I1:J1", f"fp32 (scale={loss_scale})", self.title_format
            )
            title_names.extend(
                [
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "max_value",
                    "min_value",
                ]
            )
        elif self.log_fp32_dir is None:
            # only fp16 values
            worksheet.merge_range(
                "E1:J1", f"fp16 (scale={loss_scale})", self.title_format
            )
            title_names.extend(
                [
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "num_zero",
                    "num_inf",
                    "num_nan",
                ]
            )
        else:
            # fp32 and fp16 values
            worksheet.merge_range("E1:H1", "fp32", self.title_format)
            worksheet.merge_range(
                "I1:N1", f"fp16 (scale={loss_scale})", self.title_format
            )
            worksheet.merge_range("O1:Q1", "fp16 / fp32", self.title_format)
            title_names.extend(
                [
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "num_zero",
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "num_zero",
                    "num_inf",
                    "num_nan",
                    "max_value",
                    "min_value",
                    "mean_value",
                ]
            )

        for col in range(len(title_names)):
            col_char = chr(ord("A") + col)
            worksheet.set_column(
                col_char + ":" + col_char, column_width_dict[title_names[col]]
            )
        for col in range(len(title_names)):
            worksheet.write(row, col, title_names[col], self.title_format)

    def add_worksheet(
        self, mp_tensor_info_list, sheetname, loss_scale, skip_normal_tensors
    ):

        assert self.workbook is not None

        worksheet = self.workbook.add_worksheet(sheetname)
        row = 1

        self._write_titles(worksheet, loss_scale, row)
        row += 1

        infinite_op_types = []
        for tensor_info in mp_tensor_info_list:
            if (
                not tensor_info.is_normal
                and tensor_info.op_type not in infinite_op_types
            ):
                infinite_op_types.append(tensor_info.op_type)

            if skip_normal_tensors and tensor_info.is_normal:
                continue

            worksheet.write(row, 0, tensor_info.op_type)
            self._write_tensor_name(worksheet, tensor_info, row, 1)

            if tensor_info.numel > np.iinfo(np.int32).max:
                worksheet.write(
                    row, 2, tensor_info.numel, self.bad_value_format
                )
            else:
                worksheet.write(row, 2, tensor_info.numel)

            if tensor_info.is_normal:
                worksheet.write(row, 3, "0")
            else:
                worksheet.write(row, 3, "1", self.red_bg_cell_format)

            col = 4

            if self.log_fp32_dir is not None:
                self._write_dtype(worksheet, tensor_info.fp32_dtype, row, col)
                self._write_maxmin_value(
                    worksheet, tensor_info.fp32_max_value, row, col + 1
                )
                self._write_maxmin_value(
                    worksheet, tensor_info.fp32_min_value, row, col + 2
                )
                self._write_maxmin_value(
                    worksheet, tensor_info.fp32_mean_value, row, col + 3
                )
                self._write_tensor_num_zero(
                    worksheet, tensor_info.fp32_num_zero, row, col + 4
                )
                col += 5

                if self.log_fp16_dir is None:
                    self._write_maxmin_value(
                        worksheet, tensor_info.scaled_fp32_max_value, row, col
                    )
                    self._write_maxmin_value(
                        worksheet,
                        tensor_info.scaled_fp32_min_value,
                        row,
                        col + 1,
                    )
                    col += 2

            if self.log_fp16_dir is not None:
                self._write_dtype(worksheet, tensor_info.fp16_dtype, row, col)
                self._write_maxmin_value(
                    worksheet, tensor_info.fp16_max_value, row, col + 1
                )
                self._write_maxmin_value(
                    worksheet, tensor_info.fp16_min_value, row, col + 2
                )
                self._write_maxmin_value(
                    worksheet, tensor_info.fp16_mean_value, row, col + 3
                )
                self._write_tensor_num_zero(
                    worksheet, tensor_info.fp32_num_zero, row, col + 4
                )
                col += 5

                self._write_infinite_status(
                    worksheet, tensor_info.fp16_has_inf, row, col
                )
                self._write_infinite_status(
                    worksheet, tensor_info.fp16_has_nan, row, col + 1
                )
                col += 2

            if self.log_fp32_dir is not None and self.log_fp16_dir is not None:
                self._write_fp32divfp16_value(
                    worksheet,
                    tensor_info.fp32_div_fp16_max_value,
                    row,
                    col,
                    loss_scale,
                )
                self._write_fp32divfp16_value(
                    worksheet,
                    tensor_info.fp32_div_fp16_min_value,
                    row,
                    col + 1,
                    loss_scale,
                )
                self._write_fp32divfp16_value(
                    worksheet,
                    tensor_info.fp32_div_fp16_mean_value,
                    row,
                    col + 2,
                    loss_scale,
                )
                col += 3

            row += 1

        print(f"-- OP Types produce infinite outputs: {infinite_op_types}")


def parse_lines(lines, specified_op_list=None):
    tensor_info_list = []

    for i in range(len(lines)):
        if i % 10 == 0:
            print(
                f"-- Processing {i:-8d} / {len(lines):-8d} line",
                end="\r",
            )
        line = lines[i]
        if "[PRECISION]" in line:
            tensor_info = TensorInfo()
            tensor_info.init_from_string(line)
            if (
                tensor_info.tensor_name is not None
                and tensor_info.tensor_name != ""
            ):
                has_tensor_name = True
            if (
                specified_op_list is None
                or tensor_info.op_type in specified_op_list
            ):
                tensor_info_list.append(tensor_info)
            # print(tensor_info)
    return tensor_info_list


def parse_log(log_dir, filename, specified_op_list=None):
    if log_dir is None or filename is None:
        return None

    complete_filename = log_dir + "/" + filename
    tensor_info_list = None
    has_tensor_name = False

    try:
        with open(complete_filename, 'r') as f:
            lines = f.readlines()
            tensor_info_list = parse_lines(lines, specified_op_list)
    except FileNotFoundError:
        print("the file ", complete_filename, "is not found")
        return None, has_tensor_name
    return tensor_info_list, has_tensor_name


def merge_tensor_info_list(
    fp32_tensor_info_list, fp16_tensor_info_list, grad_scale
):
    mp_tensor_info_list = []
    if fp16_tensor_info_list is not None:
        fp32_tensor_info_dict = {}
        fp32_write_count = {}
        if fp32_tensor_info_list is not None:
            for tensor_info in fp32_tensor_info_list:
                tensor_info_key = tensor_info.key()
                count = fp32_write_count.get(tensor_info_key, 0)
                fp32_write_count[tensor_info_key] = count + 1
                fp32_tensor_info_dict[
                    tensor_info_key + "#" + str(count)
                ] = tensor_info

        fp32_read_count = {}
        for i in range(len(fp16_tensor_info_list)):
            if i % 10 == 0:
                print(
                    "-- Processing {:-8d} / {:-8d} FP16 Tensor Info".format(
                        i, len(fp16_tensor_info_list)
                    ),
                    end="\r",
                )
            fp16_tensor_info = fp16_tensor_info_list[i]
            fp32_tensor_info_key = (
                fp16_tensor_info.key()
                .replace(".cast_fp16", "")
                .replace(".cast_fp32", "")
            )
            count = fp32_read_count.get(fp32_tensor_info_key, 0)
            fp32_tensor_info = fp32_tensor_info_dict.get(
                fp32_tensor_info_key + "#" + str(count), None
            )
            if fp32_tensor_info is not None:
                fp32_read_count[fp32_tensor_info_key] = count + 1
            mp_tensor_info = MixedPrecisionTensorInfo(
                fp32_tensor_info, fp16_tensor_info, count, grad_scale
            )
            mp_tensor_info_list.append(mp_tensor_info)
            # print(mp_tensor_info)
    elif fp32_tensor_info_list is not None:
        fp32_count = {}
        for i in range(len(fp32_tensor_info_list)):
            if i % 10 == 0:
                print(
                    "-- Processing {:-8d} / {:-8d} FP32 Tensor Info".format(
                        i, len(fp32_tensor_info_list)
                    ),
                    end="\r",
                )
            tensor_info = fp32_tensor_info_list[i]
            tensor_info_key = tensor_info.key()
            count = fp32_count.get(tensor_info_key, 0)
            fp32_count[tensor_info_key] = count + 1
            mp_tensor_info = MixedPrecisionTensorInfo(
                tensor_info, None, count, grad_scale
            )
            mp_tensor_info_list.append(mp_tensor_info)

    return mp_tensor_info_list


def compare_accuracy(
    dump_path,
    another_dump_path,
    output_filename,
    loss_scale=1,
    dump_all_tensors=False,
):
    excel_writer = ExcelWriter(dump_path, another_dump_path, output_filename)
    grad_scale = loss_scale
    workerlog_filenames = []
    filenames = os.listdir(dump_path)
    for name in filenames:
        if "worker_" in name:
            workerlog_filenames.append(name)
    print(
        "-- There are {} workerlogs under {}: {}".format(
            len(workerlog_filenames), dump_path, workerlog_filenames
        )
    )

    for filename in sorted(workerlog_filenames):
        print(
            "-- [Step 1/4] Parsing FP32 logs under {}/{}".format(
                dump_path, filename
            )
        )
        fp32_tensor_info_list, fp32_has_tensor_name = parse_log(
            dump_path, filename, None
        )
        print(
            "-- [Step 2/4] Parsing FP16 logs under {}/{}".format(
                another_dump_path, filename
            )
        )
        fp16_tensor_info_list, fp16_has_tensor_name = parse_log(
            another_dump_path, filename, None
        )

        print(
            "-- [Step 3/4] Merge FP32 and FP16 tensor info for {}".format(
                filename
            )
        )
        mp_tensor_info_list = merge_tensor_info_list(
            fp32_tensor_info_list, fp16_tensor_info_list, grad_scale
        )
        print(
            "-- [Step 4/4] Add worksheet for mixed precision tensor info of {}".format(
                filename
            )
        )
        excel_writer.add_worksheet(
            mp_tensor_info_list,
            filename,
            loss_scale,
            False,
        )

    print(f"-- Write to {output_filename}")

    print("")
    excel_writer.close()
