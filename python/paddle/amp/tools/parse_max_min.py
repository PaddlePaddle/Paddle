#   Copyright (c) 2022 PaddlePaddle Authors. All Rights Reserved.
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


import argparse
import os

import numpy as np
import xlsxwriter as xlw


def is_infinite(value, dtype=np.float16):
    # return value > np.finfo(np.float16).max or value < np.finfo(np.float16).min
    array = np.array([value]).astype(dtype)
    return np.isinf(array) or np.isnan(array)


def is_allclose(actual, expected, atol=1e-2, rtol=1e-2):
    return np.allclose(
        np.array([actual]), np.array([expected]), atol=atol, rtol=rtol
    )


class TensorInfo(object):
    def __init__(self):
        self.op_type = None
        self.tensor_name = None
        self.dtype = None
        self.numel = None
        self.max_value = None
        self.min_value = None
        self.mean_value = None
        self.has_inf = None
        self.has_nan = None

    def __str__(self):
        return "[TensorInfo] op_type={}, tensor_name={}, dtype={}, numel={}, has_inf={}, has_nan={}, max_value={:.6f}, min_value={:.6f}, mean_value={:.6f}".format(
            self.op_type,
            self.tensor_name,
            self.dtype,
            self.numel,
            self.has_inf,
            self.has_nan,
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
                elif words[0] == "find_inf":
                    self.has_inf = int(words[1])
                elif words[0] == "find_nan":
                    self.has_nan = int(words[1])
        except Exception as e:
            print("!! Error parsing {}".format(line))
        return self


class MixedPrecisionTensorInfo(object):
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
        self.scaled_fp32_max_value = None
        self.scaled_fp32_min_value = None

        self.fp16_tensor_name = None
        self.fp16_dtype = None
        self.fp16_max_value = None
        self.fp16_min_value = None
        self.fp16_mean_value = None
        self.fp16_has_inf = None
        self.fp16_has_nan = None

        self.fp32_div_fp16_max_value = None
        self.fp32_div_fp16_min_value = None
        self.fp32_div_fp16_mean_value = None

        if fp32_tensor_info is not None:
            self.op_type = fp32_tensor_info.op_type
            self.numel = fp32_tensor_info.numel
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
            self.fp16_tensor_name = fp16_tensor_info.tensor_name
            self.fp16_dtype = fp16_tensor_info.dtype
            self.fp16_max_value = fp16_tensor_info.max_value
            self.fp16_min_value = fp16_tensor_info.min_value
            self.fp16_mean_value = fp16_tensor_info.mean_value
            self.fp16_has_inf = fp16_tensor_info.has_inf
            self.fp16_has_nan = fp16_tensor_info.has_nan

        if fp32_tensor_info is not None and fp16_tensor_info is not None:
            assert fp32_tensor_info.op_type == fp16_tensor_info.op_type
            assert (
                fp32_tensor_info.numel == fp16_tensor_info.numel
            ), "Error:\n\tFP32 Tensor Info:{}\n\tFP16 Tensor Info:{}".format(
                fp32_tensor_info, fp16_tensor_info
            )

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
            return "{:.6f}".format(value) if value is not None else value

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

        # if self.scaled_fp32_max_value is not None and self.fp16_max_value is not None and not is_allclose(self.fp16_max_value, self.scaled_fp32_max_value):
        #    self.is_normal = False
        #    return
        # if self.scaled_fp32_min_value is not None and self.fp16_min_value is not None and not is_allclose(self.fp16_min_value, self.scaled_fp32_min_value):
        #    self.is_normal = False
        #    return


class ExcelWriter(object):
    def __init__(self, log_fp32_dir, log_fp16_dir, output_path):
        self.log_fp32_dir = log_fp32_dir
        self.log_fp16_dir = log_fp16_dir

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
            if value == "float16":
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
                value_str = "{:.6E}".format(value)
            else:
                value_str = "{:.6f}".format(value)
            if check_finite and is_infinite(value, np.float16):
                worksheet.write(row, col, value_str, self.red_bg_cell_format)
            else:
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
            value_str = "{:.6f}".format(value)
            if _in_range(value, scale=1) or _in_range(value, loss_scale):
                worksheet.write(row, col, value_str)
            else:
                worksheet.write(row, col, value_str, self.orange_bg_cell_format)

    def _write_titles(self, worksheet, loss_scale, row):
        column_width_dict = {
            "op_type": 24,
            "tensor_name": 60,
            "numel": 10,
            "infinite": 8,
            "dtype": 8,
            "max_value": 16,
            "min_value": 16,
            "mean_value": 16,
            "has_inf": 8,
            "has_nan": 8,
        }
        title_names = ["op_type", "tensor_name", "numel", "infinite"]
        if self.log_fp16_dir is None:
            # only fp32 values
            worksheet.merge_range("E1:H1", "fp32", self.title_format)
            worksheet.merge_range(
                "I1:J1", "fp32 (scale={})".format(loss_scale), self.title_format
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
                "E1:J1", "fp16 (scale={})".format(loss_scale), self.title_format
            )
            title_names.extend(
                [
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "has_inf",
                    "has_nan",
                ]
            )
        else:
            # fp32 and fp16 values
            worksheet.merge_range("E1:H1", "fp32", self.title_format)
            worksheet.merge_range(
                "I1:N1", "fp16 (scale={})".format(loss_scale), self.title_format
            )
            worksheet.merge_range("O1:Q1", "fp16 / fp32", self.title_format)
            title_names.extend(
                [
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "dtype",
                    "max_value",
                    "min_value",
                    "mean_value",
                    "has_inf",
                    "has_nan",
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
                col += 4

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
                col += 4

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

        print(
            "-- OP Types produce infinite outputs: {}".format(infinite_op_types)
        )


def parse_log(log_dir, filename, specified_op_list=None):
    if log_dir is None or filename is None:
        return None

    complete_filename = log_dir + "/" + filename
    tensor_info_list = []
    has_tensor_name = False

    with open(complete_filename, 'r') as f:
        lines = f.readlines()
        for i in range(len(lines)):
            if i % 10 == 0:
                print(
                    "-- Processing {:-8d} / {:-8d} line".format(i, len(lines)),
                    end="\r",
                )
            # [op=adamw] [tensor=encoder_layer_20_multi_head_att_output_fc_0.w_0], numel: 294912, max: 0.005773, min: -0.005774
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

            # for key, value in fp32_count.items():
            #    print("{} : {}".format(key, value))

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


def main(args, log_fp32_dir, log_fp16_dir, output_path):
    excel_writer = ExcelWriter(log_fp32_dir, log_fp16_dir, output_path)

    specified_op_list = None
    if args.specified_op_list is not None and args.specified_op_list != "None":
        specified_op_list = args.specified_op_list.split(",")
        print("-- Speficified op list: ", specified_op_list)

    grad_scale = args.loss_scale

    workerlog_filenames = []
    if args.num_workerlogs is None:
        filenames = os.listdir(log_fp32_dir)
        for name in filenames:
            if "workerlog." in name:
                workerlog_filenames.append(name)
    else:
        for i in range(args.num_workerlogs):
            workerlog_filenames.append("workerlog." + str(i))

    print(
        "-- There are {} workerlogs under {}: {}".format(
            len(workerlog_filenames), log_fp32_dir, workerlog_filenames
        )
    )

    for filename in sorted(workerlog_filenames):
        print(
            "-- [Step 1/4] Parsing FP32 logs under {}/{}".format(
                log_fp32_dir, filename
            )
        )
        fp32_tensor_info_list, fp32_has_tensor_name = parse_log(
            log_fp32_dir, filename, specified_op_list
        )
        print(
            "-- [Step 2/4] Parsing FP16 logs under {}/{}".format(
                log_fp16_dir, filename
            )
        )
        fp16_tensor_info_list, fp16_has_tensor_name = parse_log(
            log_fp16_dir, filename, specified_op_list
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
            args.loss_scale,
            args.skip_normal_tensors,
        )

    print("-- Write to {}".format(output_path))
    print("")
    excel_writer.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--loss_scale',
        type=float,
        default=524288,
        help='The init loss scale used in amp training',
    )
    parser.add_argument(
        '--grad_merge_method',
        type=str,
        default="divsum",
        help='[divsum|sumdiv]',
    )
    parser.add_argument('--accumulate_steps', type=int, default=1)
    parser.add_argument('--num_workerlogs', type=int, default=None)
    parser.add_argument('--num_subdirs', type=int, default=None)
    parser.add_argument(
        "--skip_normal_tensors", action='store_true', default=False
    )
    parser.add_argument(
        '--specified_op_list',
        type=str,
        default=None,
        help='Specify the operator list.',
    )
    args = parser.parse_args()

    if args.grad_merge_method not in ["divsum", "sumdiv"]:
        raise ValueError(
            "grad_merge_method is expected to be \"divsum\" or \"sumdiv\", but recieved \"{}\"".format(
                args.grad_merge_method
            )
        )

    # bsz3000_divsum
    grad_merge_config = (
        "bsz" + str(args.accumulate_steps) + "_" + args.grad_merge_method
    )
    loss_scale_str = str(int(args.loss_scale))
    if args.num_subdirs is not None and args.num_subdirs > 0:
        dirname = "/root/paddlejob/workspace/env_run/dingsiyu/prompt_ernie/ernie_3.0_100b_no_distill"
        # log_lyq_bsz3000_divsum-pp5mp8_0531-fp32/
        parallel_method = "pp5mp8"
        for subdir_id in range(args.num_subdirs):
            log_fp32_dir = "{}/log_lyq_debug_{}-{}_0531-fp32/log_{}".format(
                dirname, grad_merge_config, parallel_method, subdir_id
            )
            log_fp16_dir = (
                "{}/log_lyq_debug_{}-{}_0531-fp16_{}_black_mm_2/log_{}".format(
                    dirname,
                    grad_merge_config,
                    parallel_method,
                    loss_scale_str,
                    subdir_id,
                )
            )
            output_path = (
                "ernie3.0_{}-{}-{}_black_mm_2-0531_node_{}.xlsx".format(
                    grad_merge_config,
                    parallel_method,
                    loss_scale_str,
                    subdir_id,
                )
            )
            main(args, log_fp32_dir, log_fp16_dir, output_path)
    else:
        # dirname = "/root/paddlejob/workspace/work/liuyiqun/ernie_3.0_amp_precision/ernie_3.0_100b_no_distill_single_card"
        # dirname = "/root/paddlejob/workspace/env_run/dingsiyu/prompt_ernie/ernie_3.0_100b_no_distill_single_card"
        # log_debug_bsz32_divsum-pp2mp2_0531-fp32/
        # log_debug_bsz32_divsum-pp2mp2_0531-fp16_524288/
        # parallel_method = "pp2mp2"
        # log_fp32_dir = "{}/log_debug_{}-{}_0531-fp32".format(dirname, grad_merge_config, parallel_method)
        # log_fp16_dir = "{}/log_debug_{}-{}_0531-fp16_{}".format(dirname, grad_merge_config, parallel_method, loss_scale_str)
        # output_path = "ernie3.0_{}-{}-{}-0531_single.xlsx".format(grad_merge_config, parallel_method, loss_scale_str)
        # main(args, log_fp32_dir, log_fp16_dir, output_path)
        dirname = "/root/paddlejob/workspace/work/liuyiqun/PaddleDetection"
        log_fp16_dir_1 = "log_maskrcnn_algo1"
        log_fp16_dir_2 = "log_maskrcnn_algo01"
        output_path = "maskrcnn.xlsx"
        main(args, log_fp16_dir_1, log_fp16_dir_2, output_path)
