#   Copyright (c) 2020 PaddlePaddle Authors. All Rights Reserved.
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
import struct
import sys
import time

import numpy as np

__all__ = []


class ProgressBar:
    """progress bar"""

    def __init__(
        self,
        num=None,
        width=30,
        verbose=1,
        start=True,
        file=sys.stdout,
        name='step',
    ):
        self._num = num
        if isinstance(num, int) and num <= 0:
            raise TypeError('num should be None or integer (> 0)')
        max_width = self._get_max_width()
        self._width = width if width <= max_width else max_width
        self._total_width = 0
        self._verbose = verbose
        self.file = file
        self._values = {}
        self._values_order = []
        if start:
            self._start = time.time()
        self._last_update = 0
        self.name = name

        self._dynamic_display = (
            (hasattr(self.file, 'isatty') and self.file.isatty())
            or 'ipykernel' in sys.modules
            or 'posix' in sys.modules
            or 'PYCHARM_HOSTED' in os.environ
        )

    def _get_max_width(self):
        from shutil import get_terminal_size

        terminal_width, _ = get_terminal_size()
        terminal_width = terminal_width if terminal_width > 0 else 80
        max_width = min(int(terminal_width * 0.6), terminal_width - 50)
        return max_width

    def start(self):
        self.file.flush()
        self._start = time.time()

    def update(self, current_num, values={}):
        now = time.time()

        def convert_uint16_to_float(in_list):
            in_list = np.asarray(in_list)
            out = np.vectorize(
                lambda x: struct.unpack('<f', struct.pack('<I', x << 16))[0],
                otypes=[np.float32],
            )(in_list.flat)
            return np.reshape(out, in_list.shape)

        for i, (k, val) in enumerate(values):
            if k == "loss":
                if isinstance(val, list):
                    scalar_val = val[0]
                else:
                    scalar_val = val
                if isinstance(scalar_val, np.uint16):
                    values[i] = ("loss", list(convert_uint16_to_float(val)))

        if current_num:
            time_per_unit = (now - self._start) / current_num
        else:
            time_per_unit = 0

        if time_per_unit >= 1 or time_per_unit == 0:
            fps = f' - {time_per_unit:.0f}s/{self.name}'
        elif time_per_unit >= 1e-3:
            fps = f' - {time_per_unit * 1e3:.0f}ms/{self.name}'
        else:
            fps = f' - {time_per_unit * 1e6:.0f}us/{self.name}'

        info = ''
        if self._verbose == 1:
            prev_total_width = self._total_width

            if self._dynamic_display:
                sys.stdout.write('\b' * prev_total_width)
                sys.stdout.write('\r')
            else:
                sys.stdout.write('\n')

            if self._num is not None:
                numdigits = int(np.log10(self._num)) + 1

                bar_chars = (self.name + ' %' + str(numdigits) + 'd/%d [') % (
                    current_num,
                    self._num,
                )
                prog = float(current_num) / self._num
                prog_width = int(self._width * prog)

                if prog_width > 0:
                    bar_chars += '=' * (prog_width - 1)
                    if current_num < self._num:
                        bar_chars += '>'
                    else:
                        bar_chars += '='
                bar_chars += '.' * (self._width - prog_width)
                bar_chars += ']'
            else:
                bar_chars = self.name + ' %3d' % current_num

            self._total_width = len(bar_chars)
            sys.stdout.write(bar_chars)

            for k, val in values:
                info += ' - %s:' % k
                val = val if isinstance(val, list) else [val]
                for i, v in enumerate(val):
                    if isinstance(v, (float, np.float32, np.float64)):
                        if abs(v) > 1e-3:
                            info += ' %.4f' % v
                        else:
                            info += ' %.4e' % v
                    else:
                        info += ' %s' % v

            if self._num is not None and current_num < self._num:
                eta = time_per_unit * (self._num - current_num)
                if eta > 3600:
                    eta_format = '%d:%02d:%02d' % (
                        eta // 3600,
                        (eta % 3600) // 60,
                        eta % 60,
                    )
                elif eta > 60:
                    eta_format = '%d:%02d' % (eta // 60, eta % 60)
                else:
                    eta_format = '%ds' % eta

                info += ' - ETA: %s' % eta_format

            info += fps
            self._total_width += len(info)
            if prev_total_width > self._total_width:
                info += ' ' * (prev_total_width - self._total_width)

            # newline for another epoch
            if self._num is not None and current_num >= self._num:
                info += '\n'
            if self._num is None:
                info += '\n'

            sys.stdout.write(info)
            sys.stdout.flush()
            self._last_update = now
        elif self._verbose == 2 or self._verbose == 3:
            if self._num:
                numdigits = int(np.log10(self._num)) + 1
                count = (self.name + ' %' + str(numdigits) + 'd/%d') % (
                    current_num,
                    self._num,
                )
            else:
                count = self.name + ' %3d' % current_num
            info = count + info

            for k, val in values:
                info += ' - %s:' % k
                val = val if isinstance(val, list) else [val]
                for v in val:
                    if isinstance(v, (float, np.float32, np.float64)):
                        if abs(v) > 1e-3:
                            info += ' %.4f' % v
                        else:
                            info += ' %.4e' % v
                    elif (
                        isinstance(v, np.ndarray)
                        and v.size == 1
                        and v.dtype in [np.float32, np.float64]
                    ):
                        if abs(v.item()) > 1e-3:
                            info += ' %.4f' % v.item()
                        else:
                            info += ' %.4e' % v.item()
                    else:
                        info += ' %s' % v

            info += fps
            info += '\n'
            sys.stdout.write(info)
            sys.stdout.flush()
