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

import inspect
import traceback
from dataclasses import dataclass

from ..opcode_translator.instruction_utils import instrs_info
from ..utils import Singleton, log
from .executor.opcode_executor import OpcodeExecutorBase

# this file is a debug utils files for quick debug
# >>> sot.add_breakpoint(file, line)
# >>> sot.remove_breakpoint(file, line)


@dataclass
class Breakpoint:
    file: str
    line: int
    co_name: str
    offset: int

    def __hash__(self):
        return hash((self.file, self.line, self.co_name, self.offset))


class BreakpointManager(metaclass=Singleton):
    def __init__(self):
        self.breakpoints = set()
        self.executors = OpcodeExecutorBase.call_stack
        self.activate = 0
        self.record_event = []

    def clear_event(self, event):
        self.record_event.clear()

    def add_event(self, event):
        """
        event in ['All' ,'FallbackError', 'BreakGraphError', 'InnerError']
        """
        self.record_event.append(event)

    def add(self, file, line, coname=None, offset=None):
        log(1, f"add breakpoint at {file}:{line}\n")
        self.breakpoints.add(Breakpoint(file, line, coname, offset))

    def addn(self, *lines):
        """
        called inside a executor. add a list of line number in current file.
        """
        if not isinstance(lines, (list, tuple)):
            lines = [lines]
        for line in lines:
            file = self.cur_exe._code.co_filename
            self.add(file, line)

    def clear(self):
        self.breakpoints.clear()

    def hit(self, file, line, co_name, offset):
        if Breakpoint(file, line, None, None) in self.breakpoints:
            return True
        if Breakpoint(file, line, co_name, offset) in self.breakpoints:
            return True
        return False

    def locate(self, exe):
        for i, _e in enumerate(self.executors):
            if _e is exe:
                self.activate = i
                return
        raise RuntimeError("Not found executor.")

    def up(self):
        if self.activate == 0:
            return
        self.activate -= 1
        print("current function is: ", self.cur_exe._code.co_name)

    def down(self):
        if self.activate >= len(self.executors) - 1:
            return
        self.activate += 1
        print("current function is: ", self.cur_exe._code.co_name)

    def opcode(self, cur_exe=None):
        if cur_exe is None:
            cur_exe = self.cur_exe
        instr = cur_exe._instructions[cur_exe._lasti - 1]
        message = f"[Translate {cur_exe}]: (line {cur_exe._current_line:>3}) {instr.opname:<12} {instr.argval}, stack is {cur_exe._stack}\n"
        return message

    def bt(self):
        """
        display all inline calls: backtrace.
        """
        for exe in self.executors:
            lines, _ = inspect.getsourcelines(exe._code)
            print(
                "  "
                + exe._code.co_filename
                + f"({exe._current_line})"
                + f"{exe._code.co_name}()"
            )
            print(f"-> {lines[0].strip()}")
            print(f"-> {self.opcode(exe)}")
        pass

    def on_event(self, event):
        if "All" in self.record_event or event in self.record_event:
            print("event captured.")
            self.activate = len(self.executors) - 1
            breakpoint()  # noqa: T100

    def _dis_source_code(self):
        cur_exe = self.executors[self.activate]
        lines, start_line = inspect.getsourcelines(cur_exe._code)
        cur_line = cur_exe._current_line
        lines[cur_line - start_line + 1 : cur_line - start_line + 1] = (
            "  ^^^^^ HERE  \n"
        )
        print("\033[31mSource Code is: \033[0m")
        print("".join(lines))

    def dis(self, range=5):
        """
        display all instruction code and source code.
        """
        print("displaying debug info...")
        cur_exe = self.cur_exe
        print(self._dis_source_code())

        print(f"\n{cur_exe._code}")
        lasti = cur_exe._lasti
        instr_str = instrs_info(
            cur_exe._instructions, lasti - 1, range, want_str=True
        )
        print(instr_str)

    @property
    def cur_exe(self):
        exe = self.executors[self.activate]
        return exe

    def sir(self):
        """
        display sir in a page.
        """
        print("displaying sir...")
        self.cur_exe.print_sir()

    def pe(self, e):
        """
        print exception.
        """
        lines = traceback.format_tb(e.__traceback__)
        print("".join(lines))


def add_breakpoint(file, line, co_name=None, offset=None):
    BM.add(file, line, co_name, offset)


def add_event(event):
    BM.add_event(event)


BM = BreakpointManager()
