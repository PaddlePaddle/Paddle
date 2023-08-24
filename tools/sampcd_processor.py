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
"""
please make sure to run in the tools path
usage: python sampcd_processor.py --mode {cpu or gpu}
    {cpu or gpu}: running in cpu version or gpu version

for example, you can run cpu version testing like this:

    python sampcd_processor.py --mode cpu

"""

import functools
import logging
import multiprocessing
import os
import platform
import queue
import re
import sys
import threading
import time
import typing

import xdoctest
from sampcd_processor_utils import (
    TEST_TIMEOUT,
    DocTester,
    TestResult,
    logger,
    parse_args,
    run_doctest,
)

XDOCTEST_CONFIG = {
    "global_exec": r"\n".join(
        [
            "import paddle",
            "paddle.device.set_device('cpu')",
            "paddle.set_default_dtype('float32')",
            "paddle.disable_static()",
        ]
    ),
    "default_runtime_state": {"IGNORE_WHITESPACE": True},
}


def _patch_global_state(debug, verbose):
    # patch xdoctest global_state
    from xdoctest import global_state

    _debug_xdoctest = debug and verbose > 2
    global_state.DEBUG = _debug_xdoctest
    global_state.DEBUG_PARSER = global_state.DEBUG_PARSER and _debug_xdoctest
    global_state.DEBUG_CORE = global_state.DEBUG_CORE and _debug_xdoctest
    global_state.DEBUG_RUNNER = global_state.DEBUG_RUNNER and _debug_xdoctest
    global_state.DEBUG_DOCTEST = global_state.DEBUG_DOCTEST and _debug_xdoctest


def _patch_tensor_place():
    from xdoctest import checker

    pattern_tensor = re.compile(
        r"""
        (Tensor\(.*?place=)     # Tensor start
        (.*?)                   # Place=(XXX)
        (\,.*?\))
        """,
        re.X | re.S,
    )

    _check_output = checker.check_output

    def check_output(got, want, runstate=None):
        if not want:  # nocover
            return True

        return _check_output(
            got=pattern_tensor.sub(r'\1Place(cpu)\3', got),
            want=pattern_tensor.sub(r'\1Place(cpu)\3', want),
            runstate=runstate,
        )

    checker.check_output = check_output


def _patch_float_precision(digits):
    from xdoctest import checker

    pattern_number = re.compile(
        r"""
        (?:
            (?<=[\s*\[\(\'\"\:])                        # number starts
            (?:                                         # int/float or complex-real
                (?:
                    [+-]?
                    (?:
                        (?: \d*\.\d+) | (?: \d+\.?)     # int/float
                    )
                )
                (?:[Ee][+-]?\d+)?
            )
            (?:                                         # complex-imag
                (?:
                    (?:
                        [+-]?
                        (?:
                            (?: \d*\.\d+) | (?: \d+\.?)
                        )
                    )
                    (?:[Ee][+-]?\d+)?
                )
            (?:[Jj])
            )?
        )
        """,
        re.X | re.S,
    )

    _check_output = checker.check_output

    def _sub_number(match_obj, digits):
        match_str = match_obj.group()

        if 'j' in match_str or 'J' in match_str:
            try:
                match_num = complex(match_str)
            except ValueError:
                return match_str

            return (
                str(
                    complex(
                        round(match_num.real, digits),
                        round(match_num.imag, digits),
                    )
                )
                .strip('(')
                .strip(')')
            )
        else:
            try:
                return str(round(float(match_str), digits))
            except ValueError:
                return match_str

    sub_number = functools.partial(_sub_number, digits=digits)

    def check_output(got, want, runstate=None):
        if not want:  # nocover
            return True

        return _check_output(
            got=pattern_number.sub(sub_number, got),
            want=pattern_number.sub(sub_number, want),
            runstate=runstate,
        )

    checker.check_output = check_output


class Directive:
    """Base class of global direvtives just for `xdoctest`."""

    pattern: typing.Pattern

    def parse_directive(self, docstring: str) -> typing.Tuple[str, typing.Any]:
        pass


class TimeoutDirective(Directive):
    pattern = re.compile(
        r"""
        (?:
            (?:
                \s*\>{3}\s*\#\s*x?doctest\:\s*
            )
            (?P<op>[\+\-])
            (?:
                TIMEOUT
            )
            \(
                (?P<time>\d+)
            \)
            (?:
                \s*?
            )
        )
        """,
        re.X | re.S,
    )

    def __init__(self, timeout):
        self._timeout = timeout

    def parse_directive(self, docstring):
        match_obj = self.pattern.search(docstring)
        if match_obj is not None:
            op_time = match_obj.group('time')
            match_start = match_obj.start()
            match_end = match_obj.end()

            return (
                (docstring[:match_start] + '\n' + docstring[match_end:]),
                float(op_time),
            )

        return docstring, float(self._timeout)


class Xdoctester(DocTester):
    """A Xdoctest doctester."""

    directives: typing.Dict[str, typing.Tuple[typing.Type[Directive], ...]] = {
        'timeout': (TimeoutDirective, TEST_TIMEOUT)
    }

    def __init__(
        self,
        debug=False,
        style='freeform',
        target='codeblock',
        mode='native',
        verbose=2,
        patch_global_state=True,
        patch_tensor_place=True,
        patch_float_precision=5,
        use_multiprocessing=True,
        **config,
    ):
        self.debug = debug

        self.style = style
        self.target = target
        self.mode = mode
        self.verbose = verbose
        self.config = {**XDOCTEST_CONFIG, **(config or {})}

        self._patch_global_state = patch_global_state
        self._patch_tensor_place = patch_tensor_place
        self._patch_float_precision = patch_float_precision
        self._use_multiprocessing = use_multiprocessing

        # patch xdoctest before `xdoctest.core.parse_docstr_examples`
        self._patch_xdoctest()

        self.docstring_parser = functools.partial(
            xdoctest.core.parse_docstr_examples, style=self.style
        )

        self.directive_pattern = re.compile(
            r"""
            (?<=(\#\s))     # positive lookbehind, directive begins
            (doctest)       # directive prefix, which should be replaced
            (?=(:\s*.*\n))  # positive lookahead, directive content
            """,
            re.X,
        )

        self.directive_prefix = 'xdoctest'

    def _patch_xdoctest(self):
        if self._patch_global_state:
            _patch_global_state(self.debug, self.verbose)

        if self._patch_tensor_place:
            _patch_tensor_place()

        if self._patch_float_precision is not None:
            _patch_float_precision(self._patch_float_precision)

    def _parse_directive(
        self, docstring: str
    ) -> typing.Tuple[str, typing.Dict[str, Directive]]:
        directives = {}
        for name, directive_cls in self.directives.items():
            docstring, direct = directive_cls[0](
                *directive_cls[1:]
            ).parse_directive(docstring)
            directives[name] = direct

        return docstring, directives

    def convert_directive(self, docstring: str) -> str:
        """Replace directive prefix with xdoctest"""
        return self.directive_pattern.sub(self.directive_prefix, docstring)

    def prepare(self, test_capacity: set):
        """Set environs for xdoctest directive.
        The keys in environs, which also used in `# xdoctest: +REQUIRES(env:XX)`, should be UPPER case.

        If `test_capacity = {"cpu"}`, then we set:

            - `os.environ["CPU"] = "True"`

        which makes this SKIPPED:

            - # xdoctest: +REQUIRES(env:GPU)

        If `test_capacity = {"cpu", "gpu"}`, then we set:

            - `os.environ["CPU"] = "True"`
            - `os.environ["GPU"] = "True"`

        which makes this SUCCESS:

            - # xdoctest: +REQUIRES(env:GPU)
        """
        logger.info("Set xdoctest environ ...")
        for capacity in test_capacity:
            key = capacity.upper()
            os.environ[key] = "True"
            logger.info("Environ: %s , set to True.", key)

        logger.info("API check using Xdoctest prepared!-- Example Code")
        logger.info("running under python %s", platform.python_version())
        logger.info("running under xdoctest %s", xdoctest.__version__)

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Run the xdoctest with a docstring."""
        # parse global directive
        docstring, directives = self._parse_directive(docstring)

        # extract xdoctest examples
        examples_to_test, examples_nocode = self._extract_examples(
            api_name, docstring, **directives
        )

        # run xdoctest
        try:
            result = self._execute_xdoctest(
                examples_to_test, examples_nocode, **directives
            )
        except queue.Empty:
            result = [
                TestResult(
                    name=api_name,
                    timeout=True,
                    time=directives.get('timeout', TEST_TIMEOUT),
                )
            ]

        return result

    def _extract_examples(self, api_name, docstring, **directives):
        """Extract code block examples from docstring."""
        examples_to_test = {}
        examples_nocode = {}
        for example_idx, example in enumerate(
            self.docstring_parser(docstr=docstring, callname=api_name)
        ):
            example.mode = self.mode
            example.config.update(self.config)
            example_key = f"{api_name}_{example_idx}"

            # check whether there are some parts parsed by xdoctest
            if not example._parts:
                examples_nocode[example_key] = example
                continue

            examples_to_test[example_key] = example

        if not examples_nocode and not examples_to_test:
            examples_nocode[api_name] = api_name

        return examples_to_test, examples_nocode

    def _execute_xdoctest(
        self, examples_to_test, examples_nocode, **directives
    ):
        if self._use_multiprocessing:
            _ctx = multiprocessing.get_context('spawn')
            result_queue = _ctx.Queue()
            exec_processer = functools.partial(_ctx.Process, daemon=True)
        else:
            result_queue = queue.Queue()
            exec_processer = functools.partial(threading.Thread, daemon=True)

        processer = exec_processer(
            target=self._execute_with_queue,
            args=(
                result_queue,
                examples_to_test,
                examples_nocode,
            ),
        )

        processer.start()
        result = result_queue.get(
            timeout=directives.get('timeout', TEST_TIMEOUT)
        )
        processer.join()

        return result

    def _execute(self, examples_to_test, examples_nocode):
        """Run xdoctest for each example"""
        # patch xdoctest first in each process
        self._patch_xdoctest()

        # run the xdoctest
        test_results = []
        for _, example in examples_to_test.items():
            start_time = time.time()
            result = example.run(verbose=self.verbose, on_error='return')
            end_time = time.time()

            test_results.append(
                TestResult(
                    name=str(example),
                    passed=result['passed'],
                    skipped=result['skipped'],
                    failed=result['failed'],
                    test_msg=str(result['exc_info']),
                    time=end_time - start_time,
                )
            )

        for _, example in examples_nocode.items():
            test_results.append(TestResult(name=str(example), nocode=True))

        return test_results

    def _execute_with_queue(self, queue, examples_to_test, examples_nocode):
        queue.put(self._execute(examples_to_test, examples_nocode))

    def print_summary(self, test_results, whl_error=None):
        summary_success = []
        summary_failed = []
        summary_skiptest = []
        summary_timeout = []
        summary_nocodes = []

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(stdout_handler)
        logger.info("----------------End of the Check--------------------")
        if whl_error is not None and whl_error:
            logger.info("%s is not in whl.", whl_error)
            logger.info("")
            logger.info("Please check the whl package and API_PR.spec!")
            logger.info(
                "You can follow these steps in order to generate API.spec:"
            )
            logger.info("1. cd ${paddle_path}, compile paddle;")
            logger.info("2. pip install build/python/dist/(build whl package);")
            logger.info(
                "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
            )
            for test_result in test_results:
                if test_result.failed:
                    logger.info(
                        "In addition, mistakes found in sample codes: %s",
                        test_result.name,
                    )
            logger.info("----------------------------------------------------")
            sys.exit(1)
        else:
            for test_result in test_results:
                if not test_result.nocode:
                    if test_result.passed:
                        summary_success.append(test_result.name)

                    if test_result.skipped:
                        summary_skiptest.append(test_result.name)

                    if test_result.failed:
                        summary_failed.append(test_result.name)

                    if test_result.timeout:
                        summary_timeout.append(
                            {
                                'api_name': test_result.name,
                                'run_time': test_result.time,
                            }
                        )
                else:
                    summary_nocodes.append(test_result.name)

            if len(summary_success):
                logger.info("%d sample codes ran success", len(summary_success))
                logger.info('\n'.join(summary_success))

            if len(summary_skiptest):
                logger.info("%d sample codes skipped", len(summary_skiptest))
                logger.info('\n'.join(summary_skiptest))

            if len(summary_nocodes):
                logger.info(
                    "%d apis could not run test or don't have sample codes",
                    len(summary_nocodes),
                )
                logger.info('\n'.join(summary_nocodes))

            if len(summary_timeout):
                logger.info("%d sample codes ran timeout", len(summary_timeout))
                for _result in summary_timeout:
                    logger.info(
                        f"{_result['api_name']} - more than {_result['run_time']}s"
                    )

            if len(summary_failed):
                logger.info("%d sample codes ran failed", len(summary_failed))
                logger.info('\n'.join(summary_failed))

            if summary_failed or summary_timeout or summary_nocodes:
                logger.info(
                    "Mistakes found in sample codes. Please recheck the sample codes."
                )
                sys.exit(1)

        logger.info("Sample code check is successful!")


if __name__ == '__main__':
    args = parse_args()
    run_doctest(args, doctester=Xdoctester(debug=args.debug))
