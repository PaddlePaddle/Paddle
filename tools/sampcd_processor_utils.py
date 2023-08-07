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

import argparse
import collections
import inspect
import logging
import os
import re
import subprocess
import sys
import time
import typing

logger = logging.getLogger()
if logger.handlers:
    console = logger.handlers[
        0
    ]  # we assume the first handler is the one we want to configure
else:
    console = logging.StreamHandler(stream=sys.stderr)
    logger.addHandler(console)
console.setFormatter(logging.Formatter("%(message)s"))


RUN_ON_DEVICE = 'cpu'
ENV_KEY_TEST_CAPACITY = 'SAMPLE_CODE_TEST_CAPACITY'
API_DEV_SPEC_FN = 'paddle/fluid/API_DEV.spec'
API_PR_SPEC_FN = 'paddle/fluid/API_PR.spec'
API_DIFF_SPEC_FN = 'dev_pr_diff_api.spec'
TEST_TIMEOUT = 10


TestResult = collections.namedtuple(
    "TestResult",
    (
        "name",
        "nocode",
        "passed",
        "skipped",
        "failed",
        "time",
        "test_msg",
        "extra_info",
    ),
    defaults=(None, False, False, False, False, -1, "", None),
)


class DocTester:
    """A DocTester can be used to test the codeblock from the API's docstring.

    Attributes:

        style(str): `style` should be in {'google', 'freeform'}.
            `google`, codeblock in `Example(s):` section of docstring.
            `freeform`, all codeblocks in docstring wrapped with PS1(>>> ) and PS2(... ).
            **CAUTION** no matter `.. code-block:: python` used or not,
                the docstring in PS1(>>> ) and PS2(... ) should be considered as codeblock.
        target(str): `target` should be in {'docstring', 'codeblock'}.
            `docstring`, the test target is a docstring with optional description, `Args:`, `Returns:`, `Examples:` and so on.
            `codeblock`, the codeblock extracted by `extract_code_blocks_from_docstr` from the docstring, and the pure codeblock is the docstring to test.
                If we use `.. code-block:: python` wrapping the codeblock, the target should be `codeblock` instead of `docstring`.
                Because the `doctest` and `xdoctest` do NOT care the `.. code-block:: python` directive.
                If the `style` is set to `google` and `target` is set to `codeblock`, we should implement/overwrite `ensemble_docstring` method,
                where ensemble the codeblock into a docstring with a `Examples:` and some indents as least.
        directives(list[str]): `DocTester` hold the default directives, we can/should replace them with method `convert_directive`.
    """

    style = 'google'
    target = 'docstring'
    directives = [
        "# doctest: +SKIP",
        "# doctest: +REQUIRES(env:CPU)",
        "# doctest: +REQUIRES(env:GPU)",
        "# doctest: +REQUIRES(env:XPU)",
        "# doctest: +REQUIRES(env:DISTRIBUTED)",
        "# doctest: +REQUIRES(env:GPU, env:XPU)",
    ]

    def ensemble_docstring(self, codeblock: str) -> str:
        """Ensemble a cleaned codeblock into a docstring.

        For example, we can add `Example:` before the code block and some indents, which makes it a `google` style docstring.
        Otherwise, a codeblock is just a `freeform` style docstring.

        Args:
            codeblock(str): a str of codeblock and its outputs.

        Returns:
            a docstring for test.
        """
        if self.style == 'google':
            return 'Examples:\n' + '\n'.join(
                ['    ' + line for line in codeblock.splitlines()]
            )

        return codeblock

    def convert_directive(self, docstring: str) -> str:
        """Convert the standard directive from default DocTester into the doctester's style:

        For example:
        From: # doctest: +SKIP
        To: # xdoctest: +SKIP

        Args:
            docstring(str): the raw docstring

        Returns:
            a docstring with directives converted.
        """
        return docstring

    def prepare(self, test_capacity: set) -> None:
        """Something before run the test.

        Xdoctest need to set the `os.environ` according to the test capacity,
        which `+REQUIRES` used to match the test required environment.

        Legacy sample code processor do NOT need.

        Args:
            test_capacity(set): the test capacity, like `cpu`, `gpu` and so on.
        """
        pass

    def run(self, api_name: str, docstring: str) -> typing.List[TestResult]:
        """Extract codeblocks from docstring, and run the test.
        Run only one docstring at a time.

        Args:
            api_name(str): api name
            docstring(str): docstring.

        Returns:
            list[TestResult]: test results. because one docstring may extract more than one code examples, so return a list.
        """
        raise NotImplementedError

    def print_summary(
        self, test_results: typing.List[TestResult], whl_error: typing.List[str]
    ) -> None:
        """Post process test results and print test summary.

        There are some `required not match` in legacy test processor, but NOT exist in Xdoctest.
        When using the legacy processor, we can set test result to `skipped=True` and store the `not match` information in `extra_info`,
        then logging the `not match` in `print_summary`.

        Args:
            test_results(list[TestResult]): test results generated from doctester.
            whl_error(list[str]): wheel error when we extract apis from module.
        """
        pass


def get_api_md5(path):
    """
    read the api spec file, and scratch the md5sum value of every api's docstring.

    Args:
        path: the api spec file. ATTENTION the path relative

    Returns:
        api_md5(dict): key is the api's real fullname, value is the md5sum.
    """
    api_md5 = {}
    API_spec = os.path.abspath(os.path.join(os.getcwd(), "..", path))
    if not os.path.isfile(API_spec):
        return api_md5
    pat = re.compile(r'\((paddle[^,]+)\W*document\W*([0-9a-z]{32})')
    patArgSpec = re.compile(
        r'^(paddle[^,]+)\s+\(ArgSpec.*document\W*([0-9a-z]{32})'
    )
    with open(API_spec) as f:
        for line in f.readlines():
            mo = pat.search(line)
            if not mo:
                mo = patArgSpec.search(line)
            if mo:
                api_md5[mo.group(1)] = mo.group(2)
    return api_md5


def get_incrementapi():
    '''
    this function will get the apis that difference between API_DEV.spec and API_PR.spec.
    '''
    global API_DEV_SPEC_FN, API_PR_SPEC_FN, API_DIFF_SPEC_FN  # readonly
    dev_api = get_api_md5(API_DEV_SPEC_FN)
    pr_api = get_api_md5(API_PR_SPEC_FN)
    with open(API_DIFF_SPEC_FN, 'w') as f:
        for key in pr_api:
            if key in dev_api:
                if dev_api[key] != pr_api[key]:
                    logger.debug(
                        "%s in dev is %s, different from pr's %s",
                        key,
                        dev_api[key],
                        pr_api[key],
                    )
                    f.write(key)
                    f.write('\n')
            else:
                logger.debug("%s is not in dev", key)
                f.write(key)
                f.write('\n')


def get_full_api_by_walk():
    """
    get all the apis
    """
    global API_DIFF_SPEC_FN  # readonly
    from print_signatures import get_all_api

    apilist = get_all_api()
    with open(API_DIFF_SPEC_FN, 'w') as f:
        f.write("\n".join([ai[0] for ai in apilist]))


def get_full_api_from_pr_spec():
    """
    get all the apis
    """
    global API_PR_SPEC_FN, API_DIFF_SPEC_FN  # readonly
    pr_api = get_api_md5(API_PR_SPEC_FN)
    if len(pr_api):
        with open(API_DIFF_SPEC_FN, 'w') as f:
            f.write("\n".join(pr_api.keys()))
    else:
        get_full_api_by_walk()


def get_full_api():
    """
    get all the apis
    """
    global API_DIFF_SPEC_FN  # readonly
    from print_signatures import get_all_api_from_modulelist

    member_dict = get_all_api_from_modulelist()
    with open(API_DIFF_SPEC_FN, 'w') as f:
        f.write("\n".join(member_dict.keys()))


def extract_code_blocks_from_docstr(docstr, google_style=True):
    """
    extract code-blocks from the given docstring.
    DON'T include the multiline-string definition in code-blocks.
    The *Examples* section must be the last.
    Args:
        docstr(str): docstring
        google_style(bool): if not use google_style, the code blocks will be extracted from all the parts of docstring.
    Return:
        code_blocks: A list of code-blocks, indent removed.
                     element {'name': the code-block's name, 'id': sequence id.
                              'codes': codes, 'required': 'gpu', 'in_examples': bool, code block in `Examples` or not,}
    """
    code_blocks = []

    mo = re.search(r"Examples?:", docstr)

    if google_style and mo is None:
        return code_blocks

    example_start = len(docstr) if mo is None else mo.start()
    docstr_describe = docstr[:example_start].splitlines()
    docstr_examples = docstr[example_start:].splitlines()

    docstr_list = []
    if google_style:
        example_lineno = 0
        docstr_list = docstr_examples
    else:
        example_lineno = len(docstr_describe)
        docstr_list = docstr_describe + docstr_examples

    lastlineindex = len(docstr_list) - 1

    cb_start_pat = re.compile(r"code-block::\s*python")
    cb_param_pat = re.compile(r"^\s*:(\w+):\s*(\S*)\s*$")
    cb_required_pat = re.compile(r"^\s*#\s*require[s|d]\s*:\s*(\S+)\s*$")

    cb_info = {}
    cb_info['cb_started'] = False
    cb_info['cb_cur'] = []
    cb_info['cb_cur_indent'] = -1
    cb_info['cb_cur_name'] = None
    cb_info['cb_cur_seq_id'] = 0
    cb_info['cb_required'] = None

    def _cb_started():
        # nonlocal cb_started, cb_cur_name, cb_required, cb_cur_seq_id
        cb_info['cb_started'] = True
        cb_info['cb_cur_seq_id'] += 1
        cb_info['cb_cur_name'] = None
        cb_info['cb_required'] = None

    def _append_code_block(in_examples):
        # nonlocal code_blocks, cb_cur, cb_cur_name, cb_cur_seq_id, cb_required
        code_blocks.append(
            {
                'codes': inspect.cleandoc("\n" + "\n".join(cb_info['cb_cur'])),
                'name': cb_info['cb_cur_name'],
                'id': cb_info['cb_cur_seq_id'],
                'required': cb_info['cb_required'],
                'in_examples': in_examples,
            }
        )

    for lineno, linecont in enumerate(docstr_list):
        if re.search(cb_start_pat, linecont):
            if not cb_info['cb_started']:
                _cb_started()
                continue
            else:
                # cur block end
                if len(cb_info['cb_cur']):
                    _append_code_block(lineno > example_lineno)
                _cb_started()  # another block started
                cb_info['cb_cur_indent'] = -1
                cb_info['cb_cur'] = []
        else:
            if cb_info['cb_started']:
                # handle the code-block directive's options
                mo_p = cb_param_pat.match(linecont)
                if mo_p:
                    if mo_p.group(1) == 'name':
                        cb_info['cb_cur_name'] = mo_p.group(2)
                    continue
                # read the required directive
                mo_r = cb_required_pat.match(linecont)
                if mo_r:
                    cb_info['cb_required'] = mo_r.group(1)
                # docstring end
                if lineno == lastlineindex:
                    mo = re.search(r"\S", linecont)
                    if (
                        mo is not None
                        and cb_info['cb_cur_indent'] <= mo.start()
                    ):
                        cb_info['cb_cur'].append(linecont)
                    if len(cb_info['cb_cur']):
                        _append_code_block(lineno > example_lineno)
                    break
                # check indent for cur block start and end.
                if cb_info['cb_cur_indent'] < 0:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        continue
                    # find the first non empty line
                    cb_info['cb_cur_indent'] = mo.start()
                    cb_info['cb_cur'].append(linecont)
                else:
                    mo = re.search(r"\S", linecont)
                    if mo is None:
                        cb_info['cb_cur'].append(linecont)
                        continue
                    if cb_info['cb_cur_indent'] <= mo.start():
                        cb_info['cb_cur'].append(linecont)
                    else:
                        if linecont[mo.start()] == '#':
                            continue
                        else:
                            # block end
                            if len(cb_info['cb_cur']):
                                _append_code_block(lineno > example_lineno)
                            cb_info['cb_started'] = False
                            cb_info['cb_cur_indent'] = -1
                            cb_info['cb_cur'] = []
    return code_blocks


def init_logger(debug=True, log_file=None):
    """
    init logger level and file handler
    """
    if debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)

    if log_file is not None:
        logfHandler = logging.FileHandler(log_file)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(logfHandler)


def check_test_mode(mode="cpu", gpu_id=0):
    """
    check test mode in {cpu, gpu}
    """
    if mode == "gpu":
        logger.info("using GPU_ID %d", gpu_id)

    elif mode == "cpu":
        logger.info("using CPU")

    else:
        logger.error(
            "Unrecognized argument:%s, 'cpu' or 'gpu' is desired.", mode
        )
        sys.exit("Invalid arguments")

    return mode


def get_test_capacity(run_on_device="cpu"):
    """
    collect capacities and set to sample_code_test_capacity
    """
    sample_code_test_capacity = set()
    if ENV_KEY_TEST_CAPACITY in os.environ:
        for env_value in os.environ[ENV_KEY_TEST_CAPACITY].split(','):
            if env_value:
                sample_code_test_capacity.add(env_value.strip().lower())

    if 'cpu' not in sample_code_test_capacity:
        sample_code_test_capacity.add('cpu')

    if run_on_device:
        sample_code_test_capacity.add(run_on_device)

    logger.info("Sample code test capacity: %s", sample_code_test_capacity)

    return sample_code_test_capacity


def get_docstring(full_test=False):
    '''
    this function will get the docstring for test.
    '''
    import paddle  # noqa: F401
    import paddle.static.quantization  # noqa: F401

    if full_test:
        get_full_api_from_pr_spec()
    else:
        get_incrementapi()

    docstrings_to_test = {}
    whl_error = []
    with open(API_DIFF_SPEC_FN) as f:
        for line in f.readlines():
            api = line.replace('\n', '')
            try:
                api_obj = eval(api)
            except AttributeError:
                whl_error.append(api)
                continue
            except SyntaxError:
                logger.warning('line:%s, api:%s', line, api)
                # paddle.Tensor.<lambda>
                continue
            if hasattr(api_obj, '__doc__') and api_obj.__doc__:
                docstrings_to_test[api] = api_obj.__doc__

    if len(docstrings_to_test) == 0 and len(whl_error) == 0:
        logger.info("-----API_PR.spec is the same as API_DEV.spec-----")
        sys.exit(0)
    logger.info("API_PR is diff from API_DEV: %s", docstrings_to_test.keys())
    logger.info("Total api: %s", len(docstrings_to_test.keys()))

    return docstrings_to_test, whl_error


def exec_gen_doc():
    result = True
    cmd = ["bash", "document_preview.sh"]
    logger.info("----exec gen_doc----")
    start_time = time.time()
    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = subprc.communicate()
    msg = "".join(output.decode(encoding='utf-8'))
    err = "".join(error.decode(encoding='utf-8'))
    end_time = time.time()

    if subprc.returncode != 0:
        logger.info("----gen_doc msg----")
        logger.info(msg)
        logger.error("----gen_doc error msg----")
        logger.error(err)
        logger.error("----exec gen_doc failed----")
        result = False
    else:
        logger.info("----gen_doc msg----")
        logger.info(msg)
        logger.info("----exec gen_doc success----")

    for fn in [
        '/docs/en/develop/index_en.html',
        '/docs/zh/develop/index_cn.html',
    ]:
        if os.path.exists(fn):
            logger.info('%s exists.', fn)
        else:
            logger.error('%s not exists.', fn)

    # msg is the returned code execution report
    return result, msg, end_time - start_time


def get_test_results(
    doctester: DocTester, docstrings_to_test: typing.Dict[str, str]
) -> typing.List[TestResult]:
    """Get test results from doctester with docstrings to test."""
    _test_style = (
        doctester.style
        if doctester.style in {'google', 'freeform'}
        else 'google'
    )
    google_style = _test_style == 'google'

    test_results = []
    for api_name, raw_docstring in docstrings_to_test.items():
        docstrings_extracted = []
        if doctester.target == 'codeblock':
            # if the target is `codeblock`, we may extract more than one codeblocks from docsting.
            for codeblock in extract_code_blocks_from_docstr(
                raw_docstring, google_style=google_style
            ):
                codeblock_name = codeblock['name']
                codeblock_id = codeblock['id']
                docstring = doctester.ensemble_docstring(
                    codeblock=codeblock['codes']
                )
                docstring_name = '{}:{}'.format(
                    api_name, codeblock_name or codeblock_id
                )

                docstrings_extracted.append(
                    {'name': docstring_name, 'docstring': docstring}
                )
        else:
            docstrings_extracted.append(
                {'name': api_name, 'docstring': raw_docstring}
            )

        for doc_extracted in docstrings_extracted:
            # run docstester for one docstring at a time.
            test_results.extend(
                doctester.run(
                    api_name=doc_extracted['name'],
                    docstring=doctester.convert_directive(
                        doc_extracted['docstring']
                    ),
                )
            )

    return test_results


def run_doctest(args, doctester: DocTester):
    logger.info("----------------Codeblock Check Start--------------------")

    # init logger
    init_logger(debug=args.debug, log_file=args.logf)

    logger.info("Check test mode ...")
    run_on_device = check_test_mode(mode=args.mode, gpu_id=args.gpu_id)

    logger.info("Get test capacity ...")
    sample_code_test_capacity = get_test_capacity(run_on_device)

    logger.info("Get docstring from api ...")
    docstrings_to_test, whl_error = get_docstring(full_test=args.full_test)

    logger.info("Prepare doctester ...")
    doctester.prepare(sample_code_test_capacity)

    logger.info("Running doctester ...")
    test_results = get_test_results(doctester, docstrings_to_test)

    logger.info("Print summary ...")
    doctester.print_summary(test_results, whl_error)

    if args.mode == "cpu":
        # As cpu mode is also run with the GPU whl, so skip it in gpu mode.
        exec_gen_doc()


arguments = [
    # flags, dest, type, default, help
    ['--gpu_id', 'gpu_id', int, 0, 'GPU device id to use [0]'],
    ['--logf', 'logf', str, None, 'file for logging'],
    ['--threads', 'threads', int, 0, 'sub processes number'],
]


def parse_args():
    """
    Parse input arguments
    """
    global arguments
    parser = argparse.ArgumentParser(description='run Sample Code Test')
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument('--full-test', dest='full_test', action="store_true")
    parser.add_argument('mode', type=str, help='run on device', default='cpu')
    parser.add_argument(
        '--build-doc',
        dest='build_doc',
        action='store_true',
        help='build doc if need.',
    )
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3]
        )

    if len(sys.argv) == 1:
        args = parser.parse_args(['cpu'])
        return args

    args = parser.parse_args()
    return args
