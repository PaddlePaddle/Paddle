# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
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
usage: python sample_test.py {cpu or gpu}
    {cpu or gpu}: running in cpu version or gpu version

for example, you can run cpu version testing like this:

    python sampcd_processor.py cpu

"""
import logging
import multiprocessing
import os
import platform
import re
import shutil
import subprocess
import sys
import time

from sampcd_processor_utils import ENV_KEY_TEST_CAPACITY  # noqa: F401
from sampcd_processor_utils import (
    API_DIFF_SPEC_FN,
    extract_code_blocks_from_docstr,
    get_full_api_from_pr_spec,
    get_incrementapi,
    parse_args,
    run_doctest,
)
from sampcd_processor_xdoctest import Xdoctester

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
SAMPLE_CODE_TEST_CAPACITY = set()
GPU_ID = 0
whl_error = []
SAMPLECODE_TEMPDIR = 'samplecode_temp'
ENV_KEY_CODES_FRONTEND = 'CODES_INSERTED_INTO_FRONTEND'
SUMMARY_INFO = {
    'success': [],
    'failed': [],
    'skiptest': [],
    'nocodes': [],
    # ... required not-match
}


def find_all(srcstr, substr):
    """
    to find all desired substring in the source string
     and return their starting indices as a list

    Args:
        srcstr(str): the parent string
        substr(str): substr

    Returns:
        list: a list of the indices of the substrings
              found
    """
    indices = []
    gotone = srcstr.find(substr)
    while gotone != -1:
        indices.append(gotone)
        gotone = srcstr.find(substr, gotone + 1)
    return indices


def find_last_future_line_end(cbstr):
    """
    find the last `__future__` line.

    Args:
        docstr(str): docstring
    Return:
        index of the line end or None.
    """
    pat = re.compile('__future__.*\n')
    lastmo = None
    it = re.finditer(pat, cbstr)
    while True:
        try:
            lastmo = next(it)
        except StopIteration:
            break
    if lastmo:
        return lastmo.end()
    else:
        return None


def get_test_capacity():
    """
    collect capacities and set to SAMPLE_CODE_TEST_CAPACITY
    """
    global SAMPLE_CODE_TEST_CAPACITY  # write
    global ENV_KEY_TEST_CAPACITY, RUN_ON_DEVICE  # readonly
    if ENV_KEY_TEST_CAPACITY in os.environ:
        for r in os.environ[ENV_KEY_TEST_CAPACITY].split(','):
            rr = r.strip().lower()
            if r:
                SAMPLE_CODE_TEST_CAPACITY.add(rr)
    if 'cpu' not in SAMPLE_CODE_TEST_CAPACITY:
        SAMPLE_CODE_TEST_CAPACITY.add('cpu')

    if RUN_ON_DEVICE:
        SAMPLE_CODE_TEST_CAPACITY.add(RUN_ON_DEVICE)


def is_required_match(requirestr, cbtitle='not-specified'):
    """
    search the required instruction in the code-block, and check it match the current running environment.

    environment values of equipped: cpu, gpu, xpu, distributed, skip
    the 'skip' is the special flag to skip the test, so is_required_match will return False directly.

    Args:
        requirestr(str): the required string.
        cbtitle(str): the title of the code-block.
    returns:
        True - yes, matched
        False - not match
        None - skipped  # trick
    """
    global SAMPLE_CODE_TEST_CAPACITY, RUN_ON_DEVICE  # readonly
    requires = {'cpu'}
    if requirestr:
        for r in requirestr.split(','):
            rr = r.strip().lower()
            if rr:
                requires.add(rr)
    else:
        requires.add(RUN_ON_DEVICE)
    if 'skip' in requires or 'skiptest' in requires:
        logger.info('%s: skipped', cbtitle)
        return None

    if all(
        k in SAMPLE_CODE_TEST_CAPACITY
        for k in requires
        if k not in ['skip', 'skiptest']
    ):
        return True

    logger.info(
        '%s: the equipments [%s] not match the required [%s].',
        cbtitle,
        ','.join(SAMPLE_CODE_TEST_CAPACITY),
        ','.join(requires),
    )
    return False


def insert_codes_into_codeblock(codeblock, apiname='not-specified'):
    """
    insert some codes in the frontend and backend into the code-block.
    """
    global ENV_KEY_CODES_FRONTEND, GPU_ID, RUN_ON_DEVICE  # readonly
    inserted_codes_f = ''
    inserted_codes_b = ''
    if (
        ENV_KEY_CODES_FRONTEND in os.environ
        and os.environ[ENV_KEY_CODES_FRONTEND]
    ):
        inserted_codes_f = os.environ[ENV_KEY_CODES_FRONTEND]
    else:
        cpu_str = '\nimport os\nos.environ["CUDA_VISIBLE_DEVICES"] = ""\n'
        gpu_str = (
            '\nimport os\nos.environ["CUDA_VISIBLE_DEVICES"] = "{}"\n'.format(
                GPU_ID
            )
        )
        if 'required' in codeblock and codeblock['required']:
            if codeblock['required'] == 'cpu':
                inserted_codes_f = cpu_str
            elif codeblock['required'] == 'gpu':
                inserted_codes_f = gpu_str
        else:
            if RUN_ON_DEVICE == "cpu":
                inserted_codes_f = cpu_str
            elif RUN_ON_DEVICE == "gpu":
                inserted_codes_f = gpu_str
    inserted_codes_b = '\nprint("{}\'s sample code (name:{}, id:{}) is executed successfully!")'.format(
        apiname, codeblock['name'], codeblock['id']
    )

    cb = codeblock['codes']
    last_future_line_end = find_last_future_line_end(cb)
    if last_future_line_end:
        return (
            cb[:last_future_line_end]
            + inserted_codes_f
            + cb[last_future_line_end:]
            + inserted_codes_b
        )
    else:
        return inserted_codes_f + cb + inserted_codes_b


def is_ps_wrapped_codeblock(codeblock):
    """If the codeblock is wrapped by PS1(>>> ),
    we skip test and use xdoctest instead.
    """
    codes = codeblock['codes']
    match_obj = re.search(r"\n>>>\s?", "\n" + codes)
    return match_obj is not None


def sampcd_extract_to_file(srccom, name, htype="def", hname=""):
    """
    Extract sample codes from __doc__, and write them to files.

    Args:
        srccom(str): the source comment of some API whose
                     example codes will be extracted and run.
        name(str): the name of the API.
        htype(str): the type of hint banners, def/class/method.
        hname(str): the name of the hint  banners , e.t. def hname.

    Returns:
        sample_code_filenames(list of str)
    """
    global GPU_ID, RUN_ON_DEVICE, SAMPLECODE_TEMPDIR  # readonly
    global SUMMARY_INFO  # update

    codeblocks = extract_code_blocks_from_docstr(srccom)
    if len(codeblocks) == 0:
        SUMMARY_INFO['nocodes'].append(name)
        # detect sample codes using >>> to format and consider this situation as wrong
        logger.info(htype + " name:" + name)
        logger.info("-----------------------")
        if srccom.find("Examples:") != -1:
            logger.info("----example code check----")
            if srccom.find(">>>") != -1:
                logger.warning(
                    r"""Deprecated sample code style:
    Examples:
        >>>codeline
        >>>codeline

Please use '.. code-block:: python' to format the sample code."""
                )
                return []
        else:
            logger.error(
                "Error: No sample code found! Please check if the API comment contais string 'Examples:' correctly"
            )
            return []

    sample_code_filenames = []
    for y, cb in enumerate(codeblocks):
        if is_ps_wrapped_codeblock(cb):
            SUMMARY_INFO['skiptest'].append("{}-{}".format(name, cb['id']))
            logger.info(
                '{}\' code block (name:{}, id:{}) is wrapped by PS1(>>> ), which will be tested by xdoctest.'.format(
                    name, cb['name'], cb['id']
                )
            )
            continue

        matched = is_required_match(cb['required'], name)
        # matched has three states:
        # True - please execute it;
        # None - no sample code found;
        # False - it need other special equipment or environment.
        # so, the following conditional statements are intentionally arranged.
        if matched:
            tfname = os.path.join(
                SAMPLECODE_TEMPDIR,
                '{}_example{}'.format(
                    name,
                    '.py' if len(codeblocks) == 1 else f'_{y + 1}.py',
                ),
            )
            with open(tfname, 'w') as tempf:
                sampcd = insert_codes_into_codeblock(cb, name)
                tempf.write(sampcd)
            sample_code_filenames.append(tfname)
        elif matched is None:
            logger.info(
                '{}\' code block (name:{}, id:{}) is skipped.'.format(
                    name, cb['name'], cb['id']
                )
            )
            SUMMARY_INFO['skiptest'].append("{}-{}".format(name, cb['id']))
        elif not matched:
            logger.info(
                '{}\' code block (name:{}, id:{}) required({}) not match capacity({}).'.format(
                    name,
                    cb['name'],
                    cb['id'],
                    cb['required'],
                    SAMPLE_CODE_TEST_CAPACITY,
                )
            )
            if cb['required'] not in SUMMARY_INFO:
                SUMMARY_INFO[cb['required']] = []
            SUMMARY_INFO[cb['required']].append("{}-{}".format(name, cb['id']))

    return sample_code_filenames


def execute_samplecode(tfname):
    """
    Execute a sample-code test

    Args:
        tfname: the filename of the sample code

    Returns:
        result: success or not
        tfname: same as the input argument
        msg: the stdout output of the sample code executing
        time: time consumed by sample code
    """
    result = True
    msg = None
    if platform.python_version()[0] in ["3"]:
        cmd = [sys.executable, tfname]
    else:
        logger.error("Error: fail to parse python version!")
        result = False
        sys.exit(1)

    logger.info("----example code check----")
    logger.info("executing sample code: %s", tfname)
    start_time = time.time()
    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    output, error = subprc.communicate()
    msg = "".join(output.decode(encoding='utf-8'))
    err = "".join(error.decode(encoding='utf-8'))
    end_time = time.time()

    if subprc.returncode != 0:
        with open(tfname, 'r') as f:
            logger.warning(
                """Sample code error found in %s:
-----------------------
%s
-----------------------
subprocess return code: %d
Error Raised from Sample Code:
stderr: %s
stdout: %s
""",
                tfname,
                f.read(),
                subprc.returncode,
                err,
                msg,
            )
        logger.info("----example code check failed----")
        result = False
    else:
        logger.info("----example code check success----")

    # msg is the returned code execution report
    return result, tfname, msg, end_time - start_time


def get_filenames(full_test=False):
    '''
    this function will get the sample code files that pending for check.

    Args:
        full_test: the full apis or the increment

    Returns:

        dict: the sample code files pending for check .

    '''
    global whl_error
    import paddle  # noqa: F401
    import paddle.static.quantization  # noqa: F401

    whl_error = []
    if full_test:
        get_full_api_from_pr_spec()
    else:
        get_incrementapi()
    all_sample_code_filenames = {}
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
                sample_code_filenames = sampcd_extract_to_file(
                    api_obj.__doc__, api
                )
                for tfname in sample_code_filenames:
                    all_sample_code_filenames[tfname] = api
    return all_sample_code_filenames


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    else:
        logger.setLevel(logging.INFO)
    if args.logf:
        logfHandler = logging.FileHandler(args.logf)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            )
        )
        logger.addHandler(logfHandler)

    if args.mode == "gpu":
        GPU_ID = args.gpu_id
        logger.info("using GPU_ID %d", GPU_ID)
    elif args.mode != "cpu":
        logger.error(
            "Unrecognized argument:%s, 'cpu' or 'gpu' is desired.", args.mode
        )
        sys.exit("Invalid arguments")
    RUN_ON_DEVICE = args.mode
    get_test_capacity()
    logger.info("API check -- Example Code")
    logger.info(
        "sample_test running under python %s", platform.python_version()
    )

    if os.path.exists(SAMPLECODE_TEMPDIR):
        if not os.path.isdir(SAMPLECODE_TEMPDIR):
            os.remove(SAMPLECODE_TEMPDIR)
            os.mkdir(SAMPLECODE_TEMPDIR)
    else:
        os.mkdir(SAMPLECODE_TEMPDIR)

    filenames = get_filenames(args.full_test)
    if len(filenames) == 0 and len(whl_error) == 0:
        logger.info("-----API_PR.spec is the same as API_DEV.spec-----")
        # not exit if no filenames, we should do xdoctest later.
        # sys.exit(0)

        # delete temp files
        if not args.debug:
            shutil.rmtree(SAMPLECODE_TEMPDIR)

    else:
        logger.info("API_PR is diff from API_DEV: %s", filenames)

        threads = multiprocessing.cpu_count()
        if args.threads:
            threads = args.threads
        po = multiprocessing.Pool(threads)
        results = po.map_async(execute_samplecode, filenames.keys())
        po.close()
        po.join()

        result = results.get()

        # delete temp files
        if not args.debug:
            shutil.rmtree(SAMPLECODE_TEMPDIR)

        stdout_handler = logging.StreamHandler(stream=sys.stdout)
        logger.addHandler(stdout_handler)
        logger.info("----------------End of the Check--------------------")
        if len(whl_error) != 0:
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
            for temp in result:
                if not temp[0]:
                    logger.info(
                        "In addition, mistakes found in sample codes: %s",
                        temp[1],
                    )
            logger.info("----------------------------------------------------")
            sys.exit(1)
        else:
            timeovered_test = {}
            for temp in result:
                if not temp[0]:
                    logger.info(
                        "In addition, mistakes found in sample codes: %s",
                        temp[1],
                    )
                    SUMMARY_INFO['failed'].append(temp[1])
                else:
                    SUMMARY_INFO['success'].append(temp[1])
                if temp[3] > 10:
                    timeovered_test[temp[1]] = temp[3]

            if len(timeovered_test):
                logger.info(
                    "%d sample codes ran time over 10s", len(timeovered_test)
                )
                if args.debug:
                    for k, v in timeovered_test.items():
                        logger.info(f'{k} - {v}s')
            if len(SUMMARY_INFO['success']):
                logger.info(
                    "%d sample codes ran success", len(SUMMARY_INFO['success'])
                )
            for k, v in SUMMARY_INFO.items():
                if k not in ['success', 'failed', 'skiptest', 'nocodes']:
                    logger.info(
                        "%d sample codes required not match for %s", len(v), k
                    )
            if len(SUMMARY_INFO['skiptest']):
                logger.info(
                    "%d sample codes skipped", len(SUMMARY_INFO['skiptest'])
                )
                if args.debug:
                    logger.info('\n'.join(SUMMARY_INFO['skiptest']))
            if len(SUMMARY_INFO['nocodes']):
                logger.info(
                    "%d apis don't have sample codes",
                    len(SUMMARY_INFO['nocodes']),
                )
                if args.debug:
                    logger.info('\n'.join(SUMMARY_INFO['nocodes']))
            if len(SUMMARY_INFO['failed']):
                logger.info(
                    "%d sample codes ran failed", len(SUMMARY_INFO['failed'])
                )
                logger.info('\n'.join(SUMMARY_INFO['failed']))
                logger.info(
                    "Mistakes found in sample codes. Please recheck the sample codes."
                )
                sys.exit(1)

        logger.info("Sample code check is successful!")

    # run xdoctest
    run_doctest(args, doctester=Xdoctester(debug=args.debug))
