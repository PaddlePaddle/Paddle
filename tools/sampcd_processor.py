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

import os
import sys
import subprocess
import multiprocessing
import math
import platform
import inspect
import json
import argparse
import shutil
import re
import logging
"""
please make sure to run in the tools path
usage: python sample_test.py {cpu or gpu} 
    {cpu or gpu}: running in cpu version or gpu version

for example, you can run cpu version python2 testing like this:

    python sampcd_processor.py cpu 

"""

logger = logging.getLogger()
if logger.handlers:
    console = logger.handlers[
        0]  # we assume the first handler is the one we want to configure
else:
    console = logging.StreamHandler()
    logger.addHandler(console)
console.setFormatter(logging.Formatter("%(message)s"))

RUN_ON_DEVICE = 'cpu'
GPU_ID = 0
methods = []
whl_error = []
API_DEV_SPEC_FN = 'paddle/fluid/API_DEV.spec'
API_PR_SPEC_FN = 'paddle/fluid/API_PR.spec'
API_DIFF_SPEC_FN = 'dev_pr_diff_api.spec'
SAMPLECODE_TEMPDIR = 'samplecode_temp'


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
    while (gotone != -1):
        indices.append(gotone)
        gotone = srcstr.find(substr, gotone + 1)
    return indices


def check_indent(cdline):
    """
    to check the indent of a given code line

    to get the number of starting blank chars,
    e.t. blankspaces and \t

    \t will be interpreted as 4 single blankspaces,
    e.t. '\t'='    '

    Args:
        cdline(str) : a single line of code from the source file

    Returns:
        int : the indent of the number of interpreted
             blankspaces
    """
    indent = 0
    for c in cdline:
        if c == '\t':
            indent += 4
        elif c == ' ':
            indent += 1
        if c != ' ' and c != '\t':
            break
    return indent


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
    global GPU_ID, RUN_ON_DEVICE, SAMPLECODE_TEMPDIR
    CODE_BLOCK_INTERDUCTORY = "code-block:: python"

    sampcd_begins = find_all(srccom, CODE_BLOCK_INTERDUCTORY)
    if len(sampcd_begins) == 0:
        # detect sample codes using >>> to format and consider this situation as wrong
        print(htype, " name:", hname)
        print("-----------------------")
        if srccom.find("Examples:") != -1:
            print("----example code check----\n")
            if srccom.find(">>>") != -1:
                print(
                    "Deprecated sample code style:\n\n    Examples:\n\n        >>>codeline\n        >>>codeline\n\n\n ",
                    "Please use '.. code-block:: python' to ",
                    "format sample code.\n")
                return []
        else:
            print("Error: No sample code!\n")
            return []
    sample_code_filenames = []
    for y in range(1, len(sampcd_begins) + 1):
        sampcd_begin = sampcd_begins[y - 1]
        sampcd = srccom[sampcd_begin + len(CODE_BLOCK_INTERDUCTORY) + 1:]
        sampcd = sampcd.split("\n")
        # remove starting empty lines
        while sampcd[0].replace(' ', '').replace('\t', '') == '':
            sampcd.pop(0)

        # the minimum indent, which is the indent of the first
        # non-empty line
        min_indent = check_indent(sampcd[0])
        sampcd_to_write = []
        for i in range(0, len(sampcd)):
            cdline = sampcd[i]
            # handle empty lines or those only with spaces/tabs
            if cdline.strip() == '':
                continue
            this_indent = check_indent(cdline)
            if this_indent < min_indent:
                break
            else:
                cdline = cdline.replace('\t', '    ')
                sampcd_to_write.append(cdline[min_indent:])

        sampcd = '\n'.join(sampcd_to_write)
        if RUN_ON_DEVICE == "cpu":
            sampcd = '\nimport os\nos.environ["CUDA_VISIBLE_DEVICES"] = ""\n' + sampcd
        if RUN_ON_DEVICE == "gpu":
            sampcd = '\nimport os\nos.environ["CUDA_VISIBLE_DEVICES"] = "{}"\n'.format(
                GPU_ID) + sampcd
        sampcd += '\nprint(' + '\"' + name + ' sample code is executed successfully!\")'

        tfname = os.path.join(SAMPLECODE_TEMPDIR, '{}_example{}'.format(
            name, '.py' if len(sampcd_begins) == 1 else '_{}.py'.format(y)))
        with open(tfname, 'w') as tempf:
            tempf.write(sampcd)
        sample_code_filenames.append(tfname)
    return sample_code_filenames


def execute_samplecode(tfname):
    """
    Execute a sample-code test.

    Args:
        tfname: the filename of the samplecode.
    
    Returns:
        result: success or not
        tfname: same as the input argument
        msg: the stdout output of the samplecode executing.
    """
    result = True
    msg = None
    if platform.python_version()[0] in ["2", "3"]:
        cmd = [sys.executable, tfname]
    else:
        print("Error: fail to parse python version!")
        result = False
        exit(1)

    # check required envisonment
    with open(tfname, 'r') as f:
        for line in f.readlines():
            if re.match(r'#\s*required\s*:\s*(distributed|gpu|skip)', line):
                result = True
                return result, tfname, '{} is skipped. cause: {}'.format(tfname,
                                                                         line)

    logging.info('running %s', tfname)
    print("\n----example code check----")
    print("executing sample code .....", tfname)
    subprc = subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    output, error = subprc.communicate()
    msg = "".join(output.decode(encoding='utf-8'))
    err = "".join(error.decode(encoding='utf-8'))

    if subprc.returncode != 0:
        print("Sample code error found in ", tfname, ":")
        print("-----------------------")
        print(open(tfname).read())
        print("-----------------------")
        print("subprocess return code: ", str(subprc.returncode))
        print("Error Raised from Sample Code ", tfname, " :")
        print(err)
        print(msg)
        print("----example code check failed----\n")
        logging.warning('%s error: %s', tfname, err)
        logging.warning('%s msg: %s', tfname, msg)
        result = False
    else:
        print("----example code check success----\n")

    # msg is the returned code execution report
    return result, tfname, msg


def get_filenames():
    '''
    this function will get the sample code files that pending for check.

    Returns:

        dict: the sample code files pending for check .

    '''
    global methods  # write
    global whl_error
    import paddle
    whl_error = []
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
                sample_code_filenames = sampcd_extract_to_file(api_obj.__doc__,
                                                               api)
                for tfname in sample_code_filenames:
                    all_sample_code_filenames[tfname] = api
    return all_sample_code_filenames


def get_api_md5(path):
    """
    read the api spec file, and scratch the md5sum value of every api's docstring.

    Args:
        path: the api spec file. ATTENTION the path relative
    
    Returns:
        api_md5(dict): key is the api's real fullname, value is the md5sum.
    """
    api_md5 = {}
    API_spec = '%s/%s' % (os.path.abspath(os.path.join(os.getcwd(), "..")),
                          path)
    pat = re.compile(r'\((paddle[^,]+)\W*document\W*([0-9a-z]{32})')
    patArgSpec = re.compile(
        r'^(paddle[^,]+)\s+\(ArgSpec.*document\W*([0-9a-z]{32})')
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
    global API_DEV_SPEC_FN, API_PR_SPEC_FN, API_DIFF_SPEC_FN  ## readonly
    dev_api = get_api_md5(API_DEV_SPEC_FN)
    pr_api = get_api_md5(API_PR_SPEC_FN)
    with open(API_DIFF_SPEC_FN, 'w') as f:
        for key in pr_api:
            if key in dev_api:
                if dev_api[key] != pr_api[key]:
                    logger.debug("%s in dev is %s, different from pr's %s", key,
                                 dev_api[key], pr_api[key])
                    f.write(key)
                    f.write('\n')
            else:
                logger.debug("%s is not in dev", key)
                f.write(key)
                f.write('\n')


def get_wlist(fn="wlist.json"):
    '''
    this function will get the white list of API.

    Returns:

        wlist: a list of API that should not trigger the example check .

    '''
    wlist = []
    wlist_file = []
    # only white on CPU
    gpu_not_white = []
    with open(fn, 'r') as load_f:
        load_dict = json.load(load_f)
        for key in load_dict:
            if key == 'wlist_dir':
                for item in load_dict[key]:
                    wlist_file.append(item["name"])
            elif key == "gpu_not_white":
                gpu_not_white = load_dict[key]
            elif key == "wlist_api":
                for item in load_dict[key]:
                    wlist.append(item["name"])
            else:
                wlist = wlist + load_dict[key]
    return wlist, wlist_file, gpu_not_white


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
    # parser.add_argument('--cpu', dest='cpu_mode', action="store_true",
    #                     help='Use CPU mode (overrides --gpu)')
    # parser.add_argument('--gpu', dest='gpu_mode', action="store_true")
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument('mode', type=str, help='run on device', default='cpu')
    for item in arguments:
        parser.add_argument(
            item[0], dest=item[1], help=item[4], type=item[2], default=item[3])

    if len(sys.argv) == 1:
        args = parser.parse_args(['cpu'])
        return args
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    if args.debug:
        logger.setLevel(logging.DEBUG)
    if args.logf:
        logfHandler = logging.FileHandler(args.logf)
        logfHandler.setFormatter(
            logging.Formatter(
                "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"
            ))
        logger.addHandler(logfHandler)

    wlist, wlist_file, gpu_not_white = get_wlist()

    if args.mode == "gpu":
        GPU_ID = args.gpu_id
        logger.info("using GPU_ID %d", GPU_ID)
        for _gnw in gpu_not_white:
            wlist.remove(_gnw)
    elif args.mode != "cpu":
        logger.error("Unrecognized argument:%s, 'cpu' or 'gpu' is desired.",
                     args.mode)
        sys.exit("Invalid arguments")
    RUN_ON_DEVICE = args.mode
    logger.info("API check -- Example Code")
    logger.info("sample_test running under python %s",
                platform.python_version())

    if os.path.exists(SAMPLECODE_TEMPDIR):
        if not os.path.isdir(SAMPLECODE_TEMPDIR):
            os.remove(SAMPLECODE_TEMPDIR)
            os.mkdir(SAMPLECODE_TEMPDIR)
    else:
        os.mkdir(SAMPLECODE_TEMPDIR)

    filenames = get_filenames()
    if len(filenames) == 0 and len(whl_error) == 0:
        logger.info("-----API_PR.spec is the same as API_DEV.spec-----")
        exit(0)
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

    logger.info("----------------End of the Check--------------------")
    if len(whl_error) != 0:
        logger.info("%s is not in whl.", whl_error)
        logger.info("")
        logger.info("Please check the whl package and API_PR.spec!")
        logger.info("You can follow these steps in order to generate API.spec:")
        logger.info("1. cd ${paddle_path}, compile paddle;")
        logger.info("2. pip install build/python/dist/(build whl package);")
        logger.info(
            "3. run 'python tools/print_signatures.py paddle > paddle/fluid/API.spec'."
        )
        for temp in result:
            if not temp[0]:
                logger.info("In addition, mistakes found in sample codes: %s",
                            temp[1])
                logger.info("error_methods: %s", str(temp[2]))
        logger.info("----------------------------------------------------")
        exit(1)
    else:
        has_error = False
        for temp in result:
            if not temp[0]:
                logger.info("In addition, mistakes found in sample codes: %s",
                            temp[1])
                logger.info("error_methods: %s", str(temp[2]))
                has_error = True
        if has_error:
            logger.info("Mistakes found in sample codes.")
            logger.info("Please check sample codes.")
            exit(1)
    logger.info("Sample code check is successful!")
