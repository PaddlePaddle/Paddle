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
import paddle
import paddle.fluid
import json
import argparse
import shutil
import re
import logging
"""
please make sure to run in the tools path
usage: python sample_test.py {arg1} 
arg1: the first arg defined running in gpu version or cpu version

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
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"))

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


# srccom: raw comments in the source,including ''' and original indent
def sampcd_extract_and_run(srccom, name, htype="def", hname=""):
    """
    Extract and run sample codes from source comment and
    the result will be returned.

    Args:
        srccom(str): the source comment of some API whose
                     example codes will be extracted and run.
        name(str): the name of the API.
        htype(str): the type of hint banners, def/class/method.
        hname(str): the name of the hint  banners , e.t. def hname.

    Returns:
        result: True or False
        name(str): the name of the API.
        msg(str): messages
    """
    global GPU_ID, RUN_ON_DEVICE, SAMPLECODE_TEMPDIR

    result = True
    msg = None

    def sampcd_header_print(name, sampcd, htype, hname):
        """
        print hint banner headers.

        Args:
            name(str): the name of the API.
            sampcd(str): sample code string
            htype(str): the type of hint banners, def/class/method.
            hname(str): the name of the hint  banners , e.t. def hname.
            flushed.
        """
        print(htype, " name:", hname)
        print("-----------------------")
        print("Sample code ", str(y), " extracted for ", name, "   :")
        print(sampcd)
        print("----example code check----\n")
        print("executing sample code .....")
        print("execution result:")

    sampcd_begins = find_all(srccom, " code-block:: python")
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
                result = False
        else:
            print("Error: No sample code!\n")
            result = False

    for y in range(1, len(sampcd_begins) + 1):
        sampcd_begin = sampcd_begins[y - 1]
        sampcd = srccom[sampcd_begin + len(" code-block:: python") + 1:]
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
        logging.info('running %s', tfname)
        with open(tfname, 'w') as tempf:
            tempf.write(sampcd)
        if platform.python_version()[0] == "2":
            cmd = ["python", tfname]
        elif platform.python_version()[0] == "3":
            cmd = ["python3", tfname]
        else:
            print("Error: fail to parse python version!")
            result = False
            exit(1)

        subprc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output, error = subprc.communicate()
        msg = "".join(output.decode(encoding='utf-8'))
        err = "".join(error.decode(encoding='utf-8'))

        if subprc.returncode != 0:
            print("\nSample code error found in ", name, ":\n")
            sampcd_header_print(name, sampcd, htype, hname)
            print("subprocess return code: ", str(subprc.returncode))
            print("Error Raised from Sample Code ", name, " :\n")
            print(err)
            print(msg)
            logging.warning('%s error: %s', tfname, err)
            logging.warning('%s msg: %s', tfname, msg)
            result = False
        # msg is the returned code execution report

    return result, name, msg


def single_defcom_extract(start_from, srcls, is_class_begin=False):
    """
    to extract a def function/class/method comments body

    Args:
        start_from(int): the line num of "def" header
        srcls(list): the source file in lines
        is_class_begin(bool): whether the start_from is a beginning a class. \
        For a sole class body itself may end up with its method if it has no
        docstring. But the body of \
        a common def function can only be ended up by a none-indented def/class

    Returns:
        string : the extracted comment body, inclusive of its quote marks.

    """

    i = start_from
    fcombody = ""  # def comment body
    comstart = -1  # the starting line index of comment mark "'''" or """"""
    # if it is not -1, it indicates the loop is in the comment body
    comstyle = 0  # comment mark style ,comments quoted with ''' is coded as 1
    # comments quoted with """ is coded as 2
    for x in range(i + 1, len(srcls)):
        if is_class_begin:
            if srcls[x].replace('\t', '    ').startswith('    def '):
                break
        if srcls[x].startswith('def ') or srcls[x].startswith('class '):
            break
        else:
            if comstart == -1:
                s = srcls[x].replace(" ", '').replace("\t",
                                                      '').replace("\n", '')
                if s.startswith("\"\"\"") or s.startswith("r\"\"\""):
                    comstart = x
                    comstyle = 2
                    continue
            if (comstyle == 2 and comstart != -1 and
                    srcls[x].replace(" ", '').replace("\t", '').replace(
                        "\n", '').startswith("\"\"\"")):
                break
            if comstart == -1:
                s = srcls[x].replace(" ", '').replace("\t",
                                                      '').replace("\n", '')
                if s.startswith("\'\'\'") or s.startswith("r\'\'\'"):
                    comstart = x
                    comstyle = 1
                    continue
            if (comstyle == 1 and comstart != -1 and
                    srcls[x].replace(" ", '').replace("\t", '').replace(
                        "\n", '').startswith("\'\'\'")):
                break
            if (comstart !=
                    -1):  # when the comments start, begin to add line to fcombody
                fcombody += srcls[x]
    return fcombody


def srccoms_extract(srcfile, wlist, methods):
    """
    Given a source file ``srcfile``, this function will
    extract its API(doc comments) and run sample codes in the
    API.

    Args:
        srcfile(file): the source file
        wlist(list): white list
        methods(list): only elements of this list considered.

    Returns:
        result: True or False
        error_methods: the methods that failed.
    """

    process_result = True
    error_methods = []
    srcc = srcfile.read()
    # 2. get defs and classes header line number
    # set file pointer to its beginning
    srcfile.seek(0, 0)
    srcls = srcfile.readlines()  # source lines

    # 1. fetch__all__ list
    allidx = srcc.find("__all__")
    logger.debug('processing %s, methods: %s', srcfile.name, str(methods))
    srcfile_new, _ = os.path.splitext(srcfile.name)
    srcfile_list = srcfile_new.split('/')
    srcfile_str = ''
    for i in range(4, len(srcfile_list)):
        srcfile_str = srcfile_str + srcfile_list[i] + '.'
    if allidx != -1:
        alllist = []
        # get all list for layers/ops.py
        if srcfile.name.find("ops.py") != -1:
            for ai in range(0, len(srcls)):
                if srcls[ai].startswith("__all__"):
                    lb = srcls[ai].find('[')
                    rb = srcls[ai].find(']')
                    if lb == -1:
                        continue
                    allele = srcls[ai][lb + 1:rb].replace("'", '').replace(
                        " ", '').replace("\"", '')
                    alllist.append(allele)
            if '' in alllist:
                alllist.remove('')
        else:
            alllist_b = allidx + len("__all__")
            allstr = srcc[alllist_b + srcc[alllist_b:].find("[") + 1:alllist_b +
                          srcc[alllist_b:].find("]")]
            allstr = allstr.replace("\n", '').replace(" ", '').replace(
                "'", '').replace("\"", '')
            alllist = allstr.split(',')
            if '' in alllist:
                alllist.remove('')
        api_alllist_count = len(alllist)
        logger.debug('found %d items: %s', api_alllist_count, str(alllist))
        api_count = 0
        handled = []
        # get src contents in layers/ops.py
        if srcfile.name.find("ops.py") != -1:
            for i in range(0, len(srcls)):
                opname = None
                opres = re.match(r"^(\w+)\.__doc__", srcls[i])
                if opres is not None:
                    opname = opres.group(1)
                else:
                    opres = re.match(
                        r"^add_sample_code\(globals\(\)\[\"(\w+)\"\]", srcls[i])
                    if opres is not None:
                        opname = opres.group(1)
                if opname is not None:
                    if opname in wlist:
                        logger.info('%s is in the whitelist, skip it.', opname)
                        continue
                    else:
                        logger.debug('%s\'s docstring found.', opname)
                    comstart = i
                    for j in range(i, len(srcls)):
                        if srcls[j].find("\"\"\"") != -1:
                            comstart = i
                    opcom = ""
                    for j in range(comstart + 1, len(srcls)):
                        opcom += srcls[j]
                        if srcls[j].find("\"\"\"") != -1:
                            break
                    result, _, _ = sampcd_extract_and_run(opcom, opname, "def",
                                                          opname)
                    if not result:
                        error_methods.append(opname)
                        process_result = False
                    api_count += 1
                    handled.append(
                        opname)  # ops.py also has normal formatted functions
                    # use list 'handled'  to mark the functions have been handled here
                    # which will be ignored in the following step
                    # handled what?
        logger.debug('%s already handled.', str(handled))
        for i in range(0, len(srcls)):
            if srcls[i].startswith(
                    'def '):  # a function header is detected in line i
                f_header = srcls[i].replace(" ", '')
                fn = f_header[len('def'):f_header.find('(')]  # function name
                if "%s%s" % (srcfile_str, fn) not in methods:
                    logger.info(
                        '[file:%s, function:%s] not in methods list, skip it.',
                        srcfile_str, fn)
                    continue
                if fn in handled:
                    continue
                if fn in alllist:
                    api_count += 1
                    if fn in wlist or fn + "@" + srcfile.name in wlist:
                        logger.info('[file:%s, function:%s] skip by wlist.',
                                    srcfile_str, fn)
                        continue
                    fcombody = single_defcom_extract(i, srcls)
                    if fcombody == "":  # if no comment
                        print("def name:", fn)
                        print("-----------------------")
                        print("WARNING: no comments in function ", fn,
                              ", but it deserves.")
                        continue
                    else:
                        result, _, _ = sampcd_extract_and_run(fcombody, fn,
                                                              "def", fn)
                        if not result:
                            error_methods.append(fn)
                            process_result = False

            if srcls[i].startswith('class '):
                c_header = srcls[i].replace(" ", '')
                cn = c_header[len('class'):c_header.find('(')]  # class name
                if '%s%s' % (srcfile_str, cn) not in methods:
                    logger.info(
                        '[file:%s, class:%s] not in methods list, skip it.',
                        srcfile_str, cn)
                    continue
                if cn in handled:
                    continue
                if cn in alllist:
                    api_count += 1
                    if cn in wlist or cn + "@" + srcfile.name in wlist:
                        logger.info('[file:%s, class:%s] skip by wlist.',
                                    srcfile_str, cn)
                        continue
                    # class comment
                    classcom = single_defcom_extract(i, srcls, True)
                    if classcom != "":
                        result, _, _ = sampcd_extract_and_run(classcom, cn,
                                                              "class", cn)
                        if not result:
                            error_methods.append(cn)
                            process_result = False
                    else:
                        print("WARNING: no comments in class itself ", cn,
                              ", but it deserves.\n")
                    # handling methods in class bodies
                    for x in range(
                            i + 1,
                            len(srcls)):  # from the next line of class header
                        if (srcls[x].startswith('def ') or
                                srcls[x].startswith('class ')):
                            break
                        else:
                            # member method def header
                            srcls[x] = srcls[x].replace('\t', '    ')
                            if (srcls[x].startswith(
                                    '    def ')):  # detect a mehtod header..
                                thisl = srcls[x]
                                indent = len(thisl) - len(thisl.lstrip())
                                mn = thisl[indent + len('def '):thisl.find(
                                    '(')]  # method name
                                name = cn + "." + mn  # full name
                                if '%s%s' % (
                                        srcfile_str, name
                                ) not in methods:  # class method not in api.spec 
                                    logger.info(
                                        '[file:%s, func:%s] not in methods, skip it.',
                                        srcfile_str, name)
                                    continue
                                if mn.startswith('_'):
                                    logger.info(
                                        '[file:%s, func:%s] startswith _, it\'s private method, skip it.',
                                        srcfile_str, name)
                                    continue
                                if name in wlist or name + "@" + srcfile.name in wlist:
                                    logger.info(
                                        '[file:%s, class:%s] skip by wlist.',
                                        srcfile_str, name)
                                    continue
                                thismethod = [thisl[indent:]
                                              ]  # method body lines
                                # get all the lines of a single method body
                                # into thismethod(list)
                                # and send it to single_defcom_extract
                                for y in range(x + 1, len(srcls)):
                                    srcls[y] = srcls[y].replace('\t', '    ')
                                    if (srcls[y].startswith('def ') or
                                            srcls[y].startswith('class ')):
                                        # end of method
                                        break
                                    elif srcls[y].startswith('    def '):
                                        # end of method
                                        break
                                    else:
                                        thismethod.append(srcls[y][indent:])
                                thismtdcom = single_defcom_extract(0,
                                                                   thismethod)
                                if thismtdcom != "":
                                    result, _, _ = sampcd_extract_and_run(
                                        thismtdcom, name, "method", name)
                                    if not result:
                                        error_methods.append(name)
                                        process_result = False
    else:
        logger.warning('__all__ not found in file:%s', srcfile.name)

    return process_result, error_methods


def test(file_list):
    global methods  # readonly
    process_result = True
    for file in file_list:
        with open(file, 'r') as src:
            if not srccoms_extract(src, wlist, methods):
                process_result = False
    return process_result


def run_a_test(tc_filename):
    """
    execute a sample code-block.
    """
    global methods  # readonly
    process_result = True
    with open(tc_filename, 'r') as src:
        process_result, error_methods = srccoms_extract(src, wlist, methods)
    return process_result, tc_filename, error_methods


def get_filenames():
    '''
    this function will get the modules that pending for check.

    Returns:

        list: the modules pending for check .

    '''
    filenames = []
    global methods  # write
    global whl_error
    methods = []
    whl_error = []
    get_incrementapi()
    API_spec = API_DIFF_SPEC_FN
    with open(API_spec) as f:
        for line in f.readlines():
            api = line.replace('\n', '')
            try:
                module = eval(api).__module__
            except AttributeError:
                whl_error.append(api)
                continue
            except SyntaxError:
                logger.warning('line:%s, api:%s', line, api)
                # paddle.Tensor.<lambda>
                continue
            if len(module.split('.')) > 1:
                filename = '../python/'
                # work for .so?
                module_py = '%s.py' % module.split('.')[-1]
                for i in range(0, len(module.split('.')) - 1):
                    filename = filename + '%s/' % module.split('.')[i]
                filename = filename + module_py
            else:
                filename = ''
                logger.warning("WARNING: Exception in getting api:%s module:%s",
                               api, module)
            if filename in filenames:
                continue
            elif not filename:
                logger.warning('filename invalid: %s', line)
                continue
            elif not os.path.exists(filename):
                logger.warning('file not exists: %s', filename)
                continue
            else:
                filenames.append(filename)
            # get all methods
            method = ''
            if inspect.isclass(eval(api)):
                name = api.split('.')[-1]
            elif inspect.isfunction(eval(api)):
                name = api.split('.')[-1]
            elif inspect.ismethod(eval(api)):
                name = '%s.%s' % (api.split('.')[-2], api.split('.')[-1])
            else:
                name = ''
                logger.warning(
                    "WARNING: Exception when getting api:%s, line:%s", api,
                    line)
            for j in range(2, len(module.split('.'))):
                method = method + '%s.' % module.split('.')[j]
            method = method + name
            if method not in methods:
                methods.append(method)
    os.remove(API_spec)
    return filenames


def get_api_md5(path):
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
    rm_file = []
    for f in filenames:
        for w_file in wlist_file:
            if f.startswith(w_file):
                rm_file.append(f)
                filenames.remove(f)
    if len(rm_file) != 0:
        logger.info("REMOVE white files: %s", rm_file)
    logger.info("API_PR is diff from API_DEV: %s", filenames)

    threads = multiprocessing.cpu_count()
    if args.threads:
        threads = args.threads
    po = multiprocessing.Pool(threads)
    # results = po.map_async(test, divided_file_list)
    results = po.map_async(run_a_test, filenames)
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
