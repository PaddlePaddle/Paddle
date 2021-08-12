# Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
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
Print all signature of a python module in alphabet order.

Usage:
    ./print_signature  "paddle.fluid" > signature.txt
"""

import inspect
import collections
import sys
import hashlib
import pkgutil
import logging
import argparse
import re

member_dict = collections.OrderedDict()

visited_modules = set()

logger = logging.getLogger()
if logger.handlers:
    # we assume the first handler is the one we want to configure
    console = logger.handlers[0]
else:
    console = logging.StreamHandler(sys.stderr)
    logger.addHandler(console)
console.setFormatter(
    logging.Formatter(
        "%(asctime)s - %(funcName)s:%(lineno)d - %(levelname)s - %(message)s"))


def md5(doc):
    try:
        hashinst = hashlib.md5()
        hashinst.update(str(doc).encode('utf-8'))
        md5sum = hashinst.hexdigest()
    except UnicodeDecodeError as e:
        md5sum = None
        print(
            "Error({}) occurred when `md5({})`, discard it.".format(
                str(e), doc),
            file=sys.stderr)

    return md5sum


def is_primitive(instance):
    int_types = (int, )
    pritimitive_types = int_types + (float, str)
    if isinstance(instance, pritimitive_types):
        return True
    elif isinstance(instance, (list, tuple, set)):
        for obj in instance:
            if not is_primitive(obj):
                return False

        return True
    else:
        return False


arg_pat = re.compile(r'^\s*(\*{0,2}\s*\w*)\b\s*\([^\)]*\)\s*:.*')
arg_pat_no_type_desc = re.compile(r'^\s*(\*{0,2}\s*\w*)\b\s*:.*')


class ArgsDescResult:
    def __init__(self) -> None:
        self.args = []
        self.varargs = None
        self.varkw = None
        self.defaults = None
        self.kwonlyargs = []
        self.kwonlydefaults = None
        self.annotations = {}
        self.text = None


def extract_args_desc_from_docstr(docstr):
    docstr = inspect.cleandoc(docstr.replace("\t", '    '))
    lines = docstr.split('\n')
    args_started = False
    indent = None
    args_desc_lines = []
    args_desc_lines_indent = []
    find_indent_pat = re.compile(r'\S')
    for line in lines:
        if not args_started:
            if line.strip(
            ) in ['Args:', 'Parameters:', 'Parameter:', 'Params:', 'Param:']:
                args_started = True
                indent = find_indent_pat.search(line).start()
        else:
            mo = find_indent_pat.search(line)
            if mo:
                if mo.start() > indent:
                    args_desc_lines.append(line)
                    args_desc_lines_indent.append(mo.start())
                else:
                    break
    args_line_merged = []
    if args_desc_lines:
        for i, line in enumerate(args_desc_lines):
            if args_desc_lines_indent[i] == args_desc_lines_indent[0]:
                args_line_merged.append(line)
            else:
                args_line_merged[-1] += ' ' + line.strip()
    res = ArgsDescResult()
    res.text = '\n'.join(args_desc_lines)
    for line in args_line_merged:
        mo = arg_pat.search(line)
        if not mo:
            logger.debug('try another pattern')
            mo = arg_pat_no_type_desc.search(line)
        if mo:
            arg_name = mo.group(1)
            if arg_name.startswith('**'):
                res.varkw = arg_name[2:].strip()
            elif arg_name.startswith('*'):
                res.varargs = arg_name[1:].strip()
            else:
                res.args.append(arg_name)
        else:
            logger.error(f'arg desc not found in line: {line}')
    return res


def check_api_args_doc(fullargspec, docstr):
    """
    check the name and order of the args.

    The Default value has so many formats.

    Args:
        fullargspec: the inpect.FullArgSpec object
        docstr: the docstring of the function
    
    Returns:
        boolean, check result
    """
    #document = new_document('~', parser_settings)
    #parser.parse(docstr, document)
    #v = MyArgsVisitor(document)
    #document.walkabout(v)

    v = extract_args_desc_from_docstr(docstr)

    if v.args != fullargspec.args and not (fullargspec.args[0] == 'self' and
                                           v.args == fullargspec.args[1:]):
        logger.error(f'v.text={v.text}')
        logger.error('v.args is %s, not equal to FullArgSpec.args %s',
                     str(v.args), str(fullargspec.args))
        return False
    if v.varargs != fullargspec.varargs:
        logger.error(f'v.text={v.text}')
        logger.error('v.varargs is %s, not equal to FullArgSpec.varargs %s',
                     str(v.varargs), str(fullargspec.varargs))
        return False
    if v.varkw != fullargspec.varkw:
        logger.error(f'v.text={v.text}')
        logger.error('v.varkw is %s, not equal to FullArgSpec.varkw %s',
                     str(v.varkw), str(fullargspec.varkw))
        return False

    return True


ErrorSet = set()
IdSet = set()
skiplist = []


def visit_all_module(mod):
    mod_name = mod.__name__
    if mod_name != 'paddle' and not mod_name.startswith('paddle.'):
        return

    if mod_name.startswith('paddle.fluid.core'):
        return

    if mod in visited_modules:
        return
    visited_modules.add(mod)

    member_names = dir(mod)
    if hasattr(mod, "__all__"):
        member_names += mod.__all__
    for member_name in member_names:
        if member_name.startswith('_'):
            continue
        cur_name = mod_name + '.' + member_name
        if cur_name in skiplist:
            continue
        try:
            instance = getattr(mod, member_name)
            if inspect.ismodule(instance):
                visit_all_module(instance)
            else:
                instance_id = id(instance)
                if instance_id in IdSet:
                    continue
                IdSet.add(instance_id)
                if hasattr(instance,
                           '__name__') and member_name != instance.__name__:
                    print(
                        "Found alias API, alias name is: {}, original name is: {}".
                        format(member_name, instance.__name__),
                        file=sys.stderr)
        except:
            if not cur_name in ErrorSet and not cur_name in skiplist:
                ErrorSet.add(cur_name)


# all from gen_doc.py
api_info_dict = {}  # used by get_all_api


# step 1: walkthrough the paddle package to collect all the apis in api_set
def get_all_api(root_path='paddle', attr="__all__"):
    """
    walk through the paddle package to collect all the apis.
    """
    import paddle
    global api_info_dict
    api_counter = 0
    for filefinder, name, ispkg in pkgutil.walk_packages(
            path=paddle.__path__, prefix=paddle.__name__ + '.'):
        try:
            if name in sys.modules:
                m = sys.modules[name]
            else:
                # importlib.import_module(name)
                m = eval(name)
                continue
        except AttributeError:
            logger.warning("AttributeError occurred when `eval(%s)`", name)
            pass
        else:
            api_counter += process_module(m, attr)

    api_counter += process_module(paddle, attr)

    logger.info('%s: collected %d apis, %d distinct apis.', attr, api_counter,
                len(api_info_dict))

    return [(sorted(list(api_info['all_names']))[0], md5(api_info['docstring']))
            for api_info in api_info_dict.values()]


def insert_api_into_dict(full_name, gen_doc_anno=None):
    """
    insert add api into the api_info_dict
    Return:
        api_info object or None
    """
    import paddle
    try:
        obj = eval(full_name)
        fc_id = id(obj)
    except AttributeError:
        logger.warning("AttributeError occurred when `id(eval(%s))`", full_name)
        return None
    except Exception as e:
        logger.warning("Exception(%s) occurred when `id(eval(%s))`",
                       str(e), full_name)
        return None
    else:
        logger.debug("adding %s to api_info_dict.", full_name)
        if fc_id in api_info_dict:
            api_info_dict[fc_id]["all_names"].add(full_name)
        else:
            api_info_dict[fc_id] = {
                "all_names": set([full_name]),
                "id": fc_id,
                "object": obj,
                "type": type(obj).__name__,
                "docstring": '',
            }
            docstr = inspect.getdoc(obj)
            if docstr:
                api_info_dict[fc_id]["docstring"] = inspect.cleandoc(docstr)
            if gen_doc_anno:
                api_info_dict[fc_id]["gen_doc_anno"] = gen_doc_anno
            if inspect.isfunction(obj):
                argspec = inspect.getfullargspec(obj)
                api_info_dict[fc_id]["signature"] = repr(argspec).replace(
                    'FullArgSpec', 'ArgSpec', 1)
                if docstr:
                    api_info_dict[fc_id]["argsdoc_check"] = check_api_args_doc(
                        argspec, api_info_dict[fc_id]["docstring"])
        return api_info_dict[fc_id]


# step 1 fill field : `id` & `all_names`, type, docstring
def process_module(m, attr="__all__"):
    api_counter = 0
    if hasattr(m, attr):
        # may have duplication of api
        for api in set(getattr(m, attr)):
            if api[0] == '_': continue
            # Exception occurred when `id(eval(paddle.dataset.conll05.test, get_dict))`
            if ',' in api: continue

            # api's fullname
            full_name = m.__name__ + "." + api
            api_info = insert_api_into_dict(full_name)
            if api_info is not None:
                api_counter += 1
                if inspect.isclass(api_info['object']):
                    for name, value in inspect.getmembers(api_info['object']):
                        if (not name.startswith("_")) and hasattr(value,
                                                                  '__name__'):
                            method_full_name = full_name + '.' + name  # value.__name__
                            method_api_info = insert_api_into_dict(
                                method_full_name, 'class_method')
                            if method_api_info is not None:
                                api_counter += 1
    return api_counter


def check_public_api():
    import paddle
    modulelist = [  #npqa
        paddle,
        paddle.amp,
        paddle.nn,
        paddle.nn.functional,
        paddle.nn.initializer,
        paddle.nn.utils,
        paddle.static,
        paddle.static.nn,
        paddle.io,
        paddle.jit,
        paddle.metric,
        paddle.distribution,
        paddle.optimizer,
        paddle.optimizer.lr,
        paddle.regularizer,
        paddle.text,
        paddle.utils,
        paddle.utils.download,
        paddle.utils.profiler,
        paddle.utils.cpp_extension,
        paddle.sysconfig,
        paddle.vision,
        paddle.vision.datasets,
        paddle.vision.models,
        paddle.vision.transforms,
        paddle.vision.ops,
        paddle.distributed,
        paddle.distributed.fleet,
        paddle.distributed.fleet.utils,
        paddle.distributed.parallel,
        paddle.distributed.utils,
        paddle.callbacks,
        paddle.hub,
        paddle.autograd,
        paddle.incubate,
        paddle.inference,
        paddle.onnx,
        paddle.device
    ]

    apinum = 0
    alldict = {}
    for module in modulelist:
        if hasattr(module, '__all__'):
            old_all = module.__all__
        else:
            old_all = []
            dirall = dir(module)
            for item in dirall:
                if item.startswith('__'):
                    continue
                old_all.append(item)
        apinum += len(old_all)
        alldict.update({module.__name__: old_all})

    old_all = []
    dirall = dir(paddle.Tensor)
    for item in dirall:
        if item.startswith('_'):
            continue
        old_all.append(item)
    apinum += len(old_all)
    alldict.update({'paddle.Tensor': old_all})

    for module, allapi in alldict.items():
        for member_name in allapi:
            cur_name = module + '.' + member_name
            instance = eval(cur_name)
            doc_md5 = md5(instance.__doc__)
            member_dict[cur_name] = "({}, ('document', '{}'))".format(cur_name,
                                                                      doc_md5)


def check_allmodule_callable():
    import paddle
    modulelist = [paddle]
    for m in modulelist:
        visit_all_module(m)

    return member_dict


def parse_args():
    """
    Parse input arguments
    """
    parser = argparse.ArgumentParser(description='Print Apis Signatures')
    parser.add_argument('--debug', dest='debug', action="store_true")
    parser.add_argument(
        '--method',
        dest='method',
        type=str,
        default='get_all_api',
        help="using get_all_api or from_modulelist")
    parser.add_argument(
        'module', type=str, help='module', default='paddle')  # not used
    parser.add_argument(
        '--skip-api-args-doc-check',
        dest='skip_api_args_doc_check',
        action="store_true")

    if len(sys.argv) == 1:
        args = parser.parse_args(['paddle'])
        return args
    #    parser.print_help()
    #    sys.exit(1)

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    api_args_check_failed = []
    check_allmodule_callable()
    if args.method == 'from_modulelist':
        check_public_api()
        for name in member_dict:
            print(name, member_dict[name])
    elif args.method == 'get_all_api':
        get_all_api()
        all_api_names_to_k = {}
        for k, api_info in api_info_dict.items():
            # 1. the shortest suggested_name may be renamed;
            # 2. some api's fullname is not accessable, the module name of it is overrided by the function with the same name;
            api_name = sorted(list(api_info['all_names']))[0]
            all_api_names_to_k[api_name] = k
        all_api_names_sorted = sorted(all_api_names_to_k.keys())
        for api_name in all_api_names_sorted:
            api_info = api_info_dict[all_api_names_to_k[api_name]]
            print("{0} ({2}, ('document', '{1}'))".format(
                api_name,
                md5(api_info['docstring']), api_info['signature']
                if 'signature' in api_info else 'ArgSpec()'))
            if 'argsdoc_check' in api_info and (not api_info['argsdoc_check']):
                api_args_check_failed.append(api_name)

    exit_value = 0
    if len(ErrorSet):
        exit_value = 1
        for erroritem in ErrorSet:
            print(
                "Error, new function {} is unreachable".format(erroritem),
                file=sys.stderr)
    if (not args.skip_api_args_doc_check) and api_args_check_failed:
        exit_value = 1
        logger.error("some apis' args check failed: %s",
                     str(api_args_check_failed))
    sys.exit(exit_value)
