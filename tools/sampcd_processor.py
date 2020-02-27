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
"""
please make sure to run in the tools path
usage: python sample_test.py {arg1} 
arg1: the first arg defined running in gpu version or cpu version

for example, you can run cpu version python2 testing like this:

    python sampcd_processor.py cpu 

"""


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
    """

    result = True

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
        print_header(htype, hname)
        print("Sample code ", str(y), " extracted for ", name, "   :")
        print(sampcd)
        print("----example code check----\n")
        print("executing sample code .....")
        print("execution result:")

    sampcd_begins = find_all(srccom, " code-block:: python")
    if len(sampcd_begins) == 0:
        print_header(htype, hname)
        '''
        detect sample codes using >>> to format
        and consider this situation as wrong
        '''
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
        if sys.argv[1] == "cpu":
            sampcd = '\nimport os\n' + 'os.environ["CUDA_VISIBLE_DEVICES"] = ""\n' + sampcd
        if sys.argv[1] == "gpu":
            sampcd = '\nimport os\n' + 'os.environ["CUDA_VISIBLE_DEVICES"] = "0"\n' + sampcd
        sampcd += '\nprint(' + '\"' + name + ' sample code is executed successfully!\")'

        if len(sampcd_begins) > 1:
            tfname = name + "_example_" + str(y) + ".py"
        else:
            tfname = name + "_example" + ".py"
        tempf = open("samplecode_temp/" + tfname, 'w')
        tempf.write(sampcd)
        tempf.close()
        if platform.python_version()[0] == "2":
            cmd = ["python", "samplecode_temp/" + tfname]
        elif platform.python_version()[0] == "3":
            cmd = ["python3", "samplecode_temp/" + tfname]
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
            result = False
        # msg is the returned code execution report
        #os.remove("samplecode_temp/" + tfname)

    return result


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
            if (comstart == -1 and srcls[x].replace(" ", '').replace(
                    "\t", '').replace("\n", '').startswith("\"\"\"")):
                comstart = x
                comstyle = 2
                continue
            if (comstyle == 2 and comstart != -1 and
                    srcls[x].replace(" ", '').replace("\t", '').replace(
                        "\n", '').startswith("\"\"\"")):
                break
            if (comstart == -1 and srcls[x].replace(" ", '').replace(
                    "\t", '').replace("\n", '').startswith("\'\'\'")):
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


def print_header(htype, name):
    print(htype, " name:", name)
    print("-----------------------")


def srccoms_extract(srcfile, wlist):
    """
    Given a source file ``srcfile``, this function will
    extract its API(doc comments) and run sample codes in the
    API.

    Args:
        srcfile(file): the source file
        wlist(list): white list

    Returns:
        result: True or False
    """

    process_result = True
    srcc = srcfile.read()
    # 2. get defs and classes header line number
    # set file pointer to its beginning
    srcfile.seek(0, 0)
    srcls = srcfile.readlines()  # source lines

    # 1. fetch__all__ list
    allidx = srcc.find("__all__")
    srcfile_new = srcfile.name
    srcfile_new = srcfile_new.replace('.py', '')
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
        api_count = 0
        handled = []
        # get src contents in layers/ops.py
        if srcfile.name.find("ops.py") != -1:
            for i in range(0, len(srcls)):
                if srcls[i].find("__doc__") != -1:
                    opname = srcls[i][:srcls[i].find("__doc__") - 1]
                    if opname in wlist:
                        continue
                    comstart = i
                    for j in range(i, len(srcls)):
                        if srcls[j].find("\"\"\"") != -1:
                            comstart = i
                    opcom = ""
                    for j in range(comstart + 1, len(srcls)):
                        opcom += srcls[j]
                        if srcls[j].find("\"\"\"") != -1:
                            break
                    api_count += 1
                    handled.append(
                        opname)  # ops.py also has normal formatted functions
                    # use list 'handled'  to mark the functions have been handled here
                    # which will be ignored in the following step
        for i in range(0, len(srcls)):
            if srcls[i].startswith(
                    'def '):  # a function header is detected in line i
                f_header = srcls[i].replace(" ", '')
                fn = f_header[len('def'):f_header.find('(')]  # function name
                if "%s%s" % (srcfile_str, fn) not in methods:
                    continue
                if fn in handled:
                    continue
                if fn in alllist:
                    api_count += 1
                    if fn in wlist or fn + "@" + srcfile.name in wlist:
                        continue
                    fcombody = single_defcom_extract(i, srcls)
                    if fcombody == "":  # if no comment
                        print_header("def", fn)
                        print("WARNING: no comments in function ", fn,
                              ", but it deserves.")
                        continue
                    else:
                        if not sampcd_extract_and_run(fcombody, fn, "def", fn):
                            process_result = False

            if srcls[i].startswith('class '):
                c_header = srcls[i].replace(" ", '')
                cn = c_header[len('class'):c_header.find('(')]  # class name
                if '%s%s' % (srcfile_str, cn) not in methods:
                    continue
                if cn in handled:
                    continue
                if cn in alllist:
                    api_count += 1
                    if cn in wlist or cn + "@" + srcfile.name in wlist:
                        continue
                    # class comment
                    classcom = single_defcom_extract(i, srcls, True)
                    if classcom != "":
                        if not sampcd_extract_and_run(classcom, cn, "class",
                                                      cn):

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
                                    continue
                                if mn.startswith('_'):
                                    continue
                                if name in wlist or name + "@" + srcfile.name in wlist:
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
                                    if not sampcd_extract_and_run(
                                            thismtdcom, name, "method", name):
                                        process_result = False

    return process_result


def test(file_list):
    process_result = True
    for file in file_list:
        with open(file, 'r') as src:
            if not srccoms_extract(src, wlist):
                process_result = False
    return process_result


def get_filenames(path):
    '''
    Given a path ``path``, this function will
    get the modules that pending for check.

    Args:
        path(path): the path of API.spec

    Returns:

        list: the modules pending for check .

    '''
    filenames = []
    global methods
    methods = []
    API_spec = '%s/%s' % (os.path.abspath(os.path.join(os.getcwd(), "..")),
                          path)
    with open(API_spec) as f:
        for line in f.readlines():
            api = line.split(' ', 1)[0]
            try:
                module = eval(api).__module__
            except AttributeError:
                continue
            if len(module.split('.')) > 2:
                filename = '../python/'
                module_py = '%s.py' % module.split('.')[-1]
                for i in range(0, len(module.split('.')) - 1):
                    filename = filename + '%s/' % module.split('.')[i]
                filename = filename + module_py
            else:
                print("\n----Exception in get api filename----\n")
                print("\n" + api + 'module is ' + module + "\n")
            if filename not in filenames:
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
                print("\n----Exception in get api methods----\n")
                print("\n" + line + "\n")
                print("\n" + api + ' method is None!!!' + "\n")
            for j in range(2, len(module.split('.'))):
                method = method + '%s.' % module.split('.')[j]
            method = method + name
            if method not in methods:
                methods.append(method)
    return filenames


'''
Important constant lists:

    wlist : a list of API that should not trigger the example check .
            It is composed of wlist_temp + wlist_inneed + wlist_ignore.
    srcfile: the source .py code file
'''

wlist_inneed = [
    "append_LARS", "BuildStrategy.debug_graphviz_path",
    "BuildStrategy.enable_sequential_execution",
    "BuildStrategy.fuse_elewise_add_act_ops",
    "BuildStrategy.fuse_relu_depthwise_conv",
    "BuildStrategy.gradient_scale_strategy", "BuildStrategy.reduce_strategy",
    "BuildStrategy.remove_unnecessary_lock", "BuildStrategy.sync_batch_norm",
    "DynamicRNN.step_input", "DynamicRNN.static_input", "DynamicRNN.block",
    "DynamicRNN.update_memory", "DynamicRNN.output",
    "transpiler.DistributeTranspilerConfig",
    "transpiler.DistributeTranspilerConfig.slice_var_up",
    "transpiler.DistributeTranspilerConfig.split_method",
    "transpiler.DistributeTranspilerConfig.min_block_size",
    "DistributeTranspilerConfig.slice_var_up",
    "DistributeTranspilerConfig.split_method", "ModelAverage.apply",
    "ModelAverage.restore", "DistributeTranspilerConfig",
    "DistributeTranspilerConfig.min_block_size",
    "ExecutionStrategy.allow_op_delay", "load", "Accuracy.update",
    "ChunkEvaluator.update", "ExecutionStrategy.num_iteration_per_drop_scope",
    "ExecutionStrategy.num_threads", "CompiledProgram._with_inference_optimize",
    "CompositeMetric.add_metric", "CompositeMetric.update",
    "CompositeMetric.eval", "DetectionMAP.get_map_var", "MetricBase",
    "MetricBase.reset", "MetricBase.get_config", "MetricBase.update",
    "MetricBase.eval", "Accuracy.eval", "Auc.update", "Auc.eval",
    "EditDistance.update", "EditDistance.eval",
    "ExponentialMovingAverage.apply", "ExponentialMovingAverage.restore",
    "ExponentialMovingAverage.update", "StaticRNN.step", "StaticRNN.step_input",
    "StaticRNN.step_output", "StaticRNN.update_memory", "DetectionMAP.reset",
    'StaticRNN.output', "cuda_places", "CUDAPinnedPlace", "CUDAPlace",
    "Program.parse_from_string"
]

wlist_nosample = [
    'Compressor', 'Compressor.config', 'Compressor.run', 'run_check',
    'HDFSClient.upload', 'HDFSClient.download', 'HDFSClient.is_exist',
    'HDFSClient.is_dir', 'HDFSClient.delete', 'HDFSClient.rename',
    'HDFSClient.makedirs', 'HDFSClient.ls', 'HDFSClient.lsr', 'multi_download',
    'multi_upload', 'TrainingDecoder.block',
    'QuantizeTranspiler.training_transpile',
    'QuantizeTranspiler.freeze_program', 'AutoMixedPrecisionLists',
    'Uniform.sample', 'Uniform.log_prob', 'Uniform.entropy',
    'Categorical.kl_divergence', 'Categorical.entropy',
    'MultivariateNormalDiag.entropy', 'MultivariateNormalDiag.kl_divergence',
    'RNNCell', 'RNNCell.call', 'RNNCell.get_initial_states', 'GRUCell.call',
    'LSTMCell.call', 'Decoder', 'Decoder.initialize', 'Decoder.step',
    'Decoder.finalize', 'fused_elemwise_activation', 'search_pyramid_hash',
    'convert_dist_to_sparse_program', 'load_persistables_for_increment',
    'load_persistables_for_inference', 'cache', 'buffered', 'xmap_readers'
]

wlist_no_op_pass = ['gelu', 'erf']

wlist_ci_nopass = [
    'DecodeHelper', 'DecodeHelper.initialize', 'DecodeHelper.sample',
    'DecodeHelper.next_inputs', 'TrainingHelper.initialize',
    'TrainingHelper.sample', 'TrainingHelper.next_inputs',
    'GreedyEmbeddingHelper.initialize', 'GreedyEmbeddingHelper.sample',
    'GreedyEmbeddingHelper.next_inputs', 'LayerList.append', 'HDFSClient',
    'InitState', 'TracedLayer', 'SampleEmbeddingHelper.sample',
    'BasicDecoder.initialize', 'BasicDecoder.step', 'ParameterList.append',
    'GreedyEmbeddingHelper', 'SampleEmbeddingHelper', 'BasicDecoder', 'lstm',
    'partial_sum'
]

wlist_nopass = [
    'StateCell', 'StateCell.compute_state', 'TrainingDecoder',
    'TrainingDecoder.step_input', 'TrainingDecoder.static_input',
    'TrainingDecoder.output', 'BeamSearchDecoder', 'GradClipByValue',
    'GradClipByNorm', 'Variable.detach', 'Variable.numpy', 'Variable.set_value',
    'Variable.gradient', 'BeamSearchDecoder.decode',
    'BeamSearchDecoder.read_array', 'CompiledProgram',
    'CompiledProgram.with_data_parallel', 'append_backward', 'guard',
    'to_variable', 'op_freq_statistic', 'save_dygraph', 'load_dygraph',
    'ParallelExecutor', 'ParallelExecutor.run',
    'ParallelExecutor.drop_local_exe_scopes', 'GradClipByGlobalNorm',
    'extend_with_decoupled_weight_decay', 'switch', 'Normal', 'memory_usage',
    'decorate', 'PiecewiseDecay', 'InverseTimeDecay', 'PolynomialDecay',
    'NoamDecay', 'start_profiler', 'profiler', 'tree_conv', 'multiclass_nms2',
    'DataFeedDesc', 'Conv2D', 'Conv3D', 'Conv3DTranspose', 'Embedding', 'NCE',
    'PRelu', 'BilinearTensorProduct', 'GroupNorm', 'SpectralNorm', 'TreeConv',
    'prroi_pool'
]

wlist_temp = [
    'ChunkEvaluator',
    'EditDistance',
    'ErrorClipByValue',
    'Program.clone',
    'cuda_pinned_places',
    'DataFeeder',
    'elementwise_floordiv',
    'Layer',
    'Layer.create_parameter',
    'Layer.create_variable',
    'Layer.sublayers',
    'Layer.add_parameter',
    'Layer.add_sublayer',
    'Layer.parameters',
    'Tracer',
    'Layer.full_name',
    'InMemoryDataset',
    'layer_norm',
    'bipartite_match',
    'double_buffer',
    'cumsum',
    'thresholded_relu',
    'group_norm',
    'random_crop',
    'py_func',
    'row_conv',
    'hard_shrink',
    'ssd_loss',
    'retinanet_target_assign',
    'InMemoryDataset.global_shuffle',
    'InMemoryDataset.get_memory_data_size',
    'DetectionMAP',
    'hash',
    'InMemoryDataset.set_queue_num',
    'LayerNorm',
    'Preprocessor',
    'chunk_eval',
    'GRUUnit',
    'ExponentialMovingAverage',
    'QueueDataset.global_shuffle',
    'NumpyArrayInitializer',
    'create_py_reader_by_data',
    'InMemoryDataset.local_shuffle',
    'InMemoryDataset.get_shuffle_data_size',
    'size',
    'edit_distance',
    'nce',
    'BilinearInitializer',
    'NaturalExpDecay',
    'noam_decay',
    'retinanet_detection_output',
    'Pool2D',
    'PipelineOptimizer',
    'generate_mask_labels',
    'isfinite',
    'InMemoryDataset.set_fleet_send_batch_size',
    'cuda_profiler',
    'unfold',
    'Executor',
    'InMemoryDataset.load_into_memory',
    'ExponentialDecay',
    'BatchNorm',
    'deformable_conv',
    'InMemoryDataset.preload_into_memory',
    'py_reader',
    'linear_lr_warmup',
    'InMemoryDataset.wait_preload_done',
    'CosineDecay',
    'roi_perspective_transform',
    'unique',
    'ones_like',
    'LambOptimizer',
    'InMemoryDataset.release_memory',
    'Conv2DTranspose',
    'QueueDataset.local_shuffle',
    # wrong in dygraph/checkpoint.py  ok in io.py [duplicated name]
    'save_persistables@dygraph/checkpoint.py',
    'load_persistables@dygraph/checkpoint.py'
]
'''
white list of private API/ redundant API
'''
wlist_ignore = [
    'elementwise_pow', 'WeightedAverage.reset', 'ChunkEvaluator.eval',
    'NCE.forward', 'elementwise_div', 'BilinearTensorProduct.forward',
    'NoamDecay.step', 'elementwise_min', 'PiecewiseDecay.step',
    'Conv3DTranspose.forward', 'elementwise_add', 'IfElse.output',
    'IfElse.true_block', 'InverseTimeDecay.step', 'PolynomialDecay.step',
    'Precision.eval', 'enabled', 'elementwise_max', 'stop_gperf_profiler',
    'IfElse.false_block', 'WeightedAverage.add', 'Auc.trapezoid_area',
    'elementwise_mul', 'GroupNorm.forward', 'SpectralNorm.forward',
    'elementwise_sub', 'Switch.case', 'IfElse.input', 'prepare_context',
    'PRelu.forward', 'Recall.update', 'start_gperf_profiler',
    'TreeConv.forward', 'Conv2D.forward', 'Switch.default', 'elementwise_mod',
    'Precision.update', 'WeightedAverage.eval', 'Conv3D.forward',
    'Embedding.forward', 'Recall.eval', 'FC.forward', 'While.block',
    'DGCMomentumOptimizer'
]
# only white on CPU
gpu_not_white = [
    "deformable_conv", "cuda_places", "CUDAPinnedPlace", "CUDAPlace",
    "cuda_profiler", 'DGCMomentumOptimizer'
]

wlist = wlist_temp + wlist_inneed + wlist_ignore + wlist_nosample + wlist_nopass + wlist_no_op_pass + wlist_ci_nopass

if len(sys.argv) < 2:
    print("Error: inadequate number of arguments")
    print('''If you are going to run it on 
        "CPU: >>> python sampcd_processor.py cpu
        "GPU: >>> python sampcd_processor.py gpu
        ''')
    sys.exit("lack arguments")
else:
    if sys.argv[1] == "gpu":
        for _gnw in gpu_not_white:
            wlist.remove(_gnw)
    elif sys.argv[1] != "cpu":
        print("Unrecognized argument:'", sys.argv[1], "' , 'cpu' or 'gpu' is ",
              "desired\n")
        sys.exit("Invalid arguments")
    print("API check -- Example Code")
    print("sample_test running under python", platform.python_version())
    if not os.path.isdir("./samplecode_temp"):
        os.mkdir("./samplecode_temp")
    cpus = multiprocessing.cpu_count()
    filenames = get_filenames('paddle/fluid/API_PR.spec')
    filenames.remove('../python/paddle/fluid/core_avx.py')
    one_part_filenum = int(math.ceil(len(filenames) / cpus))
    divided_file_list = [
        filenames[i:i + one_part_filenum]
        for i in range(0, len(filenames), one_part_filenum)
    ]

    po = multiprocessing.Pool()
    results = po.map_async(test, divided_file_list)
    po.close()
    po.join()

    result = results.get()

    # delete temp files
    for root, dirs, files in os.walk("./samplecode_temp"):
        for fntemp in files:
            os.remove("./samplecode_temp/" + fntemp)
    os.rmdir("./samplecode_temp")

    print("----------------End of the Check--------------------")
    for temp in result:
        if not temp:
            print("Mistakes found in sample codes")
            exit(1)
    print("Sample code check is successful!")
