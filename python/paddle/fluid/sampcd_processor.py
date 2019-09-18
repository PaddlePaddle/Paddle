#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 14 14:10:36 2019

@author: haowang101779990
"""
"""
This script is for scraping and executing sample codes in the 
comments of paddle .py source file in order to validate the 
sample codes.

Put this script at directory fluid/

log July 4 : CPU is implemented, wlist is added,
transpiler module need to be finished

"""

import os
import subprocess


def find_all(srcstr, substr):
    indices = []
    gotone = srcstr.find(substr)
    while (gotone != -1):
        indices.append(gotone)
        gotone = srcstr.find(substr, gotone + 1)
    return indices


def check_indent(cdline):
    indent = 0
    for c in cdline:
        if c == '\t':
            indent += 4
        elif c == ' ':
            indent += 1
        if c != ' ' and c != '\t':
            break
    return indent


#srccom: raw comments in the source,including ''' and original indent
def sampcd_extract_and_run(srccom, name, logf):
    sampcd_begins = find_all(srccom, ".. code-block:: python")
    #no sample code
    #have sample code but not formatted by code block 

    status = []
    '''
    status:

    3:error sample code
    2:have sample code but format is wrong
    1:no sample code
    0：successful
    -1:no comments found 
    -2:in white list
    there may be several examples in a source comment
    so status is a list to contain the states
    '''

    if (len(sampcd_begins) == 0):
        if (srccom.find("Examples:") != -1):
            print "----example code check----\n"
            logf.write("\n----example code check----\n")
            if (srccom.find(">>>") != -1):
                logf.write(
                    "Deprecated sample code style:\n\n    Examples:\n\n        >>>codeline\n        >>>codeline\n\n\n "
                    + "Please use '.. code-block:: python' to " +
                    "format sample code.\n")
                print(
                    "Deprecated sample code style:\n\n    Examples:\n\n        >>>codeline\n        >>>codeline\n\n\n "
                    + "Please use '.. code-block:: python' to " +
                    "format sample code.\n")
                status.append(2)
        else:
            print "No sample code!\n"
            logf.write("\nNo sample code!\n")
            status.append(1)

    for y in range(1, len(sampcd_begins) + 1):
        sampcd_begin = sampcd_begins[y - 1]
        sampcd = srccom[sampcd_begin + len(".. code-block:: python") + 1:]
        sampcd = sampcd.split("\n")
        #remove starting empty lines
        while sampcd[0].replace(' ', '').replace('\t', '') == '':
            sampcd.pop(0)
        min_indent = check_indent(sampcd[0])
        sampcd_to_write = []
        for i in range(0, len(sampcd)):
            cdline = sampcd[i]
            #handle empty lines or those only with spaces/tabs
            if cdline.strip() == '':
                continue
            this_indent = check_indent(cdline)
            if (this_indent < min_indent):
                break
            else:
                cdline = cdline.replace('\t', '    ')
                sampcd_to_write.append(cdline[min_indent:])
        sampcd = '\n'.join(sampcd_to_write)
        sampcd = '\nimport os\n' + 'os.environ["CUDA_VISIBLE_DEVICES"] = ""\n' + sampcd
        sampcd += '\nprint ' + '\"' + name + ' sample code is executed successfully!\"\n'

        print "\n"
        print "Sample code " + str(y) + " extracted for " + name + "   :"
        print "^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^"
        print(sampcd)

        logf.write("\nSample code extracted for " + name + "   :\n")
        logf.write("\n" + sampcd + "\n")

        print "----example code check----\n"
        print "executing sample code ....."

        logf.write("\n----example code check----\n")
        logf.write("\nexecuting sample code .....\n")

        if (len(sampcd_begins) > 1):
            tfname = name + "_example_" + str(y) + ".py"
        else:
            tfname = name + "_example" + ".py"

        tempf = open("samplecode_temp/" + tfname, 'w')
        tempf.write(sampcd)
        tempf.close()
        cmd = ["python", "samplecode_temp/" + tfname]
        subprc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        output = subprc.communicate()
        print "execution result:"
        logf.write("\nexecution result:\n")
        msg = "\n".join(output)

        if (msg.find("sample code is executed successfully!") == -1):
            print("Error Raised from Sample Code " + name + " :\n")
            logf.write("\nError Raised from Sample Code " + name + " :\n")
            status.append(3)
        else:
            status.append(0)

        #msg is the returned code execution report
        print msg
        logf.write("\n" + msg + "\n")
        os.remove("samplecode_temp/" + tfname)

    print status
    logf.write("\n" + "execution status" + str(status) + "\n")
    return status


'''
to extract a def function/class comments body
start_from: the line num of "def" header
'''


def single_defcom_extract(start_from, srcls, is_class_begin=False):
    i = start_from
    fcombody = ""  #def comment body
    comstart = -1
    comstyle = 0

    for x in range(i + 1, len(srcls)):
        if is_class_begin:
            if (srcls[x].startswith('    def ')):
                break
        if ((srcls[x].startswith('def ') or srcls[x].startswith('class '))):
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
                    -1):  #when the comments start, begin to add line to fcombody
                fcombody += srcls[x]
    return fcombody


def print_header(logf, htype, name):
    print "\n"
    print htype + " name:" + name
    print "-----------------------"
    logf.write("\n\n" + htype + " name:" + name + "\n")
    logf.write("-----------------------\n")


def srccoms_extract(srcfile, logf, status_all, wlist):
    print "source file name:" + srcfile.name
    print "---------------------------------------------------"

    logf.write("source file name:" + srcfile.name + "\n")
    logf.write("---------------------------------------------------\n\n")

    srcc = srcfile.read()

    #2. get defs and classes header line number
    #set file pointer to its beginning
    srcfile.seek(0, 0)
    srcls = srcfile.readlines()  #source lines

    #1. fetch__all__ list
    allidx = srcc.find("__all__")

    if (allidx != -1):
        alllist = []
        if (srcfile.name.find("ops.py") != -1):
            for ai in range(0, len(srcls)):
                if (srcls[ai].startswith("__all__")):
                    lb = srcls[ai].find('[')
                    rb = srcls[ai].find(']')
                    if (lb == -1):
                        continue
                    allele = srcls[ai][lb + 1:rb].replace("'", '').replace(
                        " ", '').replace("\"", '')
                    alllist.append(allele)
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
            print "__all__:" + str(alllist) + "\n"
            logf.write("__all__:" + str(alllist) + "\n\n")
        api_alllist_count = len(alllist)
        api_count = 0
        handled = []
        if (srcfile.name.find("ops.py") != -1):
            for i in range(0, len(srcls)):
                if srcls[i].find("__doc__") != -1:
                    opname = srcls[i][:srcls[i].find("__doc__") - 1]
                    print_header(logf, "def", opname)
                    if opname in wlist:
                        print opname + " is in white list, thus skipped"
                        logf.write("\n" + opname +
                                   " is in white list, thus skipped\n")
                        status_all[opname] = [-2]
                        print status_all[opname]
                        logf.write("\n" + "execution status" + str(status_all[
                            opname]) + "\n")
                        continue
                    comstart = i
                    for j in range(i, len(srcls)):
                        if (srcls[j].find("\"\"\"") != -1):
                            comstart = i
                    opcom = ""
                    for j in range(comstart + 1, len(srcls)):
                        opcom += srcls[j]
                        if (srcls[j].find("\"\"\"") != -1):
                            break
                    if opname in wlist:
                        print opname + " is in white list, thus skipped"
                        logf.write("\n" + opname +
                                   " is in white list, thus skipped\n")
                        status_all[opname] = [-2]
                        print status_all[opname]
                        logf.write("\n" + "execution status" + str(status_all[
                            opname]) + "\n")
                        continue
                    status = sampcd_extract_and_run(opcom, opname, logf)
                    api_count += 1
                    status_all[opname] = status
                    handled.append(opname)

        for i in range(0, len(srcls)):
            if srcls[i].startswith('def '):
                f_header = srcls[i].replace(" ", '')
                fn = f_header[len('def'):f_header.find('(')]  #function name
                if fn in handled:
                    continue
                print_header(logf, "def", fn)
                if fn in alllist:
                    api_count += 1
                    if fn in wlist:
                        print fn + " is in white list, thus skipped"
                        logf.write("\n" + fn +
                                   " is in white list, thus skipped\n")
                        status_all[fn] = [-2]
                        print status_all[fn]
                        logf.write("\n" + "execution status" + str(status_all[
                            fn]) + "\n")
                        continue
                    fcombody = single_defcom_extract(i, srcls)
                    if (fcombody == ""):
                        print "no comments in function " + fn
                        logf.write("no comments in function " + fn + "\n\n")
                        status_all[fn] = [-1]
                        print status_all[fn]
                        logf.write("\n" + "execution status" + str(status_all[
                            fn]) + "\n")
                        continue
                    else:
                        status = sampcd_extract_and_run(fcombody, fn, logf)
                        status_all[fn] = status
                else:
                    print fn + " not in __all__ list"
                    logf.write(fn + " not in __all__ list\n\n")
            if srcls[i].startswith('class '):
                print srcls[i]
                c_header = srcls[i].replace(" ", '')
                cn = c_header[len('class'):c_header.find('(')]  #function name
                if cn in handled:
                    continue
                print_header(logf, "class", cn)
                if cn in alllist:
                    api_count += 1
                    if cn in wlist:
                        print cn + " is in white list, thus skipped"
                        logf.write("\n" + cn +
                                   " is in white list, thus skipped\n")
                        status_all[cn] = [-2]
                        print status_all[cn]
                        logf.write("\n" + "execution status" + str(status_all[
                            cn]) + "\n")
                        continue
                    allcoms = []
                    classcom = single_defcom_extract(i, srcls, True)
                    allcoms.append(classcom)
                    if (classcom != ""):
                        status = sampcd_extract_and_run(classcom, cn, logf)
                        status_all[cn] = status
                    else:
                        print "no comments in class itself " + cn + "\n"
                        logf.write("no comments in class itself " + cn +
                                   "\n\n\n")
                        status_all[cn] = [-1]
                        print status_all[cn]
                        logf.write("\n" + "execution status" + str(status_all[
                            cn]) + "\n")
                    for x in range(
                            i + 1,
                            len(srcls)):  #from the next line of class header 
                        if (srcls[x].startswith('def ') or
                                srcls[x].startswith('class ')):
                            break
                        else:
                            if (srcls[x].startswith(
                                    '    def ')):  #detect a mehtod header..
                                thisl = srcls[x]
                                indent = len(thisl) - len(thisl.lstrip())
                                mn = thisl[indent + len('def '):thisl.find(
                                    '(')]  #method name
                                name = cn + "." + mn
                                print_header(logf, "method", name)
                                if mn.startswith('_'):
                                    print mn + "is hidden, not visible to users"
                                    logf.write(
                                        "\n" + mn +
                                        "is hidden, not visible to users\n")
                                    continue
                                if name in wlist:
                                    print name + " is in white list, thus skipped"
                                    logf.write(
                                        "\n" + name +
                                        " is in white list, thus skipped\n")
                                    status_all[name] = [-2]
                                    print status_all[name]
                                    logf.write("\n" + "execution status" + str(
                                        status_all[name]) + "\n")
                                    continue
                                thismethod = []
                                thismtdstr = ""
                                thismethod.append(thisl[indent:])
                                thismtdstr += thisl[indent:]
                                for y in range(x + 1, len(srcls)):
                                    if (srcls[y].startswith('def ') or
                                            srcls[y].startswith('class ')):
                                        break
                                    elif (srcls[y].lstrip().startswith('def ')):
                                        break
                                    else:
                                        thismethod.append(srcls[y][indent:])
                                        thismtdstr += srcls[y][indent:]
                                thismtdcom = single_defcom_extract(0,
                                                                   thismethod)
                                allcoms.append(thismtdcom)
                                if (thismtdcom != ""):
                                    status = sampcd_extract_and_run(thismtdcom,
                                                                    name, logf)
                                    status_all[name] = status
                                else:
                                    print "no comments in method " + name + "\n"
                                    logf.write("no comments in method " + name +
                                               "\n\n\n")
                                    status_all[name] = [-1]
                                    print status_all[name]
                                    logf.write("\n" + "execution status" + str(
                                        status_all[name]) + "\n")
                else:
                    print cn + " is not in __all__ list"
                    logf.write(cn + " is not in __all__ list\n\n")
    return [
        srcfile.name + " all list length: " + str(api_alllist_count),
        "analysed api count: " + str(api_count)
    ]


filenames = [
    "layers/control_flow.py", "layers/io.py", "layers/nn.py", "layers/ops.py",
    "layers/tensor.py", "layers/learning_rate_scheduler.py",
    "layers/detection.py", "layers/metric_op.py"
]
filenames += [
    "dygraph/layers.py", "dygraph/base.py", "dygraph/nn.py",
    "dygraph/tracer.py", "dygraph/profiler.py", "dygraph/parallel.py",
    "dygraph/checkpoint.py", "dygraph/learning_rate_scheduler.py",
    "dygraph/backward_strategy.py"
]

filenames += [
    "data_feeder.py", "dataset.py", "clip.py", "metrics.py", "executor.py",
    "initializer.py", "io.py", "nets.py", "optimizer.py", "profiler.py",
    "regularizer.py", "backward.py", "average.py", "profiler.py",
    "unique_name.py"
]

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
    "ExecutionStrategy.num_threads", "CompiledProgram.with_inference_optimize",
    "CompositeMetric.add_metric", "CompositeMetric.update",
    "CompositeMetric.eval", "DetectionMAP.get_map_var", "MetricBase",
    "MetricBase.reset", "MetricBase.get_config", "MetricBase.update",
    "MetricBase.eval", "Accuracy.eval", "Auc.update", "Auc.eval",
    "EditDistance.update", "EditDistance.eval",
    "ExponentialMovingAverage.apply", "ExponentialMovingAverage.restore",
    "ExponentialMovingAverage.update", "StaticRNN.step", "StaticRNN.step_input",
    "StaticRNN.step_output", "StaticRNN.update_memory", "DetectionMAP.reset",
    'StaticRNN.output'
]

wlist_temp = [
    'elementwise_floordiv', 'Layer', 'Layer.create_parameter',
    'Layer.create_variable', 'Layer.sublayers', 'Layer.add_parameter',
    'Layer.add_sublayer', 'Layer.parameters', 'Tracer', 'Layer.full_name',
    'InMemoryDataset', 'layer_norm', 'bipartite_match', 'double_buffer',
    'cumsum', 'thresholded_relu', 'group_norm', 'random_crop', 'py_func',
    'row_conv', 'hard_shrink', 'ssd_loss', 'retinanet_target_assign',
    'InMemoryDataset.global_shuffle', 'InMemoryDataset.get_memory_data_size',
    'DetectionMAP', 'hash', 'InMemoryDataset.set_queue_num', 'LayerNorm',
    'Preprocessor', 'chunk_eval', 'GRUUnit', 'ExponentialMovingAverage',
    'QueueDataset.global_shuffle', 'NumpyArrayInitializer',
    'create_py_reader_by_data', 'InMemoryDataset.local_shuffle',
    'InMemoryDataset.get_shuffle_data_size', 'size', 'edit_distance', 'nce',
    'BilinearInitializer', 'NaturalExpDecay', 'noam_decay',
    'retinanet_detection_output', 'Pool2D', 'PipelineOptimizer',
    'generate_mask_labels', 'isfinite',
    'InMemoryDataset.set_fleet_send_batch_size', 'cuda_profiler', 'unfold',
    'Executor', 'InMemoryDataset.load_into_memory', 'ExponentialDecay',
    'BatchNorm', 'deformable_conv', 'InMemoryDataset.preload_into_memory',
    'py_reader', 'linear_lr_warmup', 'InMemoryDataset.wait_preload_done',
    'CosineDecay', 'roi_perspective_transform', 'unique', 'ones_like',
    'LambOptimizer', 'InMemoryDataset.release_memory', 'Conv2DTranspose',
    'QueueDataset.local_shuffle'
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
    'Embedding.forward', 'Recall.eval', 'FC.forward', 'While.block'
]

wlist = wlist_temp + wlist_inneed + wlist_ignore
status_all = {}
logf = open("log.txt", 'w')
statusf = open("status.txt", 'w')

if not os.path.isdir("./samplecode_temp"):
    os.mkdir("./samplecode_temp")
for filename in filenames:
    srcfile = open(filename, 'r')
    counts = srccoms_extract(srcfile, logf, status_all, wlist)
    logf.write("\n\n" + str(counts) + "\n\n")
    srcfile.close()
for root, dirs, files in os.walk("./samplecode_temp"):
    for fntemp in files:
        os.remove("./samplecode_temp/" + fntemp)

os.rmdir("./samplecode_temp")
statusf.write("status_all:\n" + str(status_all))
status_groups = {-2: [], -1: [], 0: [], 1: [], 2: [], 3: []}
ci_pass = True

for key in status_all:
    statusl = status_all[key]
    for ele in statusl:
        if (ele != 0 and ele != -2):
            ci_pass = False
            break
    if len(statusl) == 1:
        status_groups[statusl[0]].append(key)
    else:
        for u in range(0, len(statusl)):
            status_groups[statusl[u]].append(key + '_' + str(u + 1))

statusf.write('\n\ngrouped apis:\n' + str(status_groups) + '\n')
statusf.close()
logf.close()

temp_wlistf = open("tempwlist.txt", 'w')
wlist_temp = status_groups[1] + status_groups[2] + status_groups[
    3] + status_groups[-1]
temp_wlistf.write(str(wlist_temp))
temp_wlistf.close()
print str(wlist_temp)

if not ci_pass:
    print "Mistakes found in sample codes, refer to the log for details"
    exit(1)
else:
    print "Sample code check is successful!"
