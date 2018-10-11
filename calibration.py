from __future__ import absolute_import
from __future__ import division
# from __future__ import print_function
import os
import numpy as np
import time
import sys
import paddle
import paddle.fluid as fluid
import models
import reader
import argparse
import functools
from models.learning_rate import cosine_decay
from utility import add_arguments, print_arguments
import math
import paddle.fluid.core as core

parser = argparse.ArgumentParser(description=__doc__)
add_arg = functools.partial(add_arguments, argparser=parser)
# yapf: disable
add_arg('batch_size',       int,  32,                 "Minibatch size.")
add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
add_arg('class_dim',        int,  1000,                "Class number.")
add_arg('image_shape',      str,  "3,224,224",         "Input image size")
add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
add_arg('model',            str, "SE_ResNeXt50_32x4d", "Set the network to use.")
# yapf: enable

model_list = [m for m in dir(models) if "__" not in m]

DEBUG = 1

def dot(program):
    dot_graph = ""
    dot_nodes = []
    dot_edges = []
    dot_graph += "digraph pm {\n"
    for block in program.blocks:
        ops = list(block.ops)
        block_id = block.idx
        for op in ops:
            op_type = op.type
            op_name = op_type + "_" + op.input_arg_names[0].replace(".", "_")
            for name in op.input_arg_names:
                name = name.replace(".", "_")
                dot_edge = name + " -> " + op_name
                if dot_edge not in dot_edges:
                    dot_edges.append(dot_edge)
                dot_node = name + " [shape=oval]"
                if dot_node not in dot_nodes:
                    dot_nodes.append(dot_node)

            for name in op.output_arg_names:
                name = name.replace(".", "_")
                dot_edge = op_name + " -> " + name
                if dot_edge not in dot_edges:
                    dot_edges.append(dot_edge)

            dot_node = op_name + " [shape=box]"
            if dot_node not in dot_nodes:
                dot_nodes.append(dot_node)

    for dot_edge in dot_edges:
        dot_graph += dot_edge + "\n"
    for dot_node in dot_nodes:
        dot_graph += dot_node + "\n"
    dot_graph += "}"

    file = open("model.dot", 'w')
    file.write(dot_graph)
    file.close()

def get_quantization_op_pos(program):
    conv_op_index = [index for index, value in enumerate(program.global_block().ops) if value.type == 'conv2d']
    if len(conv_op_index) < 2:
        return None
    return [conv_op_index[1]]

def get_dequantization_op_pos(program):
    conv_op_index = [index for index, value in enumerate(program.global_block().ops) if value.type == 'conv2d']
    if len(conv_op_index) < 2:
        return None
    res = []
    support_int8_op_type = ["pool2d"]
 
    for index, value in enumerate(conv_op_index[:-1]):
        if index == 0: continue

        if value + 1 == conv_op_index[index + 1]:
            continue
        else:
            start_index = index + 1
            end_index = conv_op_index[index + 1]
            while start_index < end_index:
                if program.global_block().ops[start_index].type not in support_int8_op_type:
                    print program.global_block().ops[start_index].type, end_index
                    res.append(start_index)
                    break
                else:
                    start_index += 1
    last_dequantize_op_index = conv_op_index[-1]
    # skip pooling op which is the Successor of the last conv op
    while program.global_block().ops[last_dequantize_op_index + 1].type in support_int8_op_type:
        last_dequantize_op_index += 1
    res.append(last_dequantize_op_index) # need to fix
    
    return res


def get_requantization_op_pos(program):
    pass

# def create_op(program, op_name, data_type):
def update_program_for_saving_var(program, name, value, data_shape, dst, data_type="float32"):
    tmp_var = program.current_block().create_var(
        name=name,
        dtype=data_type,
        persistable=True,
    )

    program.current_block().append_op(
        type='assign_value',
        outputs={'Out': [tmp_var]},
        attrs={
            'dtype':core.VarDesc.VarType.FP32,
            'shape': data_shape,
            'fp32_values': value
        }
    )

    program.current_block().append_op(
        type = 'save',
        inputs={'X': '{}'.format(name)},
        outputs={},
        attrs={"file_path": "{}/{}".format(dst, name)}
    )


def eval(args):
    # parameters from arguments
    class_dim = args.class_dim
    model_name = args.model
    pretrained_model = args.pretrained_model
    with_memory_optimization = args.with_mem_opt
    image_shape = [int(m) for m in args.image_shape.split(",")]

    assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                     model_list)

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    # model definition
    model = models.__dict__[model_name]()
    
    if model_name is "GoogleNet":
        out0, out1, out2 = model.net(input=image, class_dim=class_dim)
        cost0 = fluid.layers.cross_entropy(input=out0, label=label)
        cost1 = fluid.layers.cross_entropy(input=out1, label=label)
        cost2 = fluid.layers.cross_entropy(input=out2, label=label)
        avg_cost0 = fluid.layers.mean(x=cost0)
        avg_cost1 = fluid.layers.mean(x=cost1)
        avg_cost2 = fluid.layers.mean(x=cost2)

        avg_cost = avg_cost0 + 0.3 * avg_cost1 + 0.3 * avg_cost2
        acc_top1 = fluid.layers.accuracy(input=out0, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out0, label=label, k=5)
    else:
        out = model.net(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)

        avg_cost = fluid.layers.mean(x=cost)
        acc_top1 = fluid.layers.accuracy(input=out, label=label, k=1)
        acc_top5 = fluid.layers.accuracy(input=out, label=label, k=5)

    test_program = fluid.default_main_program().clone(for_test=True)
            
    if with_memory_optimization:
        fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if pretrained_model:
        def if_exist(var):
            return os.path.exists(os.path.join(pretrained_model, var.name))

        fluid.io.load_vars(exe, pretrained_model, predicate=if_exist)
    
    t = fluid.transpiler.InferenceTranspiler()
    t.transpile(test_program, fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace())

    conv_op_index = [index for index, value in enumerate(test_program.global_block().ops) if value.type == 'conv2d']
    weights_var_name = []
    conv_input_var_name = []
    conv_output_var_name = []

    for i in conv_op_index[1:]:
        weights_var_name.append(test_program.current_block().ops[i].input('Filter')[0])
        conv_input_var_name.append(test_program.current_block().ops[i].input('Input')[0])
        conv_output_var_name.append(test_program.current_block().ops[i].output('Output')[0])
    
    not_persistable_vars = (i for i in test_program.list_vars() if not i.persistable)
    back_program = test_program.clone()
    for i in not_persistable_vars:
        i.persistable= True
    
    var_name = [i.name for i in test_program.list_vars()]

    val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])

    fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]

    test_info = [[], [], []]
    cnt = 0
    var_max = {}
    for batch_id, data in enumerate(val_reader()):
        t1 = time.time()
        loss, acc1, acc5 = exe.run(test_program,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))
        for i in var_name:
            # print (np.array(fluid.global_scope().find_var(i).get_tensor()).shape)
            np_data = np.array(fluid.global_scope().find_var(i).get_tensor())

            if i in weights_var_name:
                max_value = [float(np.amax(np_data[j])) for j in range(np_data.shape[0])]
            else:
                max_value = [float(np.amax(np_data))]
            var_max[i] = []
            var_max[i].append(max_value)
        
        t2 = time.time()
        period = t2 - t1
        loss = np.mean(loss)
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        test_info[0].append(loss * len(data))
        test_info[1].append(acc1 * len(data))
        test_info[2].append(acc5 * len(data))
        cnt += len(data)
        if batch_id % 10 == 0:
            print("Testbatch {0},loss {1}, "
                  "acc1 {2},acc5 {3},time {4}".format(batch_id, \
                  loss, acc1, acc5, \
                  "%2.2f sec" % period))
            sys.stdout.flush()
        
        break
    
    test_loss = np.sum(test_info[0]) / cnt
    test_acc1 = np.sum(test_info[1]) / cnt
    test_acc5 = np.sum(test_info[2]) / cnt

    print("Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
        test_loss, test_acc1, test_acc5))
    sys.stdout.flush()

    infer_prog = test_program.clone()

    for i in conv_input_var_name:
        update_program_for_saving_var(infer_prog, i+"_scale.input.test", var_max[i][0], np.array(var_max[i]).shape, pretrained_model)
    
    for i in conv_output_var_name:
        update_program_for_saving_var(infer_prog, i+"_scale.output.test", var_max[i][0], np.array(var_max[i]).shape, pretrained_model)
    
    for i in weights_var_name:
        update_program_for_saving_var(infer_prog, i+"_scale.weights.test", var_max[i][0], np.array(var_max[i]).shape, pretrained_model)
    # update_program_for_saving_var(infer_prog, 'conv2_int8_tmp',  var_max[var_name[1]][0], [1,], pretrained_model)

    #Step 2 save all variable
    for batch_id, data in enumerate(val_reader()):
        loss, acc1, acc5 = exe.run(infer_prog,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))
        break
    
    int8_prog = back_program.clone()
    
    # for index, value in enumerate(conv_op_index[1:]):
    #     # print index,conv_input_var_name[index], ["{}_scale.input.test".format(conv_input_var_name[index])]
    #     int8_prog.current_block().ops[value].desc.set_input("Scale_in", ["{}_scale.input.test".format(conv_input_var_name[index])])
    #     int8_prog.current_block().ops[value].desc.set_input("Scale_out", ["{}_scale.output.test".format(conv_output_var_name[index])])
    #     int8_prog.current_block().ops[value].desc.set_input("Scale_weights", ["{}_scale.weights.test".format(weights_var_name[index])])
    #     if int8_prog.current_block().ops[value].desc.input("ResidualData"):
    #         name = int8_prog.current_block().ops[value].desc.input("ResidualData")[0]
    #         int8_prog.current_block().ops[value].desc.set_input("Scale_in_eltwise", ["{}_scale.output.test".format(name)])
    
    
    quantize_pos = get_quantization_op_pos(int8_prog)

    conv2_quantize_tmp = int8_prog.current_block().create_var(
        name="conv2_quantize_tmp",
        dtype=core.VarDesc.VarType.UINT8,
        # persistable=True,
        # lod_level= 0,
        # shape= shape
    )

    op = int8_prog.current_block()._insert_op(
       index=quantize_pos[0] ,
        
       type="quantize",
        
       inputs={"Input": int8_prog.current_block().ops[quantize_pos[0] - 1].output('Out')[0],
               "Scale": "{}_scale.input.test".format(conv_input_var_name[1])},
        
       outputs={"Output": conv2_quantize_tmp},

    )
    op._set_attr("data_format", "NCHW")
    op._set_attr("use_mkldnn", 1)

    # int8_prog.current_block().ops[quantize_pos[0] + 1 ].desc.set_input("Input", ["conv2_quantize_tmp"])
    # for i in int8_prog.current_block().ops[quantize_pos[0] + 2:]:
    #     if i.type == 'conv2d' and i.input('Input')[0] == int8_prog.current_block().ops[quantize_pos[0] + 1].output('Out')[0]:
    #         i.desc.set_input("Input",  ["conv2_quantize_tmp"])
   
    # dequantize_pos = get_dequantization_op_pos(int8_prog)
    # dequantize_tmp_var = int8_prog.current_block().create_var(
    #     name="dequantize_tmp_var",
    #     dtype="float32",
    #     persistable=True,
    #     #shape= (np.array(fluid.global_scope().find_var('pool2d_0.tmp_0').get_tensor())).shape
    # )
    
    # op = int8_prog.current_block()._insert_op(
    #    index=dequantize_pos[0] + 1,
        
    #    type= "dequantize",
        
    #    inputs={"Input": int8_prog.current_block().ops[dequantize_pos[0]].output('Out')[0],
    #            "Scale": "{}_scale.output.test".format( int8_prog.current_block().ops[dequantize_pos[0]].output('Out')[0])},
        
    #    outputs={"Output": dequantize_tmp_var},
    # )

    # int8_prog.current_block().ops[dequantize_pos[0] + 2].desc.set_input("X", ["dequantize_tmp_var"])

    #Step 3 Save the new model 
    # print int8_prog
    # for i in int8_prog.current_block().ops:
    #     print '********'
    #     print i
        # if i.type == 'conv2d':
        #     print i
    #     # print i.input_names;
    #     print  '----'
    #     print i.type
    #     for j in i.input_names:
    #         print j, i.input(j)[0] if i.input(j) else ' '
    #     for k in i.output_names:
    #         print k, i.output(k)[0]
    # print conv_op_index
    # print dequantize_pos
    # sys.exit(0)
    # if DEBUG:
    #     dot(int8_prog)
    # for i in int8_prog.current_block().ops:
    #     print i
    print int8_prog
    for batch_id, data in enumerate(val_reader()):
        loss, acc1, acc5 = exe.run(int8_prog,
                            fetch_list=fetch_list,
                            feed=feeder.feed(data))
        loss = np.mean(loss)
        acc1 = np.mean(acc1)
        acc5 = np.mean(acc5)
        test_info[0].append(loss * len(data))
        test_info[1].append(acc1 * len(data))
        test_info[2].append(acc5 * len(data))
        cnt += len(data)
        if batch_id % 10 == 0:
            print("Testbatch {0},loss {1}, "
                    "acc1 {2},acc5 {3}".format(batch_id, \
                    loss, acc1, acc5))
            sys.stdout.flush()
        break      
    with open("__model_quantized__", "wb") as f:
        f.write(int8_prog.desc.serialize_to_string())


def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
