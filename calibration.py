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
                    res.append(start_index)
                    break
                else:
                    start_index += 1
    res.append(conv_op_index[-1]) #need to fix
    return res


def get_requantization_op_pos(program):
    pass

# def create_op(program, op_name, data_type):

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
    
    print 120, pretrained_model
    t = fluid.transpiler.InferenceTranspiler()
    t.transpile(test_program, fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace())
    # for i in test_program.current_block().ops:
    #     print i
    # sys.exit(0)
    conv_op_index = [index for index, value in enumerate(test_program.global_block().ops) if value.type == 'conv2d']
    print (conv_op_index)
    weights_var_name = []
    conv_input_var_name = []
    conv_output_var_name = []
    weights_channel = {}
    for i in conv_op_index[1:]:
        weights_var_name.append(test_program.current_block().ops[i].input('Filter')[0])
        conv_input_var_name.append(test_program.current_block().ops[i].input('Input')[0])
        conv_output_var_name.append(test_program.current_block().ops[i].output('Output')[0])

    for i in test_program.list_vars():
        if i.name in weights_var_name:
            weights_channel[i.name] = i.shape[0]

    # print weights_var_name
    # print '-------'
    # print conv_input_var_name    
    # print '-------'

    # print conv_output_var_name
    # for i in test_program.current_block().ops:
    #     print ('-----------')
    #     print (i.input_names, i.output_names)
    #     if i.type == 'conv2d':
    #         print i.input('Filter')
        # print (i.input_arg_names)
    #     print (i.output_arg_names)
    #     # print (i.block_attr)
    #     print (dir(i))
    #     print (i.attr_names)
    #     print ((i.attr))
    #     for j in i.attr_names:
    #         print ((i.attr(j)))
        # print (i.blocks_attr)
    # sys.exit(0)
    # for i in test_program.list_vars():
    #     print (i.name)
    #     # print dir(i)
    #     print i.shape, i.type, i.dtype
        # if i.name == "batch_norm_52.b_0_fuse_bn":
        #     i.dtype = fluid.core.VarDesc.VarType.INT8;
    # print (test_program.global_block().ops[23].type)
    # for i in conv_op_index:
    #     op = test_program.current_block().ops[i]
    #     print (op)
    #     print (op.input_names, op.input_arg_names, op.output_arg_names)
    not_persistable_vars = (i for i in test_program.list_vars() if not i.persistable)
    for i in not_persistable_vars:
    #     # print (i.name, i.persistable)
        i.persistable= True
    # int8_prog = test_program.clone()
    var_name = [i.name for i in test_program.list_vars()]
    # get_dequantization_op_pos(int8_prog)
    # print var_name
    # sys.exit(0)
    
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
        # print var_max
        
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
    #insert quantization op
    infer_prog = test_program.clone()
    pos = get_quantization_op_pos(infer_prog)
    print pos
    print infer_prog.current_block().ops[1].output('Out')[0]
    conv2_scale_in = infer_prog.global_block().create_var(
        name="conv2_scale_in",
        dtype="float32",
        persistable=True,
    )
    # conv2_weights_in = infer_prog.global_block().create_var(
    #     name="conv2_weights_in",
    #     dtype="float32",
    #     persistable=True,
    # )
    conv2_int8_tmp = infer_prog.global_block().create_var(
        name="conv2_int8_tmp",
        dtype="int8",
        persistable=True,
        shape= (np.array(fluid.global_scope().find_var('pool2d_0.tmp_0').get_tensor())).shape
    )
    # print ((np.array(fluid.global_scope().find_var('pool2d_0.tmp_0').get_tensor())).shape)
    # sys.exit(0)
    # fluid.initializer.Constant(value=1.0)(conv2_int8_tmp, infer_prog.global_block())

    infer_prog.current_block().append_op(
        type='assign_value',
        outputs={'Out': [conv2_scale_in]},
        attrs={
            'dtype':core.VarDesc.VarType.FP32,
            'shape': [1,1],
            'fp32_values': var_max[var_name[1]][0]
        }
    )

    # infer_prog.current_block().append_op(
    #     type='assign_value',
    #     outputs={'Out': [conv2_int8_tmp]},
    #     attrs={
    #         'dtype':core.VarDesc.VarType.UINT8,
    #         'shape': (np.array(fluid.global_scope().find_var('pool2d_0.tmp_0').get_tensor())).shape,
    #         # 'fp32_values': var_max[var_name[1]][0]
    #     }
    # )
    # op = infer_prog.current_block()._insert_op(
    #     index=pos[0],
    #     type= "quantize",
    #     inputs={"Input": infer_prog.current_block().ops[1].output('Out')[0],
    #             "Scale": conv2_scale_in},
    #     outputs={"Output":conv2_int8_tmp},
    #     # attrs= {
    #     #     "data_format": "NCHW"
    #     # }
    #     )
    # op.set_attr("data_format", "NCHW")
    # op.set_attr("use_mkldnn", 1)
    # infer_prog.current_block().ops[3].set_input("Input", ['conv2_int8_tmp'])
    # infer_prog.current_block().append_op(
    #     type='assign_value',
    #     outputs={'Out': [conv2_weights_in]},
    #     attrs={
    #         'dtype':core.VarDesc.VarType.FP32,
    #         'shape': [1,1],
    #         'fp32_values':  [3.12]
    #     }
    # )

    # for i in infer_prog.current_block().ops[:4]:
    #     print (i)
    # sys.exit(0)
    # with open("/home/guomingz/__model_xiaoli_quantize__", "wb") as f:
    #     f.write(infer_prog.desc.serialize_to_string())

    infer_prog.current_block().append_op(
        type = 'save',
        inputs={'X': 'conv2_scale_in'},
        outputs={},
        attrs={"file_path": "{}/conv2_scale_in".format(pretrained_model)}
    )
    
        # infer_prog.current_block().append_op(
        #     type = 'save',
        #     inputs={'X': 'conv2_int8_tmp'},
        #     outputs={},
        #     attrs={"file_path": "{}/conv2_int8_tmp".format(pretrained_model)}
        # )
    # val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)

    for batch_id, data in enumerate(val_reader()):
        # print (feeder.feed(data))
        # print (fetch_list)
        loss, acc1, acc5 = exe.run(infer_prog,
                                   fetch_list=fetch_list,
                                   feed=feeder.feed(data))
        sys.exit(0)
    # infer_prog.current_block().append_op(
    #     type = 'save',
    #     inputs={'X': 'conv2_weights_in'},
    #     outputs={},
    #     attrs={"file_path": "{}/conv2_weights_in".format(pretrained_model)}
    # )

    #insert dequantization op
    
    #rerun to save variable

    # for batch_id, data in enumerate(val_reader()):
    #     t1 = time.time()
    # loss, acc1, acc5 = exe.run(test_program,
    #                            fetch_list=fetch_list,
    #                            feed=feeder.feed(data))
    # with open("/home/guomingz/__model__", "wb") as f:
    #     f.write(test_program.desc.serialize_to_string())

def main():
    args = parser.parse_args()
    print_arguments(args)
    eval(args)


if __name__ == '__main__':
    main()
