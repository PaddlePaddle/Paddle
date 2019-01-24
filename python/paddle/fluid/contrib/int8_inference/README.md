The 8-bits (**INT8**) inference is also known as the Low Precision Inference which could speed up the inference with the lower accuracy loss. It has higher throughput and lower memory requirements compared to FP32. As the PaddlePaddle enables the INT8 inference supporting from the 1.3 release, we release a accuracy tool interface([Calibrator Class](https://github.com/PaddlePaddle/Paddle/blob/develop/python/paddle/fluid/contrib/int8_inference/utility.py#L21)) at the same time. This offline tool will generate the initial quantization parameters firstly and save those scaling factors into the model definition file.

## 0.Prerequisite
This tool is dedicated to run on Intel SkyLake platform which supports the 8-Bit Inference.

## 1.Usage
1. Set the env with `export FLAGS_use_mkldnn=True` and build the PaddlePaddle.
2. Generate the model definition file with quantized parameters firstly. We provide a template calibration.py to sample and save the model definition for       int8 inference. Below is the full code snippet. You need to save this file to **/path/to/models/fluid/PaddleCV/image_classification**.
    ```python
    from __future__ import absolute_import
    from __future__ import division
    import numpy as np
    import time
    import sys
    import paddle
    import paddle.fluid as fluid
    import models
    import reader
    import argparse
    import functools
    from utility import add_arguments, print_arguments
    import paddle.fluid.core as core
    import os
    sys.path.append('/Path/To/python/paddle/fluid/contrib/int8_inference')
    import int8_inference.utility as ut
    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('batch_size',       int,  32,                 "Minibatch size.")
    add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
    add_arg('class_dim',        int,  1000,                "Class number.")
    add_arg('image_shape',      str,  "3,224,224",         "Input image size")
    add_arg('out',              str,  "calibration_out",   "Output INT8 model")
    add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
    add_arg('use_train_data',   bool, False,               "Whether to use train data for sampling or not.")
    add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
    add_arg('model',            str, "SE_ResNeXt50_32x4d", "Set the network to use.")
    add_arg('iterations',       int, 1, "Sampling iteration")
    add_arg('algo',             str, 'direct', "calibration algo")
    add_arg('debug',            bool, False, "print program and save the dot file")

    # yapf: enable

    model_list = [m for m in dir(models) if "__" not in m]

    def eval(args):
        # parameters from arguments
        class_dim = args.class_dim
        model_name = args.model
        pretrained_model = args.pretrained_model
        with_memory_optimization = args.with_mem_opt
        image_shape = [int(m) for m in args.image_shape.split(",")]

        assert model_name in model_list, "{} is not in lists: {}".format(args.model,
                                                                        model_list)
        int8_model = os.path.join(os.getcwd(), args.out)
        print("Start calibration for {}...".format(model_name))

        tmp_scale_folder = ".tmp_scale"

        if os.path.exists(
                int8_model):  # Not really need to execute below operations
            os.system("rm -rf " + int8_model)
            os.system("mkdir " + int8_model)

        if not os.path.exists(tmp_scale_folder):
            os.system("mkdir {}".format(tmp_scale_folder))

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
        t.transpile(test_program,
                    fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace())

        sampling_reader = paddle.batch(
            reader.train() if args.use_train_data else reader.val(),
            batch_size=args.batch_size)
        feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
        fetch_list = [avg_cost.name, acc_top1.name, acc_top5.name]
        # Create calibrator object firstly.
        calibrator = ut.Calibrator(
            program=test_program,
            pretrained_model=pretrained_model,
            iterations=args.iterations,
            debug=args.debug,
            algo=args.algo)

        sampling_data = {}

        calibrator.generate_sampling_program()
        feeded_var_names = None
        for batch_id, data in enumerate(sampling_reader()):
            _, _, _ = exe.run(calibrator.sampling_program,
                            fetch_list=fetch_list,
                            feed=feeder.feed(data))
            for i in calibrator.sampling_program.list_vars():
                if i.name in calibrator.sampling_vars:
                    np_data = np.array(fluid.global_scope().find_var(i.name)
                                    .get_tensor())
                    if i.name not in sampling_data:
                        sampling_data[i.name] = []
                    sampling_data[i.name].append(np_data)

            if batch_id != args.iterations - 1:
                continue
            feeded_var_names = feeder.feed(data).keys()
            break

        calibrator.generate_quantized_data(sampling_data)

        fluid.io.save_inference_model(int8_model, feeded_var_names, [
            calibrator.sampling_program.current_block().var(i) for i in fetch_list
        ], exe, calibrator.sampling_program)
        print(
            "Calibration is done and the corresponding files were generated at {}".
            format(os.path.abspath(args.out)))

    def main():
        args = parser.parse_args()
        print_arguments(args)
        eval(args)

    if __name__ == '__main__':
        main()
    ```
    The typical command of this tool is `python calibration.py --model=ResNet50  --batch_size=50  --image_shape=3,224,224  --with_mem_opt=True  --use_gpu=False  --pretrained_model=weights/resnet50 --debug=False`. When this command executed, it will generate the model definition file which contains scaling factor and save it into the specified output folder.

    Here is the parameter definitions:
    * --out: the output folder name for saving the quantized model definition file, the default value is calibration_out;
    * --pretrained_model: the folder that contains pretrained FP32weights and model;
    * --model: model name, e.g ResNet50/Mobilenet
    * --debug: generate the dot file and print the program if set the flag to True

3. Run the INT8 inference with the model has the scaling factors. We also provide a template `eval_tp_with_model.py` to run INT8 inference.
   Below is the full code snippet.You need to save this file to **/path/to/models/fluid/PaddleCV/image_classification**.
   ```python
    from __future__ import absolute_import
    from __future__ import division
    from __future__ import print_function
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
    from utility import add_arguments, print_arguments
    import paddle.fluid.profiler as profiler

    parser = argparse.ArgumentParser(description=__doc__)
    add_arg = functools.partial(add_arguments, argparser=parser)
    # yapf: disable
    add_arg('batch_size',       int,  256,                 "Minibatch size.")
    add_arg('use_gpu',          bool, True,                "Whether to use GPU or not.")
    add_arg('class_dim',        int,  1000,                "Class number.")
    add_arg('image_shape',      str,  "3,224,224",         "Input image size")
    add_arg('with_mem_opt',     bool, True,                "Whether to use memory optimization or not.")
    add_arg('pretrained_model', str,  None,                "Whether to use pretrained model.")
    add_arg('skip_batch_num',   int,  5,                    "Skip batch num.")
    add_arg('use_transpiler',   bool, True,                 'Whether to use transpiler.')
    add_arg('use_fake_data',    bool, False,                'If set, use fake data instead of real data.')
    add_arg('iterations',	    int,  100,               	'Fake data iterations')
    add_arg('profiler',	    bool, False,                'If true, do profiling.')
    # yapf: enable

    model_list = [m for m in dir(models) if "__" not in m]

    def user_data_reader(data):
        '''
        Creates a data reader for user data.
        '''

        def data_reader():
            while True:
                for b in data:
                    yield b

        return data_reader

    def eval(args):
        # parameters from arguments
        class_dim = args.class_dim
        pretrained_model = args.pretrained_model
        with_memory_optimization = args.with_mem_opt
        skip_batch_num = args.skip_batch_num
        if skip_batch_num >= args.iterations:
        print("Please ensure the skip_batch_num less than iterations.")
        sys.exit(0)
        image_shape = [int(m) for m in args.image_shape.split(",")]

        if with_memory_optimization:
            fluid.memory_optimize(fluid.default_main_program())

        place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
        exe = fluid.Executor(place)

        [infer_program, feed_dict,
        fetch_targets] = fluid.io.load_inference_model(pretrained_model, exe)
        program = infer_program.clone()
        if args.use_transpiler:
            inference_transpiler_program = program.clone()
            t = fluid.transpiler.InferenceTranspiler()
            t.transpile(inference_transpiler_program, place)
            program = inference_transpiler_program

        fake_data = [(
            np.random.rand(image_shape[0] * image_shape[1] * image_shape[2]).astype(np.float32),
            np.random.randint(1, class_dim)) for _ in range(1)]

        if args.use_fake_data:
            val_reader = paddle.batch(
                user_data_reader(fake_data), batch_size=args.batch_size)
        else:
            val_reader = paddle.batch(reader.val(), batch_size=args.batch_size)

        test_info = [[], [], []]
        cnt = 0
        periods = []
        for batch_id, data in enumerate(val_reader()):
            if args.use_fake_data:
                data = val_reader().next()
            image = np.array(map(lambda x: x[0].reshape(image_shape), data)).astype(
                "float32")
            label = np.array(map(lambda x: x[1], data)).astype("int64")
            label = label.reshape([-1, 1])

            t1 = time.time()
            loss, acc1, acc5 = exe.run(program, feed={feed_dict[0]: image, feed_dict[1]: label}, fetch_list=fetch_targets)

            t2 = time.time()
            period = t2 - t1
            loss = np.mean(loss)
            acc1 = np.mean(acc1)
            acc5 = np.mean(acc5)
            test_info[0].append(loss * len(data))
            test_info[1].append(acc1 * len(data))
            test_info[2].append(acc5 * len(data))
            periods.append(period)
            cnt += len(data)
            if batch_id % 10 == 0:
                print("Testbatch {0},loss {1}, "
                    "acc1 {2},acc5 {3},time {4}".format(batch_id, loss, acc1, acc5, "%2.2f sec" % period))
                sys.stdout.flush()
            if batch_id == args.iterations - 1:
                break

        test_loss = np.sum(test_info[0]) / cnt
        test_acc1 = np.sum(test_info[1]) / cnt
        test_acc5 = np.sum(test_info[2]) / cnt
        throughput = cnt / np.sum(periods)
        throughput_skip = (cnt-skip_batch_num*args.batch_size) / np.sum(periods[skip_batch_num:])
        latency = np.average(periods)
        latency_skip = np.average(periods[skip_batch_num:])
        print("Test_loss {0}, test_acc1 {1}, test_acc5 {2}".format(
            test_loss, test_acc1, test_acc5))
        sys.stdout.flush()
        print("throughput {0}, throughput_skip {1}, latency {2}, latency_skip {3}".format(
            throughput, throughput_skip, latency, latency_skip))
        sys.stdout.flush()

    def main():
        args = parser.parse_args()
        print_arguments(args)
        if args.profiler:
            if args.use_gpu:
                with profiler.cuda_profiler("cuda_profiler.txt", 'csv') as nvprof:
                    eval(args)
            else:
                with profiler.profiler("CPU", sorted_key='total') as cpuprof:
                    eval(args)
        else:
            eval(args)

    if __name__ == '__main__':
        main()
   ```
    The typical command of this tool is ` python eval_tp_with_model.py  --batch_size=50  --image_shape=3,224,224  --with_mem_opt=True  --use_gpu=False --pretrained_model=calibration_out --iterations 1000`. When this command exectued, it will display the accuracy and throughput on the screen.

    Here is the parameter definitions:
    * --pretrained_model: the folder that contains pretrained INT8 weights and model;
    * --iterations: equal to test iteration definition, usually its value equal to epoch/batch size
