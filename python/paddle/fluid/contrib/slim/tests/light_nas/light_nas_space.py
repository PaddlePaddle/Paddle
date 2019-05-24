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

from light_nasnet import LightNASNet

total_images = 1
lr = 1
num_epochs = 1
batch_size = 1
lr_strategy = ""
l2_decay = 1
momentum_rate = 1
image_shape = ""

__all__ = ['LightNASSpace']


class LightNASSpace(SearchSpace):
    def __init__(self):
        super(LightNASSpace, self).__init__()

    def init_tokens(self):
        """Get init tokens in search space.
        """
        return [
            0, 1, 2, 0, 1, 0, 0, 2, 1, 1, 1, 0, 3, 2, 0, 1, 1, 0, 3, 1, 0, 0, 1,
            0, 3, 2, 2, 1, 1, 0
        ]

    def range_table(self):
        """Get range table of current search space.
        """
        # [NAS_FILTER_SIZE, NAS_LAYERS_NUMBER, NAS_KERNEL_SIZE, NAS_FILTERS_MULTIPLIER, NAS_SHORTCUT, NAS_SE]
        return [
            4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2, 2, 4, 3, 3, 2, 2,
            2, 4, 3, 3, 2, 2, 2
        ]

    def create_net(self, tokens=None):
        """Create a network for training by tokens.
        """
        if tokens is None:
            tokens = self.init_tokens()
        startup_prog = fluid.Program()
        train_prog = fluid.Program()
        test_prog = fluid.Program()
        train_py_reader, train_cost, train_acc1, train_acc5, global_lr = build_program(
            is_train=True,
            main_prog=train_prog,
            startup_prog=startup_prog,
            args=args)
        test_py_reader, test_cost, test_acc1, test_acc5 = build_program(
            is_train=False,
            main_prog=test_prog,
            startup_prog=startup_prog,
            args=args)
        test_prog = test_prog.clone(for_test=True)

        train_reader = paddle.batch(
            reader.train(), batch_size=train_batch_size, drop_last=True)
        test_reader = paddle.batch(reader.val(), batch_size=test_batch_size)

        train_py_reader.decorate_paddle_reader(train_reader)
        test_py_reader.decorate_paddle_reader(test_reader)
        return startup_prog, train_prog, test_prog, (
            train_cost, train_acc1, train_acc5, global_lr), (
                test_cost, test_acc1, test_acc5)


def build_program(is_train, main_prog, startup_prog, tokens):

    with fluid.program_guard(main_prog, startup_prog):
        py_reader = fluid.layers.py_reader(
            capacity=16,
            shapes=[[-1] + image_shape, [-1, 1]],
            lod_levels=[0, 0],
            dtypes=["float32", "int64"],
            use_double_buffer=True)
        with fluid.unique_name.guard():
            image, label = fluid.layers.read_file(py_reader)
            model = LightNASNet()
            avg_cost, acc_top1, acc_top5 = net_config(
                image,
                label,
                model,
                class_dim=1000,
                bottleneck_params_list=tokens,
                scale_loss=1.0)
            avg_cost.persistable = True
            acc_top1.persistable = True
            acc_top5.persistable = True
            if is_train:
                params = model.params
                params["total_images"] = total_images
                params["lr"] = lr
                params["num_epochs"] = num_epochs
                params["learning_strategy"]["batch_size"] = batch_size
                params["learning_strategy"]["name"] = lr_strategy
                params["l2_decay"] = l2_decay
                params["momentum_rate"] = momentum_rate
                optimizer = optimizer_setting(params)
                optimizer.minimize(avg_cost)
                global_lr = optimizer._global_learning_rate()

        if is_train:
            return py_reader, avg_cost, acc_top1, acc_top5, global_lr
        else:
            return py_reader, avg_cost, acc_top1, acc_top5


def net_config(image,
               label,
               model,
               class_dim=1000,
               bottleneck_params_list=None,
               scale_loss=1.0):
    bottleneck_params_list = json.loads(bottleneck_params_list)
    bottleneck_params_list = [
        bottleneck_params_list[i:i + 7]
        for i in range(0, len(bottleneck_params_list), 7)
    ]
    out = model.net(input=image,
                    bottleneck_params_list=bottleneck_params_list,
                    class_dim=class_dim)
    cost, pred = fluid.layers.softmax_with_cross_entropy(
        out, label, return_softmax=True)
    if scale_loss > 1:
        avg_cost = fluid.layers.mean(x=cost) * float(scale_loss)
    else:
        avg_cost = fluid.layers.mean(x=cost)
    acc_top1 = fluid.layers.accuracy(input=pred, label=label, k=1)
    acc_top5 = fluid.layers.accuracy(input=pred, label=label, k=5)
    return avg_cost, acc_top1, acc_top5


def optimizer_setting(params):
    """optimizer setting.
    Args:
        params: dict, params.
    """
    ls = params["learning_strategy"]
    l2_decay = params["l2_decay"]
    momentum_rate = params["momentum_rate"]
    if ls["name"] == "piecewise_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        bd = [step * e for e in ls["epochs"]]
        base_lr = params["lr"]
        lr = []
        lr = [base_lr * (0.1**i) for i in range(len(bd) + 1)]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=fluid.layers.piecewise_decay(
                boundaries=bd, values=lr),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "cosine_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        step = int(total_images / batch_size + 1)
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "cosine_warmup_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        l2_decay = params["l2_decay"]
        momentum_rate = params["momentum_rate"]
        step = int(math.ceil(float(total_images) / batch_size))
        lr = params["lr"]
        num_epochs = params["num_epochs"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=cosine_decay_with_warmup(
                learning_rate=lr, step_each_epoch=step, epochs=num_epochs),
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "linear_decay":
        if "total_images" not in params:
            total_images = IMAGENET1000
        else:
            total_images = params["total_images"]
        batch_size = ls["batch_size"]
        num_epochs = params["num_epochs"]
        start_lr = params["lr"]
        end_lr = 0
        total_step = int((total_images / batch_size) * num_epochs)
        lr = fluid.layers.polynomial_decay(
            start_lr, total_step, end_lr, power=1)
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    elif ls["name"] == "adam":
        lr = params["lr"]
        optimizer = fluid.optimizer.Adam(learning_rate=lr)
    else:
        lr = params["lr"]
        optimizer = fluid.optimizer.Momentum(
            learning_rate=lr,
            momentum=momentum_rate,
            regularization=fluid.regularizer.L2Decay(l2_decay))
    return optimizer
