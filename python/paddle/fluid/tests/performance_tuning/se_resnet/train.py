import os
import paddle.v2 as paddle
import paddle.fluid as fluid
import paddle.v2.dataset.flowers as flowers
import paddle.fluid.profiler as profiler

import time

import os

cards = os.getenv("CUDA_VISIBLE_DEVICES") or ""
cards_num = len(cards.split(","))

total_batch_num = 40
batch_num = total_batch_num / cards_num

per_gpu_batch_size=6
batch_size=per_gpu_batch_size * cards_num

print("cards_num=" + str(cards_num))
print("per_gpu_batch_size=" + str(per_gpu_batch_size))
print("batch_size=" + str(batch_size))
print("batch_num=" + str(batch_num))

def conv_bn_layer(input, num_filters, filter_size, stride=1, groups=1,
                  act=None):
    conv = fluid.layers.conv2d(
        input=input,
        num_filters=num_filters,
        filter_size=filter_size,
        stride=stride,
        padding=(filter_size - 1) / 2,
        groups=groups,
        act=None,
        bias_attr=False)
    return fluid.layers.batch_norm(input=conv, act=act)


def squeeze_excitation(input, num_channels, reduction_ratio):
    pool = fluid.layers.pool2d(
        input=input, pool_size=0, pool_type='avg', global_pooling=True)
    squeeze = fluid.layers.fc(input=pool,
                              size=num_channels / reduction_ratio,
                              act='relu')
    excitation = fluid.layers.fc(input=squeeze,
                                 size=num_channels,
                                 act='sigmoid')
    scale = fluid.layers.elementwise_mul(x=input, y=excitation, axis=0)
    return scale


def shortcut(input, ch_out, stride):
    ch_in = input.shape[1]
    if ch_in != ch_out:
        if stride == 1:
            filter_size = 1
        else:
            filter_size = 3
        return conv_bn_layer(input, ch_out, filter_size, stride)
    else:
        return input


def bottleneck_block(input, num_filters, stride, cardinality, reduction_ratio):
    conv0 = conv_bn_layer(
        input=input, num_filters=num_filters, filter_size=1, act='relu')
    conv1 = conv_bn_layer(
        input=conv0,
        num_filters=num_filters,
        filter_size=3,
        stride=stride,
        groups=cardinality,
        act='relu')
    conv2 = conv_bn_layer(
        input=conv1, num_filters=num_filters * 2, filter_size=1, act=None)
    scale = squeeze_excitation(
        input=conv2,
        num_channels=num_filters * 2,
        reduction_ratio=reduction_ratio)

    short = shortcut(input, num_filters * 2, stride)

    return fluid.layers.elementwise_add(x=short, y=scale, act='relu')


def SE_ResNeXt(input, class_dim, infer=False):
    cardinality = 64
    reduction_ratio = 16
    depth = [3, 8, 36, 3]
    num_filters = [128, 256, 512, 1024]

    conv = conv_bn_layer(
        input=input, num_filters=64, filter_size=3, stride=2, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=64, filter_size=3, stride=1, act='relu')
    conv = conv_bn_layer(
        input=conv, num_filters=128, filter_size=3, stride=1, act='relu')
    conv = fluid.layers.pool2d(
        input=conv, pool_size=3, pool_stride=2, pool_padding=1, pool_type='max')

    for block in range(len(depth)):
        for i in range(depth[block]):
            conv = bottleneck_block(
                input=conv,
                num_filters=num_filters[block],
                stride=2 if i == 0 and block != 0 else 1,
                cardinality=cardinality,
                reduction_ratio=reduction_ratio)

    pool = fluid.layers.pool2d(
        input=conv, pool_size=0, pool_type='avg', global_pooling=True)
    if not infer:
        drop = fluid.layers.dropout(x=pool, dropout_prob=0.2)
    else:
        drop = pool
    out = fluid.layers.fc(input=drop, size=class_dim, act='softmax')
    return out


def train(learning_rate,
          batch_size,
          num_passes,
          init_model=None,
          model_save_dir='model',
          parallel=True):
    class_dim = 1000
    image_shape = [3, 224, 224]

    image = fluid.layers.data(name='image', shape=image_shape, dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    if parallel:
        places = fluid.layers.get_places()
        pd = fluid.layers.ParallelDo(places, use_nccl=True)

        with pd.do():
            image_ = pd.read_input(image)
            label_ = pd.read_input(label)
            out = SE_ResNeXt(input=image_, class_dim=class_dim)
            cost = fluid.layers.cross_entropy(input=out, label=label_)
            avg_cost = fluid.layers.mean(x=cost)
            accuracy = fluid.layers.accuracy(input=out, label=label_)
            pd.write_output(avg_cost)
            pd.write_output(accuracy)

        avg_cost, accuracy = pd()
        avg_cost = fluid.layers.mean(x=avg_cost)
        accuracy = fluid.layers.mean(x=accuracy)
    else:
        out = SE_ResNeXt(input=image, class_dim=class_dim)
        cost = fluid.layers.cross_entropy(input=out, label=label)
        avg_cost = fluid.layers.mean(x=cost)
        accuracy = fluid.layers.accuracy(input=out, label=label)

    #optimizer = fluid.optimizer.Momentum(
    #    learning_rate=learning_rate,
    #    momentum=0.9,
    #    regularization=fluid.regularizer.L2Decay(1e-4))
    optimizer = fluid.optimizer.SGD(learning_rate=0.002)
    opts = optimizer.minimize(avg_cost)

    fluid.memory_optimize(fluid.default_main_program())

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(fluid.default_startup_program())

    if init_model is not None:
        fluid.io.load_persistables(exe, init_model)

    train_reader = paddle.batch(flowers.train(), batch_size=batch_size)
    test_reader = paddle.batch(flowers.test(), batch_size=batch_size)
    feeder = fluid.DataFeeder(place=place, feed_list=[image, label])
    train_reader_iter = train_reader()
    train_reader_iter.next()

    for pass_id in range(num_passes):
      #with profiler.profiler('GPU', 'total') as prof:
      #with profiler.profiler(state="All"):
        train_time = 0.0
        reader_time = 0.0

        data = train_reader_iter.next()
        for batch_id in range(batch_num):
            reader_start = time.time()
            reader_stop = time.time()
            reader_time += reader_stop - reader_start

            train_start = time.time()
            exe.run(fluid.default_main_program(),
                           feed=feeder.feed(data),
                           fetch_list=[])
            train_stop = time.time()
            train_time += train_stop - train_start
            print("Pass {0}, batch {1}".format(pass_id, batch_id))

        print("\n\n\n")
        print("train_time=" + str(train_time))
        print("reader_time=" + str(reader_time))
        break
      #break


if __name__ == '__main__':
    train(
        learning_rate=0.1,
        batch_size=batch_size,
        num_passes=100,
        init_model=None,
        parallel=True)
