import paddle
import paddle.nn as nn
import paddle.sparse as sparse
from paddle.fluid.framework import _test_eager_guard
import time
import numpy as np
import torch
import spconv.pytorch as spconv
import inspect

class MiddleExtractor(paddle.nn.Layer):
    def __init__(self,
                #output_shape,
                use_norm=True,
                num_input_features=128,
                num_filters_down1=[64],
                num_filters_down2=[64, 64],
                name='MiddleExtractor'):
        super(MiddleExtractor, self).__init__()
        self.name = name
        if not use_norm:
            self.middle_conv = paddle.nn.Sequential(
                #nn.Pad3D(1),
                nn.Conv3D(num_input_features, 64, 3, stride=(2, 1, 1), data_format='NDHWC'),
                #nn.BatchNorm3D(64, epsilon=1e-3, momentum=0.001, data_format='NDHWC'),
                nn.ReLU(),
                #nn.Pad3D([1, 1, 1, 1, 0, 0]),
                nn.Conv3D(64, 64, 3, stride=(1, 1, 1), data_format='NDHWC'),
                #nn.BatchNorm3D(64, epsilon=1e-3, momentum=0.001, data_format='NDHWC'),
                nn.ReLU(),
                #nn.Pad3D(1),
                nn.Conv3D(64, 64, 3, stride=(2, 1, 1), data_format='NDHWC'),
                #nn.BatchNorm3D(64, epsilon=1e-3, momentum=0.001, data_format='NDHWC'),
                nn.ReLU(),
            )
        else:
            self.middle_conv = paddle.nn.Sequential(
                #nn.Pad3D(1),
                nn.Conv3D(num_input_features, 64, 3, stride=(2, 1, 1), data_format='NDHWC'),
                nn.BatchNorm3D(64, epsilon=1e-3, momentum=0.001, data_format='NDHWC'),
                nn.ReLU(),
                #nn.Pad3D([1, 1, 1, 1, 0, 0]),
                nn.Conv3D(64, 64, 3, stride=(1, 1, 1), data_format='NDHWC'),
                nn.BatchNorm3D(64, epsilon=1e-3, momentum=0.001, data_format='NDHWC'),
                nn.ReLU(),
                #nn.Pad3D(1),
                nn.Conv3D(64, 64, 3, stride=(2, 1, 1), data_format='NDHWC'),
                nn.BatchNorm3D(64, epsilon=1e-3, momentum=0.001, data_format='NDHWC'),
                nn.ReLU(),
            )
    def forward(self, x):
        return self.middle_conv(x)


def get_pos_to_kw_map(func):
    pos_to_kw = {}
    fsig = inspect.signature(func)
    pos = 0
    for name, info in fsig.parameters.items():
        if info.kind is info.POSITIONAL_OR_KEYWORD:
            pos_to_kw[pos] = name
        pos += 1
    return pos_to_kw

def change_default_args(**kwargs):
    def layer_wrapper(layer_class):
        class DefaultArgLayer(layer_class):
            def __init__(self, *args, **kw):
                pos_to_kw = get_pos_to_kw_map(layer_class.__init__)
                kw_to_pos = {kw: pos for pos, kw in pos_to_kw.items()}
                for key, val in kwargs.items():
                    if key not in kw and kw_to_pos[key] > len(args):
                        kw[key] = val
                super().__init__(*args, **kw)

        return DefaultArgLayer

    return layer_wrapper

class Empty(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super(Empty, self).__init__()

    def forward(self, *args, **kwargs):
        if len(args) == 1:
            return args[0]
        elif len(args) == 0:
            return None
        return args

class SpconvMiddleExtractor(torch.nn.Module):
    def __init__(self,
                #output_shape,
                use_norm=True,
                num_input_features=128,
                num_filters_down1=[64],
                num_filters_down2=[64, 64],
                name='SpconvMiddleExtractor'):
        super(SpconvMiddleExtractor, self).__init__()
        if use_norm:
            BatchNorm1d = change_default_args(
                eps=1e-3, momentum=0.01)(torch.nn.BatchNorm1d)
            Linear = change_default_args(bias=False)(nn.Linear)
        else:
            BatchNorm1d = Empty
            Linear = change_default_args(bias=True)(nn.Linear)

        middle_layers = []

        num_filters = [num_input_features] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d1:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False))
            if use_norm:
                middle_layers.append(BatchNorm1d(o))
            middle_layers.append(torch.nn.ReLU())

        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        if use_norm:
            middle_layers.append(
                BatchNorm1d(num_filters[-1]))
        middle_layers.append(torch.nn.ReLU())
        # assert len(num_filters_down2) > 0
        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2
        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]]
                            for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d2:
            middle_layers.append(spconv.SubMConv3d(i, o, 3, bias=False))
            if use_norm:
                middle_layers.append(BatchNorm1d(o))
            middle_layers.append(torch.nn.ReLU())
        middle_layers.append(
            spconv.SparseConv3d(
                num_filters[-1],
                num_filters[-1], (3, 1, 1), (2, 1, 1),
                bias=False))
        if use_norm:
            middle_layers.append(
                BatchNorm1d(num_filters[-1]))
        middle_layers.append(torch.nn.ReLU())
        #middle_layers.append(scn.SparseToDense(3, num_filters[-1]))
        middle_layers.append(spconv.ToDense())
        self.middle_conv = spconv.SparseSequential(*middle_layers)

    def forward(self, x):
        out = self.middle_conv(x)
        return out

class SparseMiddleExtractor(paddle.nn.Layer):
    def __init__(self,
                #output_shape,
                use_norm=True,
                num_input_features=128,
                num_filters_down1=[64],
                num_filters_down2=[64, 64],
                name='SparseMiddleExtractor'):
        super(SparseMiddleExtractor, self).__init__()
        self.name = name

        middle_layers = []
        num_filters = [num_input_features] + num_filters_down1
        filters_pairs_d1 = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]
        for i, o in filters_pairs_d1:
            middle_layers.append(sparse.SubmConv3D(i, o, 3, bias_attr=False))
            if use_norm:
                middle_layers.append(sparse.BatchNorm(o, epsilon=1e-3, momentum=0.99))
            middle_layers.append(sparse.ReLU())

        middle_layers.append(sparse.Conv3D(num_filters[-1], num_filters[-1], (3, 1, 1), (2, 1, 1), bias_attr=False))
        if use_norm:
            middle_layers.append(sparse.BatchNorm(num_filters[-1], epsilon=1e-3, momentum=0.99))
        middle_layers.append(sparse.ReLU())

        if len(num_filters_down1) == 0:
            num_filters = [num_filters[-1]] + num_filters_down2
        else:
            num_filters = [num_filters_down1[-1]] + num_filters_down2

        filters_pairs_d2 = [[num_filters[i], num_filters[i + 1]] for i in range(len(num_filters) - 1)]

        for i, o in filters_pairs_d2:
            middle_layers.append(sparse.SubmConv3D(i, o, 3, bias_attr=False))
            if use_norm:
                middle_layers.append(sparse.BatchNorm(o, epsilon=1e-3, momentum=0.99))
            middle_layers.append(sparse.ReLU())

        middle_layers.append(sparse.Conv3D(num_filters[-1], num_filters[-1], (3, 1, 1), (2, 1, 1), bias_attr=False))
        if use_norm:
            middle_layers.append(sparse.BatchNorm(num_filters[-1], epsilon=1e-3, momentum=0.99))
        middle_layers.append(sparse.ReLU())

        self.middle_conv = nn.Sequential(*middle_layers)
        
    def forward(self, x):
        sparse_out = self.middle_conv(x)
        return sparse_out.to_dense()


def test():
    with _test_eager_guard():
        in_channels = 128 
        shape = [20, 400, 352]
        batch_size = 1
        sparsity = 0.9

        full_shape = [batch_size] + shape + [in_channels]
        print(full_shape)

        total_elements = np.prod(shape)
        nnz = int(total_elements * (1-sparsity))
        print("nnz=", nnz)
        #product indices
        indices = []
        for i in range(4):
           indices.append(paddle.randint(0, full_shape[i], [1, nnz])) 
        
        indices = paddle.concat(indices)
        #product values
        values = paddle.randn((nnz, in_channels))

        sparse_x = paddle.sparse.sparse_coo_tensor(indices, values, shape=full_shape)

        dense_x = sparse_x.to_dense()

        #spconv
        device = torch.device("cuda")
        torch_x = torch.tensor(dense_x.numpy(), device=device)
        torch_x.required_grad=True

        spconv_x = spconv.SparseConvTensor.from_dense(torch_x)

        use_norm = False
        dense_model = MiddleExtractor(use_norm=use_norm, num_input_features=in_channels)
        spconv_model = SpconvMiddleExtractor(use_norm=use_norm, num_input_features=in_channels).to(device)
        sparse_model = SparseMiddleExtractor(use_norm=use_norm, num_input_features=in_channels)
        layer_nums = len(sparse_model.middle_conv)
        block_size = 3 if use_norm else 2
        layer_nums = int(layer_nums / block_size)

        for i in range(0, layer_nums):
            weight = paddle.to_tensor(spconv_model.middle_conv[i * block_size].weight.detach().cpu().numpy())
            sparse_model.middle_conv[i * block_size].weight.set_value(paddle.transpose(weight, [1,2,3,4,0]))

        print(dense_model)
        print(sparse_model)
        print(spconv_model)
        paddle.device.cuda.synchronize()
        #warm up
        dense_x.stop_gradient=True
        out1 = dense_model(dense_x)
        paddle.device.cuda.synchronize()
        sparse_x.stop_gradient=True
        out2 = sparse_model(sparse_x)
        paddle.device.cuda.synchronize()
        out3 = spconv_model(spconv_x)
        torch.cuda.synchronize(device)
        #warm up

        iters = 1
        t0 = time.time()
        dense_x.stop_gradient=False
        #padde dense
        for i in range(iters):
            out1 = dense_model(dense_x)
            out1.backward(out1)
            paddle.device.cuda.synchronize()
        t1 = time.time()

        sparse_x.stop_gradient=False
        #padde sparse
        for i in range(iters):
            out2 = sparse_model(sparse_x)
            out2.backward(out2)
            paddle.device.cuda.synchronize()
        t2 = time.time()

        spconv_x.features.required_grad=True
        print(spconv_x.features)
        #spconv
        for i in range(iters):
            out3 = spconv_model(spconv_x)
            out3.backward(out3)
            torch.cuda.synchronize(device)
        t3 = time.time()

        print(out3)
        print("dense time: ", t1 - t0)
        print("sparse time: ", t2 - t1)
        print("spconv time: ", t3 - t2)

        assert np.allclose(paddle.transpose(out2, [0, 4, 1, 2, 3]).numpy(), out3.detach().cpu().numpy(), atol=1e-3, rtol=1e-3)
        #np.allclose(spconv_x.features.grad.cpu().numpy(), sparse_x.grad.values().numpy(), atol=1e-3, rtol=1e-3)

test()
