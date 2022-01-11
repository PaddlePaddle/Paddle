"""
Copyright (c) 2021 Baidu.com, Inc. All Rights Reserved
This module provides an example for paddle.optimizer.lr.PolynomialDecay.
"""
import sys
import atheris_no_libfuzzer as atheris
import paddle
import os
from fuzz_util import Mutator, IgnoredErrors

# Switches for logging and ignoring errors.
LOGGING = os.getenv('PD_FUZZ_LOGGING') == '1'
IGNORE_ERRS = os.getenv('IGNORE_ERRS') == '1'


def TestOneInput(input_bytes):
    m = Mutator(input_bytes, LOGGING)

    learning_rate = m.float_range(0.0, 0.99, 'learning_rate')
    decay_steps = m.int_range(0, 10, 'decay_steps')
    end_lr = m.float_range(-10.0, 100.0, 'end_lr')
    power = m.float_range(-10.0, 100.0, 'power')
    cycle = m.bool('cycle')
    last_epoch = m.int_range(-10, 100, 'last_epoch')
    verbose = m.bool('verbose')

    x_dim1 = m.int_range(0, 20, 'x_dim1')
    x_dim2 = m.int_range(0, 20, 'x_dim2')
    x_val = m.float_list(x_dim1 * x_dim2, -10.0, 1000.0, 'x_val')
    x = m.tensor(x_val, 2, [x_dim1, x_dim2])

    if IGNORE_ERRS:
        try:
            # Example in Paddle doc.
            linear = paddle.nn.Linear(10, 10)
            scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=learning_rate, decay_steps=decay_steps,
                                                            end_lr=end_lr, power=power, cycle=cycle,
                                                            last_epoch=last_epoch, verbose=verbose)
            scheduler.get_lr()

            sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
            out = linear(x)
            loss = paddle.mean(out)
            loss.backward()
            sgd.step()
            sgd.clear_grad()
            scheduler.step()
        except IgnoredErrors:
            pass
    else:
        linear = paddle.nn.Linear(10, 10)
        scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=learning_rate, decay_steps=decay_steps,
                                                        end_lr=end_lr, power=power, cycle=cycle,
                                                        last_epoch=last_epoch, verbose=verbose)
        scheduler.get_lr()

        sgd = paddle.optimizer.SGD(learning_rate=scheduler, parameters=linear.parameters())
        out = linear(x)
        loss = paddle.mean(out)
        loss.backward()
        sgd.step()
        sgd.clear_grad()
        scheduler.step()


def main():
    atheris.Setup(sys.argv, TestOneInput, enable_python_coverage=True)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
