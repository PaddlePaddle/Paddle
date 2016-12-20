import py_paddle.swig_paddle as api
from paddle.trainer.config_parser import parse_config


def main():
    api.initPaddle("-use_gpu=false", "-trainer_count=4")  # use 4 cpu cores
    config = parse_config('simple_mnist_network.py', '')
    m = api.GradientMachine.createFromConfigProto(config.model_config)


if __name__ == '__main__':
    main()
