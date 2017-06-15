from setuptools import setup, Distribution


class BinaryDistribution(Distribution):
    def has_ext_modules(foo):
        return True


setup(
    name='paddle_master',
    version='0.1',
    description='The client of the master server of PaddlePaddle.',
    url='https://github.com/PaddlePaddle/Paddle/go/master/python',
    author='PaddlePaddle Authors',
    author_email='paddle-dev@baidu.com',
    license='Apache 2.0',
    packages=['paddle_master'],
    package_data={'master': ['libmaster.so'], },
    distclass=BinaryDistribution)
