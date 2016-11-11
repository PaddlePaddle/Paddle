from paddle.trainer_config_helpers import *

dictionary = dict()
...  #  read dictionary from outside

define_py_data_sources2(
    train_list='train.list',
    test_list=None,
    module='sentimental_provider',
    obj='process',
    # above codes same as mnist sample.
    args={  # pass to provider.
        'dictionary': dictionary
    })
