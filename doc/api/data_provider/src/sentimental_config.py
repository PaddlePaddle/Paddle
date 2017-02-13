from paddle.trainer_config_helpers import *

dictionary = dict()
...  #  read dictionary from outside

setup_data_provider(
    'train.list',
    None,
    'sentimental_provider',
    'process',
    # above codes same as mnist sample.
    args={  # pass to provider.
        'dictionary': dictionary
    })
