class TrainConfig(object):

    # Whether to use GPU in training or not.
    use_gpu = False

    # The training batch size.
    batch_size = 4

    # The epoch number.
    num_passes = 30

    # The global learning rate.
    learning_rate = 0.01

    # Training log will be printed every log_period.
    log_period = 100
    
    # input data dim
    input_dim = 46
