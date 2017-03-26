# Use C-API for Model Inference

There are several examples in this directory about how to use Paddle C-API for model inference.

## Convert configuration file to protobuf binary.

Firstly, the user should convert Paddle's model configuration file into a protobuf binary file. In each example directory, there is a file named `convert_protobin.sh`. It will convert `trainer_config.conf` into `trainer_config.bin`.

The `convert_protobin.sh` is very simple, just invoke `dump_config` Python module to dump the binary file. The command line usages are:

```bash
python -m paddle.utils.dump_config YOUR_CONFIG_FILE 'CONFIG_EXTRA_ARGS' --binary > YOUR_CONFIG_FILE.bin
```

## Initialize paddle

```c++
char* argv[] = {"--use_gpu=False"};
paddle_init(1, (char**)argv);
```

We must initialize global context before we invoke other interfaces in Paddle. The initialize commands just like the `paddle_trainer` command line arguments.  `paddle train --help`,  will show the list of arguments. The most important argument is `use_gpu` or not.

## Load network and parameters

```c
paddle_gradient_machine machine;
paddle_gradient_machine_create_for_inference(&machine, config_file_content, content_size));
paddle_gradient_machine_load_parameter_from_disk(machine, "./some_where_to_params"));
```

The gradient machine is a Paddle concept, which represents a neural network can be forwarded and backward. We can create a gradient machine fo model inference, and load the parameter files from disk.

Moreover, if we want to inference in multi-thread, we could create a thread local gradient machine which shared the same parameter by using `paddle_gradient_machine_create_shared_param` API. Please reference `multi_thread` as an example.

## Create input

The input of a neural network is an `arguments`. The examples in this directory will show how to construct different types of inputs for prediction. Please look at `dense`, `sparse_binary`, `sequence` for details.

## Get inference

After invoking `paddle_gradient_machine_forward`, we could get the output of the neural network.  The `value` matrix of output arguments will store the neural network output values. If the output is a `SoftmaxActivation`, the `value` matrix are the probabilities of each input samples. The height of output matrix is number of sample. The width is the number of categories.
