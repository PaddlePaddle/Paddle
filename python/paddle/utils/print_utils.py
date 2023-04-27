def print_args(func):
    def inner(*args, **kwargs):
        function_name = func.__name__
        from paddle.jit.dy2static.utils import parse_arg_and_kwargs
        api_params, api_defaults = parse_arg_and_kwargs(func)
        import collections
        inputs = collections.OrderedDict()
        params = collections.OrderedDict()  
        from paddle import Tensor
        # from paddle import Tensor
        for i in range(len(args)):
            if isinstance(args[i], Tensor):
                inputs[api_params[i]] = args[i]
            elif(isinstance(args[i], (list,tuple)) and len(args[i]) > 0 and isinstance(args[i][0], Tensor)):
                inputs[api_params[i]] = args[i]
            else:
                params[api_params[i]] = args[i]
        for key, value in kwargs.items():
            if type(value) == Tensor:
                inputs[key] = value
            else:
                params[key] = value
        log_msg = "{{function_name : {function_name}, ".format(function_name=function_name)
        log_msg += "inputs: { "
        for name,value in inputs.items():
            shape = []
            if isinstance(value, (list,tuple)) and len(value) > 0 and isinstance(value[0], Tensor):
                for v in value:
                    shape.append(v.shape)
                input_type="List(Tensor)"
            else:
                shape=value.shape
                input_type=str(type(value))
            log_msg += "{{ {name}, type: {input_type}, shape: {shape} }}, ".format(name=name, input_type=input_type, shape=shape)
        log_msg += "}, "
        log_msg += "params: [ "
        for name,value in params.items():
            log_msg += "{name}: {value}, ".format(name=name, value=str(value))
        log_msg += "]}"
        print(log_msg, flush=True)
        return func(*args, **kwargs)
    return inner