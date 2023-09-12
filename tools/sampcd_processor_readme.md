# 将 xdoctest 引入到飞桨框架工作流中（补充） - 详细设计

|领域 | 将 xdoctest 引入到飞桨框架工作流中 |
|---|--------------------------------|
|提交作者 | megemini (柳顺) |
|提交时间 | 2023-07-16 |
|版本号 | V1.1 |
|依赖飞桨版本 | develop 分支 |
|文件名 | sampcd_processor_readme.md |


# 概述

本文为 [《将 xdoctest 引入到飞桨框架工作流中》](https://github.com/PaddlePaddle/community/pull/547) 的补充，主要介绍引入 `xdoctest` 后使用 `Doctester` 以及 `Xdoctester` 的详细设计，以及对原有代码测试 `sampcd_processor.py` 的重构。

本文涉及以下文件：

- `sampcd_processor_utils.py` : 代码检查的相关工具
- `sampcd_processor_xdoctest.py` : `Xdoctester` 的相关实现
- `sampcd_processor.py` : 原代码检查工具
- `test_sampcd_processor.py` : 原代码检查工具单元测试
- `test_sampcd_processor_xdoctest.py` : `Xdoctester` 单元测试

# 总体设计

在 [《将 xdoctest 引入到飞桨框架工作流中》](https://github.com/PaddlePaddle/community/pull/547) 一文中，将代码检查分为：

- 接口抽取
- 示例执行
- 结果比对

三个主要阶段，引入 `xdoctest` 后，以上三个阶段的分工为：

- 接口抽取 : 沿用原流程 -> `sampcd_processor_utils.py`
- 示例执行 : 使用 `xdoctest` -> `sampcd_processor_xdoctest.py`
- 结果比对 : 使用 `xdoctest` -> `sampcd_processor_xdoctest.py`

具体实现步骤为(参考 `sampcd_processor_utils.py` 的 `run_doctest` 函数)：

1. `init_logger(debug=args.debug, log_file=args.logf)`
    日志初始化

2. `run_on_device = check_test_mode(mode=args.mode, gpu_id=args.gpu_id)`
    检查测试模式

3. `sample_code_test_capacity = get_test_capacity(run_on_device)`
    获取测试环境

4. `docstrings_to_test, whl_error = get_docstring(full_test=args.full_test)`
    抽取测试 docstring

5. `doctester.prepare(sample_code_test_capacity)`
    准备 doctester

6. `test_results = get_test_results(doctester, docstrings_to_test)`
    运行代码检查

7. `doctester.print_summary(test_results, whl_error)`
    打印检查结果

8. `exec_gen_doc()` 可选
    生成文档

其中步骤 `1` `2` `3` `4` 沿用原代码检查逻辑， `5` `6` 为使用 `Xdoctester` 进行代码检查与结果比对， `7` `8` 沿用原代码检查逻辑。

由于需要兼容目前的代码检查，将原有工具进行重构：

- 修改 `sampcd_processor.py`:

    - 将 docstring 抽取以及此流程之前的函数，抽取为公共函数，移到 `sampcd_processor_utils.py` 中。

    - 重新从 `sampcd_processor_utils.py` 中引入这些公共函数。

    - 增加 `is_ps_wrapped_codeblock` 函数，判断是否是 `>>> ` 的示例代码。

    - 修改 `sampcd_extract_to_file`，对于 `is_ps_wrapped_codeblock` 的代码不做检查。

    - 在 `if __name__ == "__main__"` 最后的执行部分，对于没有抽取到代码，不做 `sys.exit(0)`，因为后续还需要 `xdoctest` 的检查。

    - 在 `if __name__ == "__main__"` 最后的执行部分，增加 `xdoctest` 的检查。

    - 在 `if __name__ == "__main__"` 最后的执行部分，移除 `exec_gen_doc` 方法，在 `xdoctest` 最后一起调用。

- 增加 `sampcd_processor_utils.py`

    - 增加 docstring 抽取以及此流程之前的函数，以及 args 与一些常量。移除可变 `global`，部分函数有些许修改，整体逻辑不变。

    - 增加基础类 `TestResult` 与 `Doctester`。

    - 增加 `run_doctest` 函数以及内部调用的其他函数，作为 doctest 的总入口。

- 增加 `sampcd_processor_xdoctest.py`

    - 增加 `Xdoctester`，是 `xdoctest` 的 `Doctester` 实现。

    - 增加 `if __name__ == "__main__"`，使其可以单独运行。

# 代码检查 `Doctester`

此方案中引入 `Doctester` 作为代码检查的基类，主要出于以下考虑：

- 原代码检查工具的 python 代码内部耦合较严重，如：

    - 内部逻辑绑定，`get_filenames` 只能用于原代码抽取。

    - 使用可变的 `global` 变量，状态跟踪困难。

    - 检查逻辑遵从原代码检查的逻辑，插入新方法会破坏原逻辑。

    导致在其上添加 `xdoctest` 会进一步恶化代码的可维护性。

- 引入 `Doctester` 可以分离 docstring 的抽取与代码检查的逻辑，从而方便引入 python 原生 `doctest` 或者 `xdoctest`，以及未来其他的代码检查工具。

## `Doctester` 的属性与方法

具体请参考代码中的注释，这里简单说明。

### 属性

#### `style`

代码检查服从的样式，如 `google`, `freeform`。

注意，Paddle 目前的代码块是在 `.. code-block:: python` 中，而 `doctest` 或 `xdoctest` 只关心是否有 PS1 (>>> ) 的包裹，`google` 样式则是只检查 `Examples:` 中的代码。这是目前主流的代码检查工具与 Paddle 不同的地方，所以，需要沿用 Paddle 目前的 `codeblock` 抽取过程。

#### `target`

代码检查的输入是 `codeblock` 还是 `docstring`，目前 Paddle 主要以 `codeblock` 为检查单元。

结合 `style` 参数，目前合适的方式为：

- `style = freeform`

- `target = codeblock`

也就是说，抽取 `codeblock` 作为检查单元，而其中只要使用 `>>> ` 或 `... ` 包裹的部分即为代码。

这里补充说明一下：

- 为什么不能用 `style = freeform` `target = docstring` 的模式

    因为，目前 Paddle 中存在 `.. code-block:: text` 等代码部分，这里面的代码大多只是描述或者说明，不需要保证其正确性，而如果其中代码包裹了 `>>> `，就会被 `xdoctest` 捕获，从而报错。

- 为什么不能用 `style = google` `target = docstring` 的模式

    因为，目前 Paddle 在 `Examples:` 之外的部分，也存在 `.. code-block:: python` 需要检查的代码。

- 为什么不能用 `style = google` `target = codeblock` 的模式

    可以，`Doctester` 中的 `ensemble_docstring` 方法可以将 `codeblock` 转为含有 `Examples:` 的 `docstring` 样式，但是，多此一举。

- 既然只有一种合适的模式，那么为什么要做这么多选择？

    简单说，为了以后的扩展与维护。如，以后不使用 `.. code-block::` 等情况。

#### `directives`

`Doctester` 支持的指令可以保存在此变量中。目前主要的作用是列举所支持的指令列表，帮助进行指令的转换，未来可以做指令检查、指令映射等。

这里说明一下后续建议的示例代码书写格式。

- 示例代码写在 `.. code-block:: python` 内部。

- 以 `>>> ` 表示代码开始，以 `... ` 表示代码的延续。

- 在 `>>> ` 和 `... ` 后面紧接的一行，如果没有上述两个提示符，则表示代码输出。

- 在代码中，以 `# doctest：` 表示测试指令。

- 以至少一个空行表示代码段结束。

- 其他没有提示符的地方为说明文字。

这里需要特别注意，所有代码的缩进需要统一。

正确的代码段，如：

``` python
def something():
    """ Function summary ...
    Some description ...

    .. code-block:: python
        :name: code-example-0

        this is some blabla...

        >>> # doctest: +SKIP
        >>> print(1+1)
        2

    Examples:

        .. code-block:: python
            :name: code-example-1

            this is some blabla...

            >>> # doctest: +REQUIRES(env:GPU, env:XPU)
            >>> for i in range(2):
            ...     print(i)
            0
            1
    """
```

错误的代码段，如， 没有正确使用 `.. code-block:: python`：

``` python
def something():
    """ Function summary ...
    Some description ...

    >>> # doctest: +SKIP
    >>> print(1+1)
    2

    Examples:

        .. code-block:: python
            :name: code-example-1

            this is some blabla...

            >>> # doctest: +REQUIRES(env:GPU, env:XPU)
            >>> for i in range(2):
            ...     print(i)
            0
            1
    """
```

错误的代码段，如， 没有正确缩进：

``` python
def something():
    """ Function summary ...
    Some description ...

    .. code-block:: python
        :name: code-example-0

        this is some blabla...

        >>> # doctest: +SKIP
        >>> print(1+1)
       2

    Examples:

        .. code-block:: python
            :name: code-example-1

            this is some blabla...

            >>> # doctest: +REQUIRES(env:GPU, env:XPU)
            >>> for i in range(2):
            ...     print(i)
           0
           1
    """
```

错误的代码段，如，使用特定代码检查工具的指令：


``` python
def something():
    """ Function summary ...
    Some description ...

    .. code-block:: python
        :name: code-example-0

        this is some blabla...

        >>> # xdoctest: +SKIP
        >>> print(1+1)
        2

    Examples:

        .. code-block:: python
            :name: code-example-1

            this is some blabla...

            >>> # xdoctest: +REQUIRES(env:GPU, env:XPU)
            >>> for i in range(2):
            ...     print(i)
            0
            1
    """
```

这里特别说明：

- 不建议使用特定检查工具的指令，如 `# xdoctest: +SKIP` 等。

    因为，特定的指令会绑定特定的检查工具，由于示例代码的修改工作量较大，如果后续不使用此工具了，则可能需要重新大面积的修改示例代码。

    所以，这里建议，Paddle 统一制定一套代码检查的指令，再利用 `Doctester` 的 `convert_directive` 方法，在每次检查的时候，动态修改指令为此次测试工具需要的指令样式。

    结合 python 原生的 `doctest` 与 `xdoctest` 工具的指令样式，这里建议指令样式为：

    ```
    directive             ::=  "#" "doctest:" directive_option
    directive_option      ::=  on_or_off directive_option_name [env_option]
    on_or_off             ::=  "+" | "-"
    directive_option_name ::=  "SKIP" | "REQUIRES" | ...
    env_option            ::=  "(" env_entity ("," env_entity)* ")"
    env_entity            ::=  "env:" env
    env                   ::=  "CPU" | "GPU" | "XPU" | "DISTRIBUTED" | ...
    ```

    此样式与 `xdoctest` 的指令样式主要不同是，使用 `doctest` 代替 `xdoctest`。

    特别需要注意其中的大小写，正确的指令如：

    - `# doctest: +SKIP`
    - `# doctest: +REQUIRES(env:GPU)`
    - `# doctest: +REQUIRES(env:GPU, env:XPU)`

    错误的指令如：

    - `# xdoctest: +SKIP` 使用错误的前缀
    - `# doctest: +REQUIRES(env:gpu)` 使用错误的小写
    - `# doctest: + REQUIRES(env:GPU)` 使用错误的空格


    `doctest`，`xdoctest`，Paddle 的指令关系为：

    - `doctest` 为最小子集

    - `xdoctest` 为 `doctest` 的超集，指令前缀由 `doctest` 改为 `xdoctest`

    - Paddle 与 `xdoctest` 基本一致，指令前缀由 `xdoctest` 改为 `doctest`

    也就是说，尽量兼容 python 原生指令样式，并做扩展。

>
> **参考**
> `doctest` 的指令定义[如下](https://docs.python.org/3/library/doctest.html#directives):
> ```
> directive             ::=  "#" "doctest:" directive_options
> directive_options     ::=  directive_option ("," directive_option)*
> directive_option      ::=  on_or_off directive_option_name
> on_or_off             ::=  "+" | "-"
> directive_option_name ::=  "DONT_ACCEPT_BLANKLINE" | "NORMALIZE_WHITESPACE" | ...
> ```
>


- 建议使用 python 的控制台编写并复制代码。

    python 的控制台默认以 `>>> ` 作为 PS1，这样可以最大化兼容性。

    也可以使用 `ipython`，但拷贝代码之后需要手动修改 PS1。

- 建议执行代码之前，执行 `>>> paddle.device.set_device('cpu')`，代码检查工具中已默认执行此命令。

    这样可以统一 `tensor` 的 `place` 为 `Place(cpu)`，如果需要 `gpu` 等，请显性的在示例代码中设置，并添加指令，如：

    ```python
    >>> import paddle
    >>> a = paddle.to_tensor(0.1)
    >>> print(a)
    Tensor(shape=[1], dtype=float32, place=Place(cpu), stop_gradient=True,
    [0.10000000])

    >>> # doctest: +REQUIRES(env:GPU)
    >>> paddle.device.set_device('gpu')
    >>> a = paddle.to_tensor(0.1)
    >>> print(a)
    Tensor(shape=[1], dtype=float32, place=Place(gpu:0), stop_gradient=True,
    [0.10000000])
    ```

最后，使用上述的代码书写格式与指令格式，如果后续需要改变示例样式也相对简单，如，需要改成不使用 PS 的示例代码，则只需要去掉 PS1/PS2，并 comment 其他部分即可。

### 方法

#### `ensemble_docstring`

将 `codeblock` 包装为 `docstring`，如，添加 `Examples:` 在字符串的开头，并在每行前添加缩进。

此方法主要是将，非 `google` 样式的代码段，转为 `google` 样式使用。

#### `convert_directive`

将 docstring 中的检查指令，转换为当前工具的样式。如，将 `# doctest: +SKIP` 转换为 `# xdoctest: +SKIP`。

#### `prepare`

根据当前的测试环境进行一些设置，如，`xdoctest` 需要 `os.environ` 进行 `REQUIRES` 的判断，则可以在此方法中进行设置。

这里对于 `xdoctest` 需要 `gpu` 等，只是简单的设置 `os.environ['GPU'] = "True"`。如果存在环境变量冲突，需要重新设计。

另外，此处的变量名大小写需要与指令中的一致，如 `# doctest: +REQUIRES(env:GPU)`。

#### `run`

运行代码检查。

#### `print_summary`

打印出检查的结果。由于 `xdoctest` 中对于检查结果的返回样式与当前返回的不太相同，如，如果不满足 `REQUIRES` 则直接 skip，没有返回是由于什么 skip，所以，这里将 `print_summary` 作为 `Doctester` 的方法，而不是一个单独的函数。

# 其他类与方法

## TestResult

这里只是简单的将测试结果做一个封装，后续有其他需求可以再扩展。

## Xdoctester

`xdoctest` 的 `Doctester` 实现。基本逻辑符合 `Doctester` 的约定，这里只简单说明两个参数：

- `mode='native'`

    这是 `xdoctest` 的检查模式，还可以是 `pytest`，但是这里没有用到，只是留个传参的入口。

- `verbose=2`

    `0` 基本没什么输出，`1` 会输出简单的检查通过与否，`2` 可以输出具体错误的地方。

    这里先设置为 `2`，后续程序运行稳定了可以慢慢降级。

## 一些保留的函数

- `get_api_md5`
- `get_incrementapi`
- `get_full_api_by_walk`
- `get_full_api_from_pr_spec`
- `get_full_api`
- `extract_code_blocks_from_docstr`
- `get_test_capacity`
- `exec_gen_doc`
- `parse_args`
- `get_filenames` -> `get_docstring`

# 最后

## 当前检查代码的移除

如果后续需要移除当前原有的代码检查，可以：

- 移除 `sampcd_processor.py`
- 将 `sampcd_processor_xdoctest.py` 改名为 `sampcd_processor.py`
- 移除 `test_sampcd_processor.py`，可以保留部分测试函数。

## Paddle docs 需要注意

目前 Paddle docs 对于 `>>> ` 代码的处理是，strip 掉此提示符，然后交给原有代码检查工具进行检测。这种方法在大部分情况下没什么问题，但是，如果代码中有 `requires` 项，则可能检查失败。所以，后续需要修改 Paddle docs 的检查逻辑，建议对于 `>>> ` 直接跳过，与当前 Paddle 的 `sampcd_processor.py` 一致。最后收尾的时候，移除掉 Paddle docs 的代码检查。

# 参考资料

- doctest — Test interactive Python examples, https://docs.python.org/3/library/doctest.html#module-doctest
- Xdoctest - Execute Doctests, https://xdoctest.readthedocs.io/en/latest/index.html
