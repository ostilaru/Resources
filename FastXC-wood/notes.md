# FastXC 代码结构

## 1. fastxc.py
这段代码定义了一个名为 `FastXC` 的类，该类用于处理地震数据的交叉相关（Cross-Correlation）计算。FastXC 类的实例化需要一个配置文件路径作为参数，该配置文件包含了所有必要的参数和命令。

* `__init__`：这是类的初始化方法，它接受一个配置文件路径作为参数。该方法会解析配置文件并检查其内容，然后将解析得到的结果赋值给类的属性。

* `generate_filter`：这个方法用于设计滤波器。它调用 `design_filter` 函数，该函数接受参数和内存信息作为输入。

* `generate_sac2spec_list_dir`：这个方法用于生成 SAC 到谱的列表目录。它调用 `gen_sac2spec_list_dir` 函数，该函数接受两个地震数组、参数和内存信息作为输入。

* `generate_sac2spec_cmd`：这个方法用于生成 SAC 到谱的命令。它调用 `gen_sac2spec_cmd` 函数，该函数接受命令和参数作为输入。

* `deploy_sac2spec_cmd`：这个方法用于部署 SAC 到谱的命令。它调用 `sac2spec_cmd_deployer` 函数，该函数接受参数和执行信息作为输入。

* `generate_xc_list_dir`：这个方法用于生成交叉相关（Cross-Correlation）列表目录。它调用 `gen_xc_list_dir` 函数，该函数接受参数、内存信息和一个地震数组作为输入。

* `generate_xc_cmd`：这个方法用于生成交叉相关命令。它调用 `gen_xc_cmd` 函数，该函数接受命令、参数和一个地震数组作为输入。

* `deploy_xc_cmd`：这个方法用于部署交叉相关命令。它调用 `xc_cmd_deployer` 函数，该函数接受参数、内存信息和执行信息作为输入。

* `generate_stack_list_dir`：这个方法用于生成堆叠（Stacking）列表目录。它调用 `gen_stack_list_dir` 函数，该函数接受参数、执行信息和一个地震数组作为输入。

* `generate_stack_cmd`：这个方法用于生成堆叠命令。它调用 `gen_stack_cmd` 函数，该函数接受命令和参数作为输入。

* `deploy_stack_cmd`：这个方法用于部署堆叠命令。它调用 `stack_cmd_deployer` 函数，该函数接受参数和执行信息作为输入。

* `generate_rotate_list_dir`：这个方法用于生成旋转（Rotating）列表目录。它调用 `gen_rotate_list_dir` 函数，该函数接受两个地震数组和参数作为输入。

* `generate_rotate_cmd`：这个方法用于生成旋转命令。它调用 `gen_rotate_cmd` 函数，该函数接受命令和参数作为输入。

* `deploy_rotate_cmd`：这个方法用于部署旋转命令。它调用 `rotate_cmd_deployer` 函数，该函数接受参数和执行信息作为输入。

`generate_template_config` 为静态方法，该方法用于生成一个模板配置文件：

首先，它获取当前文件的绝对路径，并使用 `os.path.dirname` 和 `os.path.abspath` 函数来获取当前文件所在的目录。这两个函数的作用是，`os.path.dirname` 返回路径的目录部分，`os.path.abspath` 返回路径的绝对路径。

然后，它使用 `os.path.join` 函数将当前目录和 `"template_config.ini"` 连接起来，生成模板文件的完整路径。`os.path.join` 函数的作用是连接两个或更多的路径部分。

接着，它使用 `shutil.copy2` 函数将模板文件复制到指定的输出路径。`shutil.copy2` 函数的作用是复制文件和元数据，如果目标是一个目录，那么它会将源文件复制到该目录下，并保持原文件名。

`FastXC` 类的 `run` 方法，该方法按顺序运行整个地震数据处理流程:

`generate filter` -> `sac2spec` -> `cross correlation` -> `stack` -> `rotate`    