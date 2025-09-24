# 开发者指南

Trinity-RFT 将 RL 训练过程拆分为了三个模块：**Explorer**、**Trainer** 和 **Buffer**。每个模块都提供了扩展接口，开发者可以基于这些接口实现自己的功能模块，从而实现对 Trinity-RFT 的定制化开发。

下表中列出了每个模块的主要功能、对应的扩展接口以及开发目标。开发者可参考对应模块的开发教程，根据自己的需求选择性地进行扩展。

| 模块     | 主要功能                                     | 扩展接口     | 开发目标         |  教程链接                   |
|--------|----------------------------------------------|-------------|------------------------|----------------------------|
| Explorer | 负责 Agent-Environment 交互，生成轨迹数据     | `Workflow`   | 将现有 RL 算法拓展到新场景  | [🔗](./develop_workflow.md) |
| Trainer  | 负责模型训练和更新                           | `Algorithm`  | 设计新的 RL 算法           | [🔗](./develop_algorithm.md) |
| Buffer   | 负责任务以及生成的轨迹数据的存储和预处理        | `Operator`   | 设计新的数据清洗、增强策略   | [🔗](./develop_operator.md) |

```{tip}
Trinity-RFT 提供了插件化的开发方式，可以在不修改框架代码的前提下，灵活地添加自定义模块。
开发者可以将自己编写的模块代码放在 `trinity/plugins` 目录下。Trinity-RFT 会在运行时自动加载该目录下的所有 Python 文件，并注册其中的自定义模块。
Trinity-RFT 也支持在运行时通过设置 `--plugin-dir` 选项来指定其他目录，例如：`trinity run --config <config_file> --plugin-dir <your_plugin_dir>`。
```

对于准备向 Trinity-RFT 提交的模块，请遵循以下步骤：

1. 在适当目录中实现你的代码，例如 `trinity/common/workflows` 用于 `Workflow`，`trinity/algorithm` 用于 `Algorithm`，`trinity/buffer/operators` 用于 `Operator`。

2. 在目录对应的 `__init__.py` 文件中注册你的模块。

3. 在 `tests` 目录中为你的模块添加测试，遵循现有测试的命名约定和结构。

4. 提交代码前，确保通过 `pre-commit run --all-files` 完成代码风格检查。

5. 向 Trinity-RFT 仓库提交 Pull Request，在描述中详细说明你的模块功能和用途。
