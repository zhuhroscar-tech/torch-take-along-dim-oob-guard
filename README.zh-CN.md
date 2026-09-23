[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-take-along-dim-oob-guard

此独立仓库已合并到 [`torch-correctness-guards`](https://github.com/zhuhroscar-tech/torch-correctness-guards)。

后续请使用 umbrella package：

```bash
python -m pip install 'torch-correctness-guards[torch]'
torch-guard run take-along-dim-oob
```

Python API：

```python
from torch_correctness_guards import safe_take_along_dim
```

原有功能已保留为 umbrella package 中的 `take-along-dim-oob` guard，并与其他 PyTorch correctness diagnostics 一起维护。

本仓库仅作历史归档。新的修复和 release 会在 `torch-correctness-guards` 中进行。

[MIT 许可证](LICENSE)。
