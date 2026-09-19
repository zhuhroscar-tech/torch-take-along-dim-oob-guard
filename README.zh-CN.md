[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-take-along-dim-oob-guard

针对 `torch.take_along_dim(input, indices, dim=<int>)` 的一个真实正确性 bug 的调用点绕过方案与诊断工具：当 `dim` 是一个显式整数时，越界索引会被**悄悄按维度大小取模**，而不是抛出异常，从而产生一个看起来合理但实际错误的结果，且没有任何错误或警告。上游参考：[pytorch/pytorch#196106](https://github.com/pytorch/pytorch/issues/196106)（"take_along_dim(..., dim=...) silently wraps out-of-bounds indices instead of raising"），于 2026-09-05 提出，目前状态为**开放且已分类**（标签：`triaged`、`module: advanced indexing`、`module: correctness (silent)`、`release triage`），有一位贡献者表示有意修复，但截至本仓库创建时尚无已合并的 PR——已通过 `gh api`/`gh search prs` 独立复核，而非直接信任一份可能过期的搜索摘要。

```python
import torch

x = torch.tensor([10., 20., 30.])          # 沿 dim 0 大小为 3（有效范围：0..2 或 -3..-1）
idx = torch.tensor([9])                     # 越界

torch.take_along_dim(x, idx, dim=0)         # -> tensor([10.])   （悄悄返回 x[9 % 3]）
torch.gather(x, 0, idx)                     # RuntimeError: index 9 is out of bounds ... size 3
torch.take(x, idx)                          # IndexError: out of range ...
x[idx]                                       # IndexError: index 9 is out of bounds ...
torch.index_select(x, 0, idx)               # IndexError: index out of range in self
torch.take_along_dim(x, idx, dim=None)      # RuntimeError: index 9 is out of bounds ...（dim=None 路径是正确的）
```

每一个同类索引操作，以及 `take_along_dim` 自身的 `dim=None`（展平）路径，在完全相同的越界索引上都会正确地抛出异常。只有 `dim=<int>` 路径会悄悄取模包裹。该问题影响较大的正索引（`idx=100`、`idx=2**40`）以及超出 `-size` 的负索引（当 `size=3` 时 `idx=-4`），且在任意维度下都会出现——已在本仓库自己的 CI 上针对当前安装的 torch 版本独立复现确认，而非仅仅信任上游 issue 报告的数字。有效的负索引（`-1..-size`）目前行为正确，经过本防护后依然保持正确；该 bug 特指那些真正超出 `[-size, size - 1]` 范围的索引。

## 安装与检查

需要 Python 3.9+ 以及兼容的 PyTorch 安装（可选依赖中的 `torch>=2.0`；NumPy 用作有效用例的正确性 oracle）。

```bash
git clone https://github.com/zhuhroscar-tech/torch-take-along-dim-oob-guard.git
cd torch-take-along-dim-oob-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-take-along-dim-oob-guard
torch-take-along-dim-oob-guard --json
```

CLI 会针对当前安装的 torch 版本重新运行 9 个用例（1D 与 2D、越界与有效、`dim=0`/`dim=1`/`dim=None`），对有效用例将原生操作的行为与 NumPy oracle（`np.take_along_axis`）比对，对无效用例则与共享的"越界即抛出"契约比对。JSON 输出包含已安装的 torch 版本、每个用例的原生/防护判定、`any_native_silently_wrong` 以及 `guard_fully_correct`。

退出码描述的是**防护检查**的结果，而不仅是原生 bug 检测：`0` 表示所有防护用例都符合预期契约，`1` 表示至少一个防护检查失败，`2` 表示无法导入 torch。

## 在 Python 中使用

```python
from torch_take_along_dim_oob_guard import safe_take_along_dim

x = torch.tensor([10., 20., 30.])
safe_take_along_dim(x, torch.tensor([-1]), dim=0)  # tensor([30.]) —— 有效，不受影响
safe_take_along_dim(x, torch.tensor([9]),  dim=0)  # 抛出 IndexError，而不是悄悄返回 10.
```

`safe_take_along_dim`（以及用于绑定特定 torch 模块的工厂函数 `make_safe_take_along_dim(torch)`）会在委托给原生操作*之前*，根据 `input` 沿 `dim` 的真实大小对 `indices` 做范围检查，仅当索引真正超出 `[-size, size - 1]` 时，才以与 `gather`/`index_select` 相同的风格抛出 `IndexError`。当 `dim is None` 时，会原样委托给原生操作——因为该路径本身已经是正确的，不对一个从未出问题的路径做重复防护。

## 适用范围与局限性

- 本工具**不会**对 PyTorch 本身打补丁。你需要像使用其他绕过方案一样，在自己的 `take_along_dim(..., dim=<int>)` 调用点显式应用该防护函数。
- 该防护在委托给原生操作之前会计算 `indices.min()`/`indices.max()` 进行范围检查，这是在底层操作自身开销之外，对索引张量的一次小型 O(n) 遍历——对于该操作通常使用的规模（索引张量，而非完整的被收集数据）而言可以忽略不计，但并非零开销。
- 已在本开发主机的 CPU 路径（macOS arm64）上复现并验证。CUDA 索引内核的行为未在此独立测试（本机无 CUDA）；防护的范围检查逻辑与后端无关（纯 Python/张量 `.item()` 读取），但该具体说法在 CUDA 上尚未验证。
- 仅针对 **torch 2.14.0** 复现并验证了防护效果。如果未来某个已发布的 torch 版本修复了 pytorch/pytorch#196106，`any_native_silently_wrong` 在该版本上应报告为 `False`，此时本防护仍是一个安全的、等价于无操作的兜底——本仓库自身的 CI 会持续针对构建时 pip 实际解析到的 `torch>=2.0` 版本重新验证。
- 本防护专门针对上游 issue 中的模式（整数类型的 `dim` 参数加上越界索引）。它并非对 PyTorch 所有索引操作边界检查行为的通用审计。

## 开发

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_take_along_dim_oob_guard
```

参见[实现代码](src/torch_take_along_dim_oob_guard/core.py)、[测试](tests/test_core.py)与 [CI 配置](.github/workflows/ci.yml)。基于 [MIT](LICENSE) 许可证。
