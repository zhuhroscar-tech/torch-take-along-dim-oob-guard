[![English](https://img.shields.io/badge/English-555555?style=flat)](README.md) [![简体中文](https://img.shields.io/badge/简体中文-555555?style=flat)](README.zh-CN.md)

# torch-take-along-dim-oob-guard

A call-site workaround and diagnostic for a real `torch.take_along_dim(input, indices, dim=<int>)` correctness bug: when `dim` is an explicit integer, an out-of-bounds index is **silently wrapped modulo the dimension size** instead of raising, producing a plausible-looking wrong result with no error or warning. Upstream reference: [pytorch/pytorch#196106](https://github.com/pytorch/pytorch/issues/196106) ("take_along_dim(..., dim=...) silently wraps out-of-bounds indices instead of raising"), opened 2026-09-05, currently **open and triaged** (labels: `triaged`, `module: advanced indexing`, `module: correctness (silent)`, `release triage`) with one contributor's stated intent to fix it but no merged PR as of this repo's creation — independently re-checked via `gh api`/`gh search prs`, not trusted from a stale search snippet.

```python
import torch

x = torch.tensor([10., 20., 30.])          # size 3 along dim 0 (valid: 0..2 or -3..-1)
idx = torch.tensor([9])                     # out of bounds

torch.take_along_dim(x, idx, dim=0)         # -> tensor([10.])   (silently returns x[9 % 3])
torch.gather(x, 0, idx)                     # RuntimeError: index 9 is out of bounds ... size 3
torch.take(x, idx)                          # IndexError: out of range ...
x[idx]                                       # IndexError: index 9 is out of bounds ...
torch.index_select(x, 0, idx)               # IndexError: index out of range in self
torch.take_along_dim(x, idx, dim=None)      # RuntimeError: index 9 is out of bounds ... (dim=None path is correct)
```

Every sibling indexing op, and `take_along_dim`'s own `dim=None` (flattened) path, correctly raise on the exact same out-of-bounds index. Only the `dim=<int>` path silently wraps. The wrap affects large positive indices (`idx=100`, `idx=2**40`) and negative indices past `-size` (`idx=-4` when `size=3`), in any number of dimensions — confirmed independently on this repo's own CI against the currently installed torch build, not just trusted from the upstream issue's reported numbers. Valid negative indices (`-1..-size`) are correct today and remain correct through this guard; the bug is specifically about indices genuinely outside `[-size, size - 1]`.

## Install and check

Requires Python 3.9+ and a compatible PyTorch installation (`torch>=2.0` in the optional extra; NumPy is used as the correctness oracle for valid cases).

```bash
git clone https://github.com/zhuhroscar-tech/torch-take-along-dim-oob-guard.git
cd torch-take-along-dim-oob-guard
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[torch]"
torch-take-along-dim-oob-guard
torch-take-along-dim-oob-guard --json
```

The CLI reruns 9 cases (1D and 2D, out-of-bounds and valid, `dim=0`/`dim=1`/`dim=None`) against the currently installed torch build, comparing the native op's behavior against a NumPy oracle (`np.take_along_axis`) for valid cases and against the shared raise-on-OOB contract for invalid ones. Its JSON includes the installed torch version, per-case native/guard verdicts, `any_native_silently_wrong`, and `guard_fully_correct`.

Exit codes describe the **guard check**, not just native bug detection: `0` means every guard case matched its expected contract, `1` means a guard check failed, and `2` means torch could not be imported.

## Use in Python

```python
from torch_take_along_dim_oob_guard import safe_take_along_dim

x = torch.tensor([10., 20., 30.])
safe_take_along_dim(x, torch.tensor([-1]), dim=0)  # tensor([30.]) -- valid, unaffected
safe_take_along_dim(x, torch.tensor([9]),  dim=0)  # raises IndexError instead of silently returning 10.
```

`safe_take_along_dim` (and its factory `make_safe_take_along_dim(torch)`, for binding to a specific torch module) range-checks `indices` against the real size of `input` along `dim` *before* delegating to the native op, raising `IndexError` in the same style as `gather`/`index_select` only when an index is genuinely outside `[-size, size - 1]`. When `dim is None` it delegates straight to the native op unchanged, since that path is already correct — no double-guarding of a path that was never broken.

## Scope and limitations

- This tool does **not** patch PyTorch itself. You apply the guard function explicitly at your own `take_along_dim(..., dim=<int>)` call sites, the same way you would apply any other workaround.
- The guard computes `indices.min()`/`indices.max()` to range-check before delegating, which is a small O(n) pass over the index tensor in addition to the underlying op's own cost — negligible for the sizes this op is normally used at (index tensors, not the full gathered data), but not free.
- Reproduced and verified on this development host's CPU path (macOS arm64). CUDA index-kernel behavior is not independently tested here (no CUDA on this host); the guard's range-check logic is backend-independent (pure Python/tensor `.item()` reads), but that specific claim is unverified on CUDA.
- Reproduced and verified only against **torch 2.14.0**. If a future released torch version fixes pytorch/pytorch#196106, `any_native_silently_wrong` should report `False` on that version, and this guard remains a safe no-op fallback — this repo's own CI re-verifies against whatever `torch>=2.0` pip actually resolves to at build time.
- This guard is specific to the pattern in the upstream issue (an integer `dim` argument with an out-of-range index). It is not a general audit of every PyTorch indexing operation's bounds-checking behavior.

## Development

```bash
python -m pip install -e ".[dev,torch]"
python -m pytest -v --cov=torch_take_along_dim_oob_guard
```

See [implementation](src/torch_take_along_dim_oob_guard/core.py), [tests](tests/test_core.py), and [CI config](.github/workflows/ci.yml). Licensed under [MIT](LICENSE).
