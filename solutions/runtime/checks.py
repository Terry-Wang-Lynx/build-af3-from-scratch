"""Shared infrastructure for per-module unit tests.

每个模块单元测试的共享基础设施。

The pattern mirrors how an exam works: we replace the module's parameters
with a deterministic ramp (``torch.linspace(-1, 1, numel)``) so every
correct implementation produces the same output, given the same inputs.
The expected output is computed once with the reference solution and
saved as a ``.pt`` file under ``<chapter>/control_values/``. Students
then run ``test_module_forward`` (etc.) on their own implementation —
which loads the same ``.pt`` and asserts ``torch.allclose``.

这一套与考试的逻辑一样: 我们把模块参数临时替换成一段固定的
``torch.linspace(-1, 1, numel)``，这样只要实现正确，任何人在相同输入
下都会得到完全相同的输出。期望输出由参考解算一次，保存为
``<chapter>/control_values/`` 下的 ``.pt`` 文件；学生在自己的实现上调用
``test_module_forward`` 就能用 ``torch.allclose`` 验证一致性。

"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable

import torch
from torch import nn


# ----------------------------------------------------------------------------
# Controlled execution
# 受控执行
# ----------------------------------------------------------------------------

def controlled_execution(
    module: nn.Module,
    inp: Iterable[Any],
    method: Callable[..., Any],
) -> Any:
    """Run ``method(*inp)`` with the module's parameters temporarily
    replaced by ``torch.linspace(-1, 1, numel)``. The originals are
    restored on exit.

    用 ``torch.linspace(-1, 1, numel)`` 临时替换模块参数后跑 ``method(*inp)``，
    返回前恢复原参数。

    Args / 参数:
        module:  the module under test / 被测模块.
        inp:     positional arguments forwarded to ``method`` /
                 转发给 ``method`` 的位置参数.
        method:  callable to invoke; usually ``lambda *x: module(*x)`` /
                 通常传 ``lambda *x: module(*x)``.

    Returns / 返回:
        Whatever ``method`` returned. / ``method`` 的返回值。
    """
    was_training = module.training
    module.eval()
    module.double()
    with torch.no_grad():
        original_params = [p.clone() for p in module.parameters()]
        for p in module.parameters():
            p.copy_(torch.linspace(-1, 1, p.numel()).reshape(p.shape))
        out = method(*inp)
        for orig, p in zip(original_params, module.parameters()):
            p.copy_(orig)
    if was_training:
        module.train()
    return out


def controlled_forward(module: nn.Module, inp: Iterable[Any]) -> Any:
    """Convenience wrapper for ``controlled_execution`` with ``module(...)``.
    便捷封装：以 ``module(...)`` 作为方法跑 ``controlled_execution``。
    """
    return controlled_execution(module, inp, lambda *x: module(*x))


# ----------------------------------------------------------------------------
# Shape check
# 形状检查
# ----------------------------------------------------------------------------

def test_module_shape(
    module: nn.Module,
    test_name: str,
    control_folder: str,
    overwrite_results: bool = False,
) -> None:
    """Assert that the trainable-parameter shapes of ``module`` match the
    expected shapes stored at ``<control_folder>/<test_name>_param_shapes.pt``.

    Set ``overwrite_results=True`` to regenerate the reference file.

    断言 ``module`` 的可训练参数 shape 与
    ``<control_folder>/<test_name>_param_shapes.pt`` 中保存的一致。
    ``overwrite_results=True`` 时重新生成参考文件。
    """
    try:
        param_shapes = {name: tuple(p.shape) for name, p in module.named_parameters()}
    except AttributeError as exc:
        # nn.Module raises this when ``__init__`` never ran (e.g. the student
        # left ``__init__``'s body as ``pass`` and ``super().__init__()`` was
        # never called). Surface a friendlier hint.
        raise AssertionError(
            f"{test_name}: {type(module).__name__} appears to be uninitialized "
            f"({exc}). This usually means your ``__init__`` body is still "
            f"``pass`` so ``super().__init__()`` (and the parameter setup) "
            f"never ran. Fill in the __init__ TODO first."
        ) from None
    shapes_path = os.path.join(control_folder, f"{test_name}_param_shapes.pt")

    if overwrite_results:
        torch.save(param_shapes, shapes_path)

    expected = torch.load(shapes_path)
    expected = {k: tuple(v) for k, v in expected.items()}

    extra = set(param_shapes) - set(expected)
    missing = set(expected) - set(param_shapes)
    if extra or missing:
        msg = f"Parameter key set mismatch for {test_name}:\n"
        if extra:
            msg += f"  unexpected keys: {sorted(extra)}\n"
        if missing:
            msg += f"  missing keys:    {sorted(missing)}\n"
        raise AssertionError(msg)

    for name, shape in param_shapes.items():
        assert shape == expected[name], (
            f"shape mismatch for {name} in {test_name}: "
            f"expected {expected[name]}, got {shape}"
        )


# ----------------------------------------------------------------------------
# Output check
# 输出检查
# ----------------------------------------------------------------------------

def _as_tuple(x: Any) -> tuple:
    if isinstance(x, tuple):
        return x
    if isinstance(x, list):
        return tuple(x)
    return (x,)


def test_module_method(
    module: nn.Module,
    test_name: str,
    inputs: Iterable[Any],
    output_names: Iterable[str] | str,
    control_folder: str,
    method: Callable[..., Any],
    overwrite_results: bool = False,
    atol: float = 1e-6,
    rtol: float = 1e-5,
) -> None:
    """Run ``controlled_execution(module, inputs, method)`` and compare each
    output to the saved tensor at
    ``<control_folder>/<test_name>_<output_name>.pt``.

    跑 ``controlled_execution(module, inputs, method)`` 并把每个输出与
    ``<control_folder>/<test_name>_<output_name>.pt`` 中保存的张量对比。

    Args / 参数:
        inputs:        tuple/list of inputs forwarded to ``method`` /
                       转发给 ``method`` 的输入元组.
        output_names:  string or sequence of strings labelling each output /
                       输出的标签 (字符串或字符串序列).
        atol / rtol:   passed to torch.allclose / 透传给 torch.allclose.
    """
    output_names = _as_tuple(output_names)
    inputs = tuple(inputs)

    with torch.no_grad():
        out = controlled_execution(module, inputs, method)
    out = _as_tuple(out)

    if overwrite_results:
        for o, name in zip(out, output_names):
            path = os.path.join(control_folder, f"{test_name}_{name}.pt")
            torch.save(o, path)
        return

    for o, name in zip(out, output_names):
        path = os.path.join(control_folder, f"{test_name}_{name}.pt")
        expected = torch.load(path)
        if o is None:
            raise AssertionError(
                f"{test_name} output `{name}` is None — your implementation "
                f"likely returned nothing (forgot to return / left ``pass``)."
            )
        if not isinstance(o, torch.Tensor):
            raise AssertionError(
                f"{test_name} output `{name}` is not a torch.Tensor "
                f"(got {type(o).__name__}); check your return value."
            )
        max_abs = (o.double() - expected.double()).abs().max().item()
        assert torch.allclose(o, expected, atol=atol, rtol=rtol), (
            f"{test_name} output `{name}` mismatch "
            f"(max abs error = {max_abs:.3e})"
        )


def test_module_forward(
    module: nn.Module,
    test_name: str,
    inputs: Iterable[Any],
    output_names: Iterable[str] | str,
    control_folder: str,
    overwrite_results: bool = False,
) -> None:
    """Convenience wrapper around ``test_module_method`` with ``method =
    module(...)``.

    ``method = module(...)`` 时对 ``test_module_method`` 的便捷封装。
    """
    test_module_method(
        module=module,
        test_name=test_name,
        inputs=inputs,
        output_names=output_names,
        control_folder=control_folder,
        method=lambda *x: module(*x),
        overwrite_results=overwrite_results,
    )
