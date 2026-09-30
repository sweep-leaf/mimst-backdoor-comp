"""攻击 / 防御方法注册表与插件加载。

支持两种注册来源，统一为 ``Callable``：
1. 内置：``backend/attacks/*`` 与 ``backend/defenses/*`` 中继承基类的子类自动发现。
2. submission 式：给定目录，按规范加载 ``attack.py:attack(env, task)``
   或 ``defense.py:defend(env, train)``（第十三、十八节接口）。

每个方法对外暴露统一签名::

    attack_fn(env, task) -> dict
    defend_fn(env, train) -> dict
"""
from __future__ import annotations

import importlib
import importlib.util
import inspect
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional


@dataclass
class MethodInfo:
    name: str
    kind: str                       # "attack" | "defense"
    description: str
    fn: Callable
    source: str = "builtin"         # "builtin" | "submission"
    path: Optional[str] = None

    def __call__(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class Registry:
    """攻防方法注册表。"""

    def __init__(self):
        self._attacks: Dict[str, MethodInfo] = {}
        self._defenses: Dict[str, MethodInfo] = {}

    # ---- 注册 ----
    def register_attack(self, name: str, fn: Callable, description: str = "",
                        source: str = "builtin", path: Optional[str] = None) -> None:
        self._attacks[name] = MethodInfo(
            name=name, kind="attack", description=description or fn.__doc__ or "",
            fn=fn, source=source, path=path,
        )

    def register_defense(self, name: str, fn: Callable, description: str = "",
                         source: str = "builtin", path: Optional[str] = None) -> None:
        self._defenses[name] = MethodInfo(
            name=name, kind="defense", description=description or fn.__doc__ or "",
            fn=fn, source=source, path=path,
        )

    # ---- 查询 ----
    def list_attacks(self) -> List[MethodInfo]:
        return sorted(self._attacks.values(), key=lambda m: m.name)

    def list_defenses(self) -> List[MethodInfo]:
        return sorted(self._defenses.values(), key=lambda m: m.name)

    def attack_names(self) -> List[str]:
        return [m.name for m in self.list_attacks()]

    def defense_names(self) -> List[str]:
        return [m.name for m in self.list_defenses()]

    def get_attack(self, name: str) -> MethodInfo:
        if name not in self._attacks:
            raise KeyError(f"未知攻击方法: {name}。可用: {self.attack_names()}")
        return self._attacks[name]

    def get_defense(self, name: str) -> MethodInfo:
        if name not in self._defenses:
            raise KeyError(f"未知防御方法: {name}。可用: {self.defense_names()}")
        return self._defenses[name]

    def has_attack(self, name: str) -> bool:
        return name in self._attacks

    def has_defense(self, name: str) -> bool:
        return name in self._defenses

    # ---- submission 式加载 ----
    def register_attack_dir(self, dir_path: str, name: Optional[str] = None) -> MethodInfo:
        """加载 submission 式攻击目录（含 attack.py）。"""
        fn = _load_submission_fn(dir_path, "attack.py", "attack")
        nm = name or os.path.basename(os.path.abspath(dir_path))
        info = MethodInfo(
            name=nm, kind="attack",
            description=fn.__doc__ or f"submission attack from {dir_path}",
            fn=fn, source="submission", path=os.path.abspath(dir_path),
        )
        self._attacks[nm] = info
        return info

    def register_defense_dir(self, dir_path: str, name: Optional[str] = None) -> MethodInfo:
        """加载 submission 式防御目录（含 defense.py）。"""
        fn = _load_submission_fn(dir_path, "defense.py", "defend")
        nm = name or os.path.basename(os.path.abspath(dir_path))
        info = MethodInfo(
            name=nm, kind="defense",
            description=fn.__doc__ or f"submission defense from {dir_path}",
            fn=fn, source="submission", path=os.path.abspath(dir_path),
        )
        self._defenses[nm] = info
        return info

    def info(self) -> dict:
        return {
            "attacks": [
                {"name": m.name, "description": m.description, "source": m.source, "path": m.path}
                for m in self.list_attacks()
            ],
            "defenses": [
                {"name": m.name, "description": m.description, "source": m.source, "path": m.path}
                for m in self.list_defenses()
            ],
        }


# ------------------------------------------------------------------ 加载工具


def _load_submission_fn(dir_path: str, filename: str, fn_name: str) -> Callable:
    """从目录加载 ``filename`` 模块并取出 ``fn_name`` 函数。"""
    dir_path = os.path.abspath(dir_path)
    file_path = os.path.join(dir_path, filename)
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"在 {dir_path} 下找不到 {filename}")

    mod_name = f"_backdoor_submission_{abs(hash(dir_path)) & 0xFFFFFFFF:x}_{filename.replace('.', '_')}"
    spec = importlib.util.spec_from_file_location(mod_name, file_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"无法为 {file_path} 构建模块 spec")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[mod_name] = mod
    # 让 submission 内的相对 import 能工作（将其目录加入 sys.path）
    sys.path.insert(0, dir_path)
    try:
        spec.loader.exec_module(mod)
    finally:
        try:
            sys.path.remove(dir_path)
        except ValueError:
            pass

    if not hasattr(mod, fn_name):
        raise AttributeError(f"{file_path} 中未定义函数 `{fn_name}`")
    fn = getattr(mod, fn_name)
    if not callable(fn):
        raise TypeError(f"{file_path} 中的 `{fn_name}` 不是可调用对象")
    return fn


# ------------------------------------------------------------------ 默认注册表


def _discover_builtin(registry: Registry) -> None:
    """扫描 backend/attacks 与 backend/defenses 下所有模块，自动注册基类子类。"""
    from ..attacks.base import AttackBase
    from ..defenses.base import DefenseBase

    _scan_package(registry, "backend.attacks", AttackBase, "attack")
    _scan_package(registry, "backend.defenses", DefenseBase, "defense")


def _scan_package(registry: Registry, pkg_name: str, base_cls: type, kind: str) -> None:
    pkg = importlib.import_module(pkg_name)
    pkg_dir = os.path.dirname(pkg.__file__)
    for fname in sorted(os.listdir(pkg_dir)):
        if not fname.endswith(".py") or fname.startswith("_") or fname.startswith("base"):
            continue
        mod_name = f"{pkg_name}.{fname[:-3]}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception as e:  # noqa: BLE001
            print(f"[registry] 跳过 {mod_name}: {e}", file=sys.stderr)
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr)
            if (inspect.isclass(obj) and issubclass(obj, base_cls)
                    and obj is not base_cls and obj.__module__ == mod_name):
                instance = obj()
                name = getattr(instance, "name", attr.lower())
                desc = (instance.__doc__ or "").strip().splitlines()
                desc = desc[0] if desc else ""
                fn = instance.run if kind == "attack" else instance.run
                if kind == "attack":
                    registry.register_attack(name, fn, desc, source="builtin")
                else:
                    registry.register_defense(name, fn, desc, source="builtin")


_default_registry: Optional[Registry] = None


def get_default_registry() -> Registry:
    """获取（按需构建的）默认注册表，包含所有内置攻防方法。"""
    global _default_registry
    if _default_registry is None:
        _default_registry = Registry()
        _discover_builtin(_default_registry)
    return _default_registry
