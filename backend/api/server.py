"""FastAPI HTTP 服务。

启动::
    python -m backend.cli serve
    # 或
    uvicorn backend.api.server:app --host 127.0.0.1 --port 8000

接口:
    GET  /                  健康检查
    GET  /attacks           列出攻击方法
    GET  /defenses          列出防御方法
    GET  /config            查看默认配置
    POST /config            设置默认计算档 (fast/full)
    GET  /datasets/status   MNIST 缓存状态
    POST /evaluate          单组评测
    POST /matrix            全对阵矩阵
    POST /attacks/upload    上传 submission 式攻击目录
    POST /defenses/upload   上传 submission 式防御目录
"""
from __future__ import annotations

import os
import shutil
import tempfile
from typing import List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..core.config import EvalConfig, get_config
from ..core.dataset import cache_status
from ..core.evaluator import MatrixResult, evaluate, run_matrix
from ..core.registry import get_default_registry

app = FastAPI(
    title="MNIST 局部触发器后门攻防对抗后端",
    version="0.1.0",
    description="任意攻击方法 × 任意防御方法的自动化评测，计算攻击与防御指标。",
)

# 全局状态：默认配置 + 已上传 submission 目录
_state = {
    "config": get_config("fast"),
    "upload_dir": os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "uploads"),
}


# ------------------------------------------------------------------ 模型


class EvaluateRequest(BaseModel):
    attack: str
    defense: str
    target_label: int = 0
    seed: int = 0
    mode: Optional[str] = None        # 覆写计算档: fast/full
    train_subset: Optional[int] = None
    epochs: Optional[int] = None


class MatrixRequest(BaseModel):
    attacks: List[str]
    defenses: List[str]
    target_labels: List[int] = [0]
    seeds: List[int] = [0]
    mode: Optional[str] = None
    train_subset: Optional[int] = None
    epochs: Optional[int] = None


class ConfigRequest(BaseModel):
    mode: str = "fast"


# ------------------------------------------------------------------ 工具


def _build_config(req) -> EvalConfig:
    mode = getattr(req, "mode", None) or _state["config"].mode
    cfg = get_config(mode)
    if getattr(req, "train_subset", None) is not None:
        cfg.train_subset = int(req.train_subset)
    if getattr(req, "epochs", None) is not None:
        cfg.train.epochs = int(req.epochs)
    return cfg


def _registry():
    return get_default_registry()


# ------------------------------------------------------------------ 路由


@app.get("/")
def health():
    return {"status": "ok", "service": "backdoor-arena", "version": "0.1.0"}


@app.get("/attacks")
def list_attacks():
    return {"attacks": [m.__dict__ if hasattr(m, "__dict__") else m for m in _registry().list_attacks()]}


@app.get("/defenses")
def list_defenses():
    return {"defenses": [m.__dict__ if hasattr(m, "__dict__") else m for m in _registry().list_defenses()]}


@app.get("/methods")
def list_all():
    """同时返回攻防方法与可选项。"""
    reg = _registry()
    return reg.info()


@app.get("/config")
def get_config_endpoint():
    c = _state["config"]
    return {"mode": c.mode, "train_subset": c.train_subset, "test_subset": c.test_subset,
            "epochs": c.train.epochs, "poison_rate": c.poison_rate, "trigger_size": c.trigger_size}


@app.post("/config")
def set_config_endpoint(req: ConfigRequest):
    _state["config"] = get_config(req.mode)
    return get_config_endpoint()


@app.get("/datasets/status")
def datasets_status():
    return cache_status()


@app.post("/evaluate")
def evaluate_endpoint(req: EvaluateRequest):
    reg = _registry()
    if not reg.has_attack(req.attack):
        raise HTTPException(404, f"未知攻击方法: {req.attack}")
    if not reg.has_defense(req.defense):
        raise HTTPException(404, f"未知防御方法: {req.defense}")
    cfg = _build_config(req)
    try:
        atk = reg.get_attack(req.attack)
        dfn = reg.get_defense(req.defense)
        res = evaluate(
            attack_fn=atk.fn, defend_fn=dfn.fn,
            target_label=req.target_label, seed=req.seed,
            config=cfg, attack_name=req.attack, defense_name=req.defense,
        )
        return res.to_dict()
    except Exception as e:  # noqa: BLE001
        import traceback
        raise HTTPException(500, f"评测失败: {e}\n{traceback.format_exc()}")


@app.post("/matrix")
def matrix_endpoint(req: MatrixRequest):
    reg = _registry()
    attacks = {}
    for name in req.attacks:
        if not reg.has_attack(name):
            raise HTTPException(404, f"未知攻击方法: {name}")
        attacks[name] = reg.get_attack(name).fn
    defenses = {}
    for name in req.defenses:
        if not reg.has_defense(name):
            raise HTTPException(404, f"未知防御方法: {name}")
        defenses[name] = reg.get_defense(name).fn
    cfg = _build_config(req)
    try:
        matrix: MatrixResult = run_matrix(
            attacks=attacks, defenses=defenses,
            target_labels=req.target_labels, seeds=req.seeds,
            config=cfg,
        )
        return matrix.to_dict()
    except Exception as e:  # noqa: BLE001
        import traceback
        raise HTTPException(500, f"矩阵评测失败: {e}\n{traceback.format_exc()}")


@app.post("/attacks/upload")
async def upload_attack(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """上传 attack.py（submission 式），注册为新攻击方法。"""
    return await _save_submission(file, name, kind="attack")


@app.post("/defenses/upload")
async def upload_defense(file: UploadFile = File(...), name: Optional[str] = Form(None)):
    """上传 defense.py（submission 式），注册为新防御方法。"""
    return await _save_submission(file, name, kind="defense")


async def _save_submission(file: UploadFile, name: Optional[str], kind: str):
    fname = file.filename or f"{kind}.py"
    expected = "attack.py" if kind == "attack" else "defense.py"
    if not fname.endswith(".py"):
        raise HTTPException(400, "仅接受 .py 文件")
    upload_root = os.path.abspath(_state["upload_dir"])
    os.makedirs(upload_root, exist_ok=True)
    sub_name = name or os.path.splitext(fname)[0]
    sub_dir = os.path.join(upload_root, f"{kind}_{sub_name}")
    os.makedirs(sub_dir, exist_ok=True)
    # 若上传文件名不是规范的 attack.py/defense.py，重命名
    dest_name = expected if fname in ("attack.py", "defense.py") else fname
    dest = os.path.join(sub_dir, dest_name)
    content = await file.read()
    with open(dest, "wb") as f:
        f.write(content)
    # 如果文件名不规范但内容是 attack/defend，复制一份规范名
    if dest_name != expected:
        shutil.copyfile(dest, os.path.join(sub_dir, expected))
    reg = _registry()
    try:
        if kind == "attack":
            info = reg.register_attack_dir(sub_dir, name=sub_name)
        else:
            info = reg.register_defense_dir(sub_dir, name=sub_name)
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"加载失败: {e}")
    return {"registered": info.name, "kind": kind, "path": sub_dir,
            "description": info.description}
