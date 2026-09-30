"""命令行接口。

用法::

    python -m backend.cli list
    python -m backend.cli run --attack a1_bottom_right_white --defense d2_median_filter_3x3 --target 0
    python -m backend.cli matrix --attacks a1,a3 --defenses d1,d2,d5 --targets 0,3,7
    python -m backend.cli register-attack ./submissions/example_attack
    python -m backend.cli serve [--host 127.0.0.1 --port 8000]
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional

from .core.config import EvalConfig, get_config
from .core.evaluator import MatrixResult, evaluate, run_matrix
from .core.registry import get_default_registry


def _parse_csv(s: Optional[str]) -> Optional[List[str]]:
    if not s:
        return None
    return [x.strip() for x in s.split(",") if x.strip()]


def _parse_int_csv(s: Optional[str]) -> Optional[List[int]]:
    parts = _parse_csv(s)
    if parts is None:
        return None
    return [int(p) for p in parts]


def _build_config(args) -> EvalConfig:
    mode = getattr(args, "mode", None) or "fast"
    cfg = get_config(mode)
    if getattr(args, "train_subset", None) is not None:
        cfg.train_subset = int(args.train_subset)
    if getattr(args, "test_subset", None) is not None:
        cfg.test_subset = int(args.test_subset)
    if getattr(args, "epochs", None) is not None:
        cfg.train.epochs = int(args.epochs)
    if getattr(args, "poison_rate", None) is not None:
        cfg.poison_rate = float(args.poison_rate)
    return cfg


def _print_metrics(res) -> None:
    m = res.metrics
    print(f"\n=== 评测结果: {res.attack} × {res.defense} (target={res.target_label}, seed={res.seed}) ===")
    print(f"耗时: {res.elapsed_sec:.1f}s | 模式: {res.config_mode}")
    if res.error:
        print(f"⚠️  错误: {res.error}")
    if not m.attack_valid:
        print("⚠️  攻击合法性校验未通过:")
        for v in m.attack_violations:
            print(f"   - {v}")
    if not m.defense_valid:
        print("⚠️  防御合法性校验未通过:")
        for v in m.defense_violations:
            print(f"   - {v}")
    print(f"投毒样本数: {m.n_poison}  触发测试集大小: {m.n_triggered_test}")
    print(f"Acc  clean / attack / defense = {m.acc_clean:.4f} / {m.acc_attack:.4f} / {m.acc_defense:.4f}")
    print(f"ASR  clean / attack / defense = {m.asr_clean:.4f} / {m.asr_attack:.4f} / {m.asr_defense:.4f}")
    print("--- 攻击指标 ---")
    print(f"  BackdoorGain        = {m.backdoor_gain:.4f}")
    print(f"  CleanRetention_atk = {m.clean_retention_attack:.4f}")
    print(f"  AttackScore        = {m.attack_score:.4f}")
    print(f"  资格线: ASR>={m.attack_qualified} (acc_drop={m.attack_acc_drop:.4f})")
    print("--- 防御指标 ---")
    skip = " (跳过: 攻击无明显后门效果)" if m.defense_score_skipped else ""
    print(f"  BackdoorRemoval    = {m.backdoor_removal:.4f}")
    print(f"  CleanRetention_def = {m.clean_retention_defense:.4f}")
    print(f"  DefenseScore       = {m.defense_score:.4f}{skip}")
    print(f"  参考线: success={m.defense_success} (acc_drop={m.defense_acc_drop:.4f})")


def _print_matrix(matrix: MatrixResult) -> None:
    print(f"\n=== 全对阵矩阵: {len(matrix.attacks)} 攻击 × {len(matrix.defenses)} 防御 "
          f"× {len(matrix.target_labels)} 标签 × {len(matrix.seeds)} 种子 ===")
    print(f"共 {len(matrix.results)} 组评测\n")

    def fmt_table(title, mat):
        print(f"--- {title} ---")
        corner = "attack\\defense"
        header = f"{corner:>28} " + " ".join(f"{d:>22}" for d in matrix.defenses)
        print(header)
        for a in matrix.attacks:
            row = f"{a:>28} " + " ".join(f"{mat[a][d]:>22.4f}" for d in matrix.defenses)
            print(row)
        print()

    fmt_table("AttackScore (越高对攻击越有利)", matrix.attack_score_matrix)
    fmt_table("DefenseScore (越高对防御越有利)", matrix.defense_score_matrix)
    fmt_table("ASR_attack (投毒模型后门成功率)", matrix.asr_attack_matrix)
    fmt_table("ASR_defense (防御后后门成功率)", matrix.asr_defense_matrix)


# ------------------------------------------------------------------ 子命令


def cmd_list(args):
    reg = get_default_registry()
    if args.submissions:
        for d in args.submissions:
            kind, path = d.split(":", 1) if ":" in d else ("attack", d)
            try:
                if kind.startswith("def"):
                    reg.register_defense_dir(path)
                else:
                    reg.register_attack_dir(path)
            except Exception as e:  # noqa: BLE001
                print(f"⚠️  无法加载 {path}: {e}", file=sys.stderr)
    info = reg.info()
    print("攻击方法 (attacks):")
    for a in info["attacks"]:
        print(f"  - {a['name']:<28} [{a['source']}] {a['description']}")
    print("\n防御方法 (defenses):")
    for d in info["defenses"]:
        print(f"  - {d['name']:<28} [{d['source']}] {d['description']}")


def cmd_run(args):
    reg = get_default_registry()
    if not reg.has_attack(args.attack):
        print(f"未知攻击方法: {args.attack}", file=sys.stderr)
        sys.exit(2)
    if not reg.has_defense(args.defense):
        print(f"未知防御方法: {args.defense}", file=sys.stderr)
        sys.exit(2)
    cfg = _build_config(args)
    res = evaluate(
        attack_fn=reg.get_attack(args.attack).fn,
        defend_fn=reg.get_defense(args.defense).fn,
        target_label=args.target, seed=args.seed,
        config=cfg, attack_name=args.attack, defense_name=args.defense,
    )
    _print_metrics(res)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(res.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n(JSON 已写入 {args.json})")


def cmd_matrix(args):
    reg = get_default_registry()
    attacks = _parse_csv(args.attacks) or reg.attack_names()
    defenses = _parse_csv(args.defenses) or reg.defense_names()
    targets = _parse_int_csv(args.targets) or [0]
    seeds = _parse_int_csv(args.seeds) or [0]
    for a in attacks:
        if not reg.has_attack(a):
            print(f"未知攻击方法: {a}", file=sys.stderr)
            sys.exit(2)
    for d in defenses:
        if not reg.has_defense(d):
            print(f"未知防御方法: {d}", file=sys.stderr)
            sys.exit(2)
    cfg = _build_config(args)
    atk_map = {a: reg.get_attack(a).fn for a in attacks}
    def_map = {d: reg.get_defense(d).fn for d in defenses}

    def progress(done, total, msg):
        print(f"\r{msg}", end="", flush=True)

    matrix = run_matrix(
        attacks=atk_map, defenses=def_map,
        target_labels=targets, seeds=seeds, config=cfg,
        progress_callback=progress if not args.quiet else None,
    )
    print()
    _print_matrix(matrix)
    if args.json:
        with open(args.json, "w", encoding="utf-8") as f:
            json.dump(matrix.to_dict(), f, ensure_ascii=False, indent=2)
        print(f"\n(JSON 已写入 {args.json})")


def cmd_serve(args):
    import uvicorn
    print(f"启动 FastAPI 服务: http://{args.host}:{args.port}")
    uvicorn.run("backend.api.server:app", host=args.host, port=args.port, reload=False)


def cmd_register_attack(args):
    reg = get_default_registry()
    info = reg.register_attack_dir(args.path, name=args.name)
    print(f"已注册攻击方法: {info.name} ({info.path})")


def cmd_register_defense(args):
    reg = get_default_registry()
    info = reg.register_defense_dir(args.path, name=args.name)
    print(f"已注册防御方法: {info.name} ({info.path})")


# ------------------------------------------------------------------ 入口


def _add_common_eval_args(p):
    p.add_argument("--mode", choices=["fast", "full"], default="fast", help="计算档 (默认 fast)")
    p.add_argument("--train-subset", type=int, default=None, help="训练子集大小 (fast 默认 6000)")
    p.add_argument("--test-subset", type=int, default=None, help="测试子集大小 (fast 默认 1000)")
    p.add_argument("--epochs", type=int, default=None, help="训练 epoch 数 (fast 默认 3)")
    p.add_argument("--poison-rate", type=float, default=None, help="投毒比例 (默认 0.05)")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="backend.cli", description="MNIST 后门攻防对抗后端 CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="列出已注册的攻防方法")
    p_list.add_argument("--submissions", nargs="*", default=None,
                        help="额外加载 submission 目录，格式 attack:./path 或 defense:./path")
    p_list.set_defaults(func=cmd_list)

    p_run = sub.add_parser("run", help="运行单组 (attack, defense) 评测")
    p_run.add_argument("--attack", required=True)
    p_run.add_argument("--defense", required=True)
    p_run.add_argument("--target", type=int, default=0, help="目标标签 (默认 0)")
    p_run.add_argument("--seed", type=int, default=0)
    p_run.add_argument("--json", default=None, help="将结果写入 JSON 文件")
    _add_common_eval_args(p_run)
    p_run.set_defaults(func=cmd_run)

    p_mat = sub.add_parser("matrix", help="运行全对阵矩阵")
    p_mat.add_argument("--attacks", default=None, help="逗号分隔的攻击方法名 (默认全部)")
    p_mat.add_argument("--defenses", default=None, help="逗号分隔的防御方法名 (默认全部)")
    p_mat.add_argument("--targets", default=None, help="逗号分隔的目标标签 (默认 0)")
    p_mat.add_argument("--seeds", default=None, help="逗号分隔的随机种子 (默认 0)")
    p_mat.add_argument("--json", default=None, help="将结果写入 JSON 文件")
    p_mat.add_argument("--quiet", action="store_true", help="不打印进度")
    _add_common_eval_args(p_mat)
    p_mat.set_defaults(func=cmd_matrix)

    p_ra = sub.add_parser("register-attack", help="注册 submission 式攻击目录")
    p_ra.add_argument("path", help="包含 attack.py 的目录")
    p_ra.add_argument("--name", default=None, help="注册名 (默认目录名)")
    p_ra.set_defaults(func=cmd_register_attack)

    p_rd = sub.add_parser("register-defense", help="注册 submission 式防御目录")
    p_rd.add_argument("path", help="包含 defense.py 的目录")
    p_rd.add_argument("--name", default=None, help="注册名 (默认目录名)")
    p_rd.set_defaults(func=cmd_register_defense)

    p_serve = sub.add_parser("serve", help="启动 FastAPI 服务")
    p_serve.add_argument("--host", default="127.0.0.1")
    p_serve.add_argument("--port", type=int, default=8000)
    p_serve.set_defaults(func=cmd_serve)

    return p


def main(argv: Optional[List[str]] = None) -> int:
    # Windows 控制台默认 GBK，强制 UTF-8 以正确显示中文与符号
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:  # noqa: BLE001
        pass
    parser = build_parser()
    args = parser.parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
