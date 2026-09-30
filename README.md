# MNIST 局部触发器后门攻防对抗后端

根据《手写数字局部触发器后门攻防对抗赛.md》实现的攻防对抗评测后端。
支持**任意攻击方法 × 任意防御方法**的组合，自动跑通完整管线并计算攻击指标与防御指标。

## 功能

- **插件化攻防方法**：内置 5 种攻击 + 5 种防御基线，并支持加载 submission 式 `attack.py` / `defense.py`。
- **完整评测管线**：干净训练 → 攻击 → 投毒训练 → 防御 → 清洗训练 → 干净/触发测试 → 指标计算。
- **自动合法性校验**：投毒预算、修改区域、统一触发器、标签约束、防御样本数下限等（第十四、十七节）。
- **全套指标**：`Acc_clean/attack/defense`、`ASR_clean/attack/defense`、`BackdoorGain`、`CleanRetention`、`AttackScore`、`BackdoorRemoval`、`DefenseScore`、资格线/参考线。
- **全对阵矩阵**：`A × D × 目标标签 × 种子`，复用 clean/attack 模型避免重复训练，输出得分矩阵。
- **三种接口**：可复用 Python 库 / CLI / FastAPI HTTP 服务。
- **两档计算规模**：默认 `fast`（MNIST 子集 + 少量 epoch，单组几秒），`--mode full` 切换完整 MNIST。

## 安装

```bash
pip install -r requirements.txt
```

首次运行会自动从公开镜像下载 MNIST 到 `data/`。

## 快速开始

```bash
# 列出内置攻防方法
python -m backend.cli list

# 单组评测
python -m backend.cli run --attack a1_bottom_right_white --defense d2_median_filter_3x3 --target 0

# 全对阵矩阵（多目标标签平均）
python -m backend.cli matrix --attacks a1,a3,a4 --defenses d1,d2,d5 --targets 0,3,7

# 启动 HTTP 服务
python -m backend.cli serve
```

## 内置方法

| 攻击 | 说明 |
|------|------|
| `a1_bottom_right_white` | A0-1 右下角 2×2 全白（主办方基线） |
| `a2_lower_black` | 下方偏中 2×2 全黑（擦除像素，更隐蔽） |
| `a3_top_right_white` | 右上角 2×2 全白（与 a1 对称位置） |
| `a4_checkerboard` | 右下角 2×2 棋盘 |
| `a5_random_binary` | 随机位置 2×2 二值图案 |

| 防御 | 说明 |
|------|------|
| `d1_no_defense` | D0-1 无防御（主办方基线） |
| `d2_median_filter_3x3` | 3×3 中值滤波（最小示例） |
| `d3_gaussian_blur` | 高斯模糊 |
| `d4_label_smoothing_weight` | 代理模型置信度降权 |
| `d5_spectral_signature` | 谱签名异常检测降权 |

## 自定义攻防方法

### submission 式（推荐参赛使用）

按赛题规范在目录中放置 `attack.py`（含 `def attack(env, task)`）或 `defense.py`（含 `def defend(env, train)`），见 `submissions/example_attack/` 与 `submissions/example_defense/`。

```bash
# CLI 注册
python -m backend.cli register-attack ./submissions/example_attack --name my_attack
python -m backend.cli run --attack my_attack --defense d2_median_filter_3x3

# 或 HTTP 上传
curl -F "file=@attack.py" -F "name=my_attack" http://127.0.0.1:8000/attacks/upload
```

### Python 库式

```python
from backend.core.registry import get_default_registry
from backend.core.evaluator import evaluate
from backend.core.config import EvalConfig

reg = get_default_registry()
res = evaluate(
    attack_fn=reg.get_attack("a1_bottom_right_white").fn,
    defend_fn=reg.get_defense("d2_median_filter_3x3").fn,
    target_label=0, seed=0, config=EvalConfig.fast(),
)
print(res.metrics.attack_score, res.metrics.defense_score)
```

## HTTP 接口

| 方法 | 路径 | 说明 |
|------|------|------|
| GET | `/` | 健康检查 |
| GET | `/attacks` `/defenses` `/methods` | 列出方法 |
| GET/POST | `/config` | 查看/设置计算档 |
| GET | `/datasets/status` | MNIST 缓存状态 |
| POST | `/evaluate` | 单组评测 |
| POST | `/matrix` | 全对阵矩阵 |
| POST | `/attacks/upload` `/defenses/upload` | 上传 submission |

`POST /evaluate` 示例：

```bash
curl -X POST http://127.0.0.1:8000/evaluate \
  -H "Content-Type: application/json" \
  -d '{"attack":"a1_bottom_right_white","defense":"d2_median_filter_3x3","target_label":0,"seed":0,"mode":"fast"}'
```

## 指标说明

| 指标 | 公式 | 含义 |
|------|------|------|
| `Acc_clean/attack/defense` | — | 干净/投毒/防御模型在干净测试集上的准确率 |
| `ASR_attack` | 预测==目标标签的比例 | 投毒模型后门成功率 |
| `ASR_clean` | — | 干净模型在触发测试集上的自然基线 |
| `BackdoorGain` | `max(0,(ASR_attack-ASR_clean)/(1-ASR_clean))` | 归一化后门增益 |
| `CleanRetention_attack` | `min(1,Acc_attack/Acc_clean)` | 攻击隐蔽性 |
| `AttackScore` | `BackdoorGain × CleanRetention_attack` | 攻击得分 |
| `BackdoorRemoval` | `(ASR_attack-ASR_defense)/(ASR_attack-ASR_clean)` | 后门消除率 |
| `CleanRetention_defense` | `min(1,Acc_defense/Acc_clean)` | 防御干净性能保持 |
| `DefenseScore` | `0.7·BackdoorRemoval + 0.3·CleanRetention_defense` | 防御得分 |

## 测试

```bash
python tests/test_pipeline.py        # 无需 pytest
# 或
python -m pytest tests/ -q
```

## 目录结构

```
backend/
├── core/        # 数据集、模型、训练、环境、触发器、校验、指标、注册表、评测器
├── attacks/     # 内置攻击基线
├── defenses/    # 内置防御基线
├── api/         # FastAPI 服务
└── cli.py       # 命令行
submissions/     # submission 式插件示例
data/            # MNIST 缓存（自动生成）
tests/           # 冒烟测试
```
