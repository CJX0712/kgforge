# KGForge

**世界顶级知识图谱嵌入与链接预测系统** · 旗舰 **KGFuse** 多模型倒数秩融合（Reciprocal Rank Fusion）

[![CI](https://github.com/CJX0712/kgforge/actions/workflows/ci.yml/badge.svg)](https://github.com/CJX0712/kgforge/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/CJX0712/kgforge)](https://github.com/CJX0712/kgforge/releases)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Quality](https://img.shields.io/badge/quality-S-brightgreen.svg)](benchmark.json)

作者：**晨星** (CJX0712) · 纯 NumPy · 离线可跑 · CPU-only · 确定性逐位可复现

---

## 一句话结论

在一个可复现的合成知识图谱基准上，**KGFuse 的 filtered MRR = 0.2018 ± 0.0277**，
而最强单一 SOTA 模型 DistMult 只有 **0.1717 ± 0.0094** —— **相对提升 +17.56%**，
且均值差 0.0301 > 显著性判据 ½(σ₁+σ₂) = 0.0186。同 seed 两次运行**逐位一致**。

## 为什么是"融合"而不是"又一个嵌入模型"

四个 SOTA 模型的打分函数天生**不可比**：TransE/RotatE 输出距离的负值，
DistMult/ComplEx 输出双线性点积。原始分数根本不在同一量纲上，直接平均是错的。
所以在**秩**空间融合才是正确做法 —— 旗舰采用 **RRF**（Cormack et al. 2009）：

```
fused_score(e) = Σ_m  1 / (k + rank_m(e)),   k = 60
```

四个基模型各自有**结构性盲区**，融合才有互补价值：

| 关系类型 | 谁强 | 谁结构性瞎 |
|---|---|---|
| symmetric | ComplEx / DistMult | TransE（需 r≈0） |
| antisymmetric | TransE / RotatE | DistMult（h,t 对称） |
| composition | RotatE（旋转加法） | DistMult / TransE |
| one-to-many | 全都弱 | — |

## 性能基线表（3 seeds，mean ± std，filtered 协议）

| 方法 | MRR | 备注 |
|---|---|---|
| TransE (Bordes 2013) | 0.0468 ± 0.0103 | |
| DistMult (Yang 2015) | 0.1717 ± 0.0094 | **最强单模型** |
| ComplEx (Trouillon 2016) | 0.1524 ± 0.0236 | |
| RotatE (Sun 2019) | 0.0828 ± 0.0120 | |
| **KGFuse（旗舰）** | **0.2018 ± 0.0277** | **+17.56% vs 最强基线** |

门槛：相对提升 ≥ 5% → **PASS**；显著性：0.0301 > 0.0186 → **SIGNIFICANT**。

> 随机对照参考：300 实体下随机排名的 MRR ≈ ln(300)/300 ≈ **0.019**。
> 所有单模型均显著高于随机；KGFuse 约为随机的 **10.6 倍**。

## 消融（全部保留，含负结果）

**融合策略对照**（同一次运行的同一批模型）：

| 策略 | MRR | 结论 |
|---|---|---|
| **rrf** | **0.2018** | ✅ 采用 |
| mean_rr | 0.1873 | 次优 |
| max_consensus | 0.1166 | ❌ **负结果：低于最强单模型 0.1717** |

> max-consensus 被否决的原因（诚实记录）：逐候选取最大会把**所有候选**（含错误候选）
> 一起抬高，真实体排名反而变差。这与"只取真实体上各模型 rr 的最大值"不是一回事 ——
> 后者不可实现（需要 oracle），前者没有理论保证。已作为反例保留在代码与报告中。

**去掉任一基模型**（drop-one，值越低说明被去掉的模型贡献越大）：

| 去掉谁 | 融合 MRR |
|---|---|
| 去掉 DistMult | 0.1842 ← 贡献最大 |
| 去掉 ComplEx | 0.1855 |
| 去掉 TransE | 0.1926 |
| 去掉 RotatE | 0.1991 |
| **全保留** | **0.2018** |

四个模型**全部**为正贡献。

## 失败案例分析（从结果派生，非手写）

| 关系类型 | KGFuse MRR | 归因 |
|---|---|---|
| antisymmetric | 0.0193 | DistMult 结构对称、方向盲；数据量不足以让 TransE/RotatE 学到方向 |
| one-to-many | 0.0211 | 高扇出稀释信号，候选集熵大 |
| composition | 0.0600 | 一阶组合模式需更长训练才能被 RotatE 稳定捕获 |
| symmetric | 0.7042 | 全部模型都能学到（最容易的模式） |

## 一键复现

```bash
git clone https://github.com/CJX0712/kgforge && cd kgforge
pip install -r requirements.lock.txt
python examples/run_demo.py          # 出 benchmark.json，含确定性二次校验
```

其他入口：

```bash
python -m kgforge.cli benchmark --seeds 7,42,123   # 完整基准
python -m kgforge.cli eval                          # 单图训练 + 评测
python -m kgforge.cli gen --out kg_synthetic.tsv    # 导出合成 KG
KG_DIM=64 KG_EPOCHS=800 python -m kgforge.cli benchmark   # 环境变量覆盖超参
```

`make demo` / `make test` / `make lint` / `make cov` 亦可。

## 架构（单向无环）

```
cli ──→ pipeline ──→ {data, models, eval} ──→ core
```

```
kgforge/
  core/       types · errors(E100~E500) · config(KG_* env 覆盖 + schema 校验) · interfaces(Protocol) · seed(确定性)
  data/       合成 KG 生成（4 类关系语义 + 难度旋钮）· TSV 载入/导出
  models/     TransE · DistMult · ComplEx · RotatE（统一"越大越合理"打分约定）
  eval/       filtered 链接预测：MRR / Hits@1,3,10（Bordes 2013 协议）
  pipeline/   KGFuse 融合旗舰 + KGFusePipeline + benchmark（多 seed / 消融 / 失败分析）
  cli.py      argparse 入口
examples/run_demo.py   端到端演示 + 确定性二次校验
tests/       38 单测（含解析梯度 vs 中心差分金标准）
```

详见 [docs/architecture.md](docs/architecture.md) · [docs/model_card.md](docs/model_card.md)

## SOTA 对标声明

基模型**不是自研**：TransE / DistMult / ComplEx / RotatE 均为已发表算法，
本仓库是其在纯 NumPy 下的确定性实现（RotatE 是 papers-with-code 链接预测方向的代表性 SOTA）。
之所以手写而非直接调用 PyTorch-BigGraph / DGL-KE 等库，是为了满足本系统的硬约束：
**零外部权重下载、CPU-only、离线可跑、逐位可复现**。
本系统的原创贡献是**融合层 KGFuse 及其策略对照与消融结论**，基模型仅作为被融合组件。

## 工程门禁

| 项 | 状态 |
|---|---|
| 单测 | 38 passed（含 4 模型 × 3 位置解析梯度 vs 中心差分，rel err < 1e-5） |
| 覆盖率 | 核心模块 ≥ 80% |
| Lint | `ruff check` + `ruff format --check` 双绿（**硬门禁，无 `\|\| true`**） |
| CI | py3.12 × 3.13 × ubuntu × windows + demo 冒烟 + 密钥扫描 |
| 确定性 | 同 seed 两次运行核心指标 max\|Δ\| = 0（BLAS 线程钉 1） |
| 性能预算 | demo 端到端 ≈ 44.5s ≤ 60s |
| 依赖 | 运行时**仅 numpy**；`requirements.lock.txt` 全量锁定 |
| 离线 | 无 torch、无 HuggingFace、无模型下载 |

## 已知限制（诚实披露）

1. **绝对 MRR 偏低是数据难度所致**，不是融合失效：合成 KG 仅 1000 条训练三元组覆盖
   300 实体 × 8 关系，极度稀疏；symmetric 之外的关系类型仍在 0.02~0.06 量级。
2. **max-consensus 已实测劣于基线**，仅作为反例保留，不作为旗舰。
3. 融合增益依赖基模型的**互补性**；若四个模型高度同质，增益会收敛到 0。
4. 评估用合成数据，未在 FB15k-237 / WN18RR 等真实数据集上验证（那些需要下载数据）。

## License

MIT © 2026 晨星 (CJX0712)
