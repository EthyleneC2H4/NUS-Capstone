# AGENT_GUIDE_ZH.md — NUS 毕业设计项目完整上下文文档

> **用途：** 本文档为任何参与本代码库工作的 AI Agent 提供完整的项目上下文。涵盖项目背景、架构设计、当前进展、实验结果、运行环境、待完成工作及下一步行动计划。  
> **最后更新：** 2026-06-14  
> **作者：** 王翊熹 (c2h4wang@u.nus.edu)，新加坡国立大学计算机科学系，大四本科生

> **2026-06-14 完整性说明：** 下文章节保留了有价值的历史信息，但其中
> “M5 尚未运行”的描述已经失效。M5 已评估五项技术，P7 异配性感知门控的
> legacy 三随机种子结果为 AUPR 0.8240 +/- 0.0044。当前项目状态与证据索引
> 以 `NUS-Capstone/PROJECT_STATE.md` 和
> `NUS-Capstone/results/RESULT_PROVENANCE.md` 为准。现有实验早于新的确定性
> split manifest。根据 2026-06-25 范围决策，固定划分 GPU 重新验证、依赖本地
> 缺失 HDF5 的直接同配性计算、最终模型 IG/GSEA 重跑均为可选未来工作，不再是
> 毕业交付必做项。最终结论应收敛到现有 legacy 证据能够支持的范围。

---

## 目录

1. [项目背景](#1-项目背景)
2. [代码仓库结构](#2-代码仓库结构)
3. [模型架构与核心代码](#3-模型架构与核心代码)
4. [数据流水线](#4-数据流水线)
5. [已完成的实验结果](#5-已完成的实验结果)
6. [改进方案 P0–P10](#6-改进方案-p0p10)
7. [论文现状](#7-论文现状)
8. [运行环境](#8-运行环境)
9. [待完成工作](#9-待完成工作)
10. [已知问题与关键发现](#10-已知问题与关键发现)
11. [实验运行指南](#11-实验运行指南)
12. [关键参考文献](#12-关键参考文献)
13. [决策日志](#13-决策日志)
14. [论文-数据映射：缺失内容与对应位置](#14-论文-数据映射缺失内容与对应位置)
15. [附录](#15-附录)

---

## 1. 项目背景

### 1.1 研究问题

**基于图神经网络 (GNN) 的癌症驱动基因预测**——利用多层蛋白质-蛋白质相互作用 (PPI) 网络与泛癌多组学数据进行计算预测。

癌症驱动基因是指其突变直接导致癌症发生发展的基因。计算识别这类基因面临以下挑战：

| 挑战 | 说明 |
|------|------|
| **标签稀缺** | ~20,000 个蛋白编码基因中仅 ~700 个已知癌症基因（约 3.5%） |
| **类别不平衡** | 正负样本比约 1:30 |
| **多源数据** | 6 个独立的 PPI 网络数据库提供互补的拓扑信息 |
| **异质性动机 (Heterophily)** | 既有研究指出，PPI 癌症驱动基因预测可能受到低同配性影响；项目数据集直接同配性测量已移为可选未来工作。 |

### 1.2 基线方法 — EMGNN

本项目扩展自 **EMGNN**（Explainable Multilayer Graph Neural Network，Chatzianastasis 等，Bioinformatics 2023）。EMGNN 采用两阶段架构：

1. **逐网络 GNN 编码：** 每个 PPI 网络由独立的 GNN（GCN/GIN/GAT）处理，生成节点嵌入
2. **元图 (Meta-Graph) 构建：** 将所有网络中的节点嵌入映射到唯一的基因级「元节点」（跨网络基因的并集）
3. **元图 GNN + 分类器：** 第二层 GCN 在元图上运行，然后通过 MLP 进行癌症/非癌症二分类

### 1.3 五大方法论

| 编号 | 名称 | 目标 |
|------|------|------|
| **M1** | 基准复现 | 在 GCN/GIN/GAT 骨干网络和 6 个 PPI 网络上复现 EMGNN 结果 |
| **M2** | 模型优化 | 消融实验 + 贝叶斯超参数搜索（Optuna，50 轮试验） |
| **M3** | 多网络扩展 | 6 个 PPI 数据库上可学习的逐网络重要性权重 |
| **M4** | 可解释性增强 | 积分梯度 (Integrated Gradients) 特征归因 + GSEA 通路富集分析 |
| **M5** | 高级技术 | 11 项模块化扩展（P0–P10）：Focal Loss、位置编码、GraphMAE、GPS Transformer、跨网络注意力、超图 GNN、HIPGNN、PINNACLE、DropEdge、异质性感知、GNNExplainer |

### 1.4 六大 PPI 网络

| 网络 | 描述 | 节点数（约） | 来源 |
|------|------|-------------|------|
| CPDB | ConsensusPathDB | ~15K | 多数据库共识 |
| IREF_2015 | iRefIndex 2015 版 | ~13K | 文献整理的相互作用 |
| IREF | iRefIndex 最新版 | ~17K | 文献整理的相互作用 |
| STRING | STRINGdb | ~18K | 多证据通道综合评分 |
| PCNET | Parsimonious Composite Network | ~19K | 经质量过滤的整合网络 |
| MULTINET | MultiNet | ~14K | 多证据网络 |

### 1.5 节点特征

每个基因有 64 维多组学特征，由 4 种组学类型 × 16 种 TCGA 癌症类型组成：

| 特征组 | 缩写 | 描述 | 维度 |
|--------|------|------|------|
| 突变频率 | MF | 各癌症类型的体细胞突变率 | 16 |
| DNA 甲基化 | METH | 启动子区甲基化水平（beta 值） | 16 |
| 基因表达 | GE | RNA-seq 表达量 | 16 |
| 拷贝数变异 | CNA | 拷贝数变化 | 16 |

---

## 2. 代码仓库结构

**本地根目录：** `/Users/ethylene/Learning/NUS/Sem2/Capstone/capstone/`  
**GitHub 地址：** `git@github.com:EthyleneC2H4/NUS-Capstone.git`

```
capstone/                              # 项目根目录
├── NUS-Capstone/                      # 主代码库（git 管理，推送至 GitHub）
│   ├── src/                           # 源代码模块
│   │   ├── models/
│   │   │   ├── emgnn_improved.py      # ★ 核心模型 (21.5 KB, ~452 行)
│   │   │   ├── hypergnn.py            # 超图 GNN 通路编码器 (P5)
│   │   │   ├── hipgnn.py              # HIPGNN 异常检测辅助头 (P6)
│   │   │   └── baselines.py           # 简单 GCN/MLP 基线模型
│   │   ├── data/
│   │   │   ├── loader.py              # 多网络数据加载器 (224 行)
│   │   │   ├── feature_engineering.py # Z-score、PCA、特征选择
│   │   │   ├── build_hypergraph.py    # GMT → 超图关联矩阵 (P5)
│   │   │   └── pinnacle_embeddings.py # 加载 PINNACLE 蛋白质嵌入 (P8)
│   │   ├── training/
│   │   │   ├── trainer.py             # 训练循环：学习率调度 + 早停 + 梯度裁剪
│   │   │   ├── hparam_search.py       # Optuna 贝叶斯超参数搜索
│   │   │   └── pretrain_graphmae.py   # GraphMAE 自监督预训练 (P2)
│   │   └── explainability/
│   │       ├── attribution.py         # 积分梯度归因 (Captum)
│   │       └── gsea.py                # GSEA: Enrichr ORA + 预排序 + Hallmark 重叠
│   │
│   ├── experiments/                   # 可执行的实验脚本
│   │   ├── run_benchmark.py           # M1: 原始 EMGNN 基准实验
│   │   ├── run_improved.py            # ★ M2/M3/M5: 主训练脚本 (40+ CLI 参数)
│   │   ├── run_hparam_search.py       # M2: Optuna 搜索驱动器
│   │   ├── run_attribution.py         # M4: 特征归因分析
│   │   ├── run_gsea.py                # M4: 基因集富集分析
│   │   ├── run_gnn_explain.py         # P10: GNNExplainer 边级解释
│   │   ├── run_m5_ablation.sh         # ★ M5 消融实验：8 个实验 × 3 个种子
│   │   └── run_m5_explain.sh          # P10: GNNExplainer 封装脚本
│   │
│   ├── benchmark/                     # 原始 EMGNN 代码（参考用，只读）
│   │   ├── model.py                   # 原始 EMGNN/GCN/MLP 模型定义
│   │   ├── train.py                   # 原始训练脚本
│   │   ├── explain.py                 # 原始积分梯度解释器
│   │   ├── gcnIO.py                   # HDF5 I/O 工具函数
│   │   └── captum_custom.py           # Captum 边/节点掩码封装
│   │
│   ├── configs/
│   │   ├── benchmark_config.yaml      # M1 超参数配置
│   │   └── improved_config.yaml       # M2/M3 优化后的超参数配置
│   │
│   ├── results/                       # 实验输出（大部分被 git-ignore）
│   │   ├── my_models/                 # ~60 个已训练模型目录（git-ignore）
│   │   ├── experiment_summary.md      # ★ 所有结果汇总
│   │   ├── hparam_search_results.csv  # Optuna 最优试验
│   │   ├── results.txt                # 基准实验结果
│   │   ├── results_improved.txt       # 改进模型结果
│   │   └── network_weights.txt        # 学习到的逐网络权重
│   │
│   ├── LaTeX/
│   │   ├── main.pdf                   # ★ 编译好的论文（提交至 git）
│   │   ├── main.tex                   # LaTeX 源文件（git-ignore）
│   │   ├── sections/*.tex             # 论文各章节（git-ignore）
│   │   └── figures/                   # 图片 (enrichr_barplot.pdf)
│   │
│   ├── requirements.txt               # Python 依赖
│   ├── README.md                      # 项目文档
│   ├── .gitignore                     # 排除 .pkl, .h5, .tex, 模型目录
│   └── LICENSE
│
├── EMGNN/                             # 原始 EMGNN 参考仓库（独立）
├── results/                           # 实验输出的本地副本
├── LaTeX/                             # LaTeX 本地工作副本
├── logs/                              # 文档与进度记录
│   ├── experiment_summary.md          # 实验结果汇总
│   ├── future_improvements.md         # ★ 11 项改进方案 (P0-P10) 详细说明
│   ├── progress_assessment_2026-05-07.md # ★ 方法论评估 + 差距分析
│   ├── implementation_guide.md        # 实现指南
│   ├── server_run_guide.md            # 服务器环境搭建指南
│   └── ...
├── config/
│   └── config_autodl.md               # AutoDL 服务器 SSH 凭证
├── mid-term/                          # 期中报告与展示
└── ref/                               # 参考论文 (PDF)
```

### .gitignore 关键规则

- `*.pkl`, `*.h5`, `*.npz` — 二进制数据文件排除
- `results/my_models/` — 训练模型目录排除（文件过大）
- `LaTeX/*.tex`, `LaTeX/sections/` — 仅提交 PDF，不提交 LaTeX 源文件
- `*.sh` 排除，**但保留** `experiments/run_m5_*.sh`

---

## 3. 模型架构与核心代码

### 3.1 EMGNNImproved 架构 (`src/models/emgnn_improved.py`)

核心模型包含 11 项可模块化开关的改进：

```
输入: 6 个 PPI 图 × (节点, 64 维特征, 边索引)
                    │
        ┌───────────┼───────────┐
        ▼           ▼           ▼       (× 6 个网络)
   [输入线性投影]                         nfeat → hidden_channels
        │
   [GCN/GIN/GAT/SAGE 层 × n_layers]
   · 可选: 残差连接 (residual connections)
   · 可选: 归一化类型 (batch|graph|layer|none)
   · 可选: DropEdge (P9)
   · 可选: 异质性感知门控 (P7)
        │
   [逐网络加权]
   · 标量 softmax 权重（默认）
   · 或 跨网络注意力 (P4): 基因级注意力 via scatter_softmax
        │
        └──────→ 元图构建
                 (跨网络基因的并集)
                    │
             [元图 GNN]
             · 标准 GCNConv（默认）
             · 或 GPS Graph Transformer (P3)
                    │
             [可选: 超图融合 (P5)]
             · HyperGNNEncoder 编码 GO/KEGG 基因集
             · 通过学习门控融合: h_final = gate * h_meta + (1-gate) * h_hyper
                    │
             [MLP 分类器 → log_softmax]
                    │
             [损失函数: NLL / Focal Loss (P0)]
             · + 标签平滑 (label smoothing, 可选)
             · + HIPGNN 辅助损失 (P6, 多任务学习)
```

### 3.2 核心类

| 类名 | 文件 | 功能 |
|------|------|------|
| `EMGNNImproved` | `emgnn_improved.py` | 包含所有改进开关的主模型 |
| `FocalLoss` | `emgnn_improved.py` | 类别不平衡损失函数 (gamma, alpha) |
| `CrossNetworkAttention` | `emgnn_improved.py` | 基因级逐网络注意力 |
| `HighLowPassSeparation` | `emgnn_improved.py` | 异质性感知的高通/低通门控融合 |
| `HyperGNNEncoder` | `hypergnn.py` | 多层超图编码器 |
| `HIPGNNAuxHead` | `hipgnn.py` | 频谱 + 空间异常检测 |
| `GraphMAE` | `pretrain_graphmae.py` | 掩码自编码器预训练 |
| `Trainer` | `trainer.py` | 训练循环：学习率调度、早停、梯度裁剪 |
| `FeatureEngineer` | `feature_engineering.py` | Z-score/MinMax 标准化、方差过滤、可选 PCA |

### 3.3 命令行参数 (`experiments/run_improved.py`)

**骨干网络选择：** `--gcn`, `--gat`, `--gin`, `--sage`, `--mlp`（布尔标志）

**数据集：** `--dataset IREF_2015 IREF STRING PCNET MULTINET CPDB`（最后一个为测试集）

**架构控制参数：**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--use_residual` | True | 残差连接 |
| `--norm_type` | batch | batch/graph/layer/none |
| `--use_net_weights` | True | 可学习逐网络重要性 |
| `--label_smoothing` | 0.05 | 标签平滑 epsilon |
| `--cross_network_attention` | False | P4: 基因级跨网络注意力 |
| `--heterophily_aware` | False | P7: 高通/低通门控 |
| `--focal_gamma` | 0.0 | P0: Focal Loss gamma（0 = 关闭） |
| `--focal_alpha` | 0.75 | P0: 正类权重 |
| `--pe_dim` | 0 | P1: 随机游走位置编码维度 |
| `--drop_edge_rate` | 0.0 | P9: 边 Dropout 比例 |
| `--gps_meta` | False | P3: 元图使用 GPS Transformer |
| `--gps_heads` | 4 | P3: 注意力头数 |
| `--pretrain_graphmae` | False | P2: 启用 GraphMAE 预训练 |
| `--pretrain_epochs` | 200 | P2: 预训练轮数 |
| `--hypergraph_gmt` | None | P5: GMT 文件路径（超图） |
| `--hipgnn_lambda` | 0.0 | P6: 异常检测辅助损失权重 |
| `--hipgnn_eigvecs` | 32 | P6: 拉普拉斯特征向量数 |
| `--pinnacle_path` | None | P8: PINNACLE .npz 嵌入路径 |

**训练超参数：**

| 参数 | 默认值 | 描述 |
|------|--------|------|
| `--lr` | 0.005 | 学习率 |
| `--weight_decay` | 5e-4 | L2 正则化 |
| `--hidden` | 64 | 隐藏层维度 |
| `--n_layers` | 3 | GNN 层数 |
| `--dropout` | 0.5 | Dropout 率 |
| `--epochs` | 2000 | 最大训练轮数 |
| `--patience` | 250 | 早停耐心值 |
| `--lr_scheduler` | cosine | cosine/step/none |
| `--seed` | 72 | 随机种子 |

---

## 4. 数据流水线

### 4.1 数据来源

多组学节点特征以 HDF5 格式存储，来自 EMOGI 基准数据集（Zenodo）。

**服务器路径：** `/root/NUS-Capstone/results/EMOGI_*/`

| 文件 | 网络 |
|------|------|
| `EMOGI_CPDB/CPDB_multiomics.h5` | CPDB |
| `EMOGI_IRefIndex/IREF_multiomics.h5` | IREF |
| `EMOGI_IRefIndex_2015/IREF_2015_multiomics.h5` | IREF_2015 |
| `EMOGI_Multinet/MULTINET_multiomics.h5` | MULTINET |
| `EMOGI_PCNet/PCNET_multiomics.h5` | PCNET |
| `EMOGI_STRINGdb/STRINGdb_multiomics.h5` | STRING |

每个 HDF5 文件包含：
- `network`: 邻接矩阵（稠密或稀疏）
- `features`: 节点特征（64 维多组学）
- `node_names`: 基因符号标识
- `y_train`, `y_test`: 二分类标签

### 4.2 数据加载流程 (`src/data/loader.py`)

```
load_multi_network_data(dataset_names, ...)
  │
  ├─ 对每个网络:
  │   ├─ 读取 HDF5 → 特征、邻接矩阵、标签
  │   ├─ 稀疏缓存: 稠密邻接 → scipy.sparse → .npz（首次运行生成）
  │   ├─ 特征对齐: 重排为标准顺序 [MF, METH, GE, CNA] × 16 类型
  │   ├─ 可选: FeatureEngineer.fit_transform()
  │   ├─ 可选: AddRandomWalkPE (pe_dim)
  │   └─ 构建 PyG Data(x, edge_index, y) + 自环
  │
  ├─ 构建 node2idx: {(db_id, gene_symbol): unique_meta_idx}
  ├─ 构建 meta_x: (n_unique_genes, feat_dim) — 跨网络平均特征
  ├─ 构建 meta_y: (n_unique_genes,) — 二分类标签
  ├─ 数据划分: 训练（前 N-1 个图）/ 验证（训练集的 10%）/ 测试（最后一个图）
  │
  └─ 返回: DataLoader, info 字典
```

### 4.3 外部数据（尚未获取）

| 数据 | 来源 | 用于 | 状态 |
|------|------|------|------|
| GMT 基因集 | MSigDB (Hallmark/C2) | P5 超图 GNN | **未下载** |
| 拉普拉斯特征向量 | 由图拉普拉斯计算 | P6 HIPGNN | **未预计算** |
| PINNACLE 嵌入 | HuggingFace / Nature Methods 2024 | P8 PINNACLE | **未下载** |

---

## 5. 已完成的实验结果

### 5.1 M1 — 基准复现

| 网络 | 最优 AUPR | 最优 AUROC | 平均 AUPR | 运行次数 |
|------|-----------|-----------|-----------|---------|
| CPDB | **0.7528** | 0.8712 | 0.7432 | 18 |
| STRING | 0.7588 | 0.8895 | 0.7391 | 6 |
| IREF_2015 | 0.7582 | 0.8795 | 0.7574 | 2 |
| MULTINET | 0.7835 | **0.9336** | 0.7760 | 2 |
| IREF | 0.6935 | 0.8968 | 0.6891 | 2 |
| PCNET | 0.7458 | 0.9296 | 0.7413 | 2 |

多骨干网络对比 (CPDB): GIN (0.7918) > GCN (0.7528) > GAT (0.6158)  
多网络 (3 网络基准 GCN): AUPR 0.7877, AUROC 0.9041

### 5.2 M2 — 模型优化

**消融实验发现：**
- BatchNorm1d 在全图批训练中**有害**：-4.2% AUPR
- 推荐：残差连接 + 不使用 BatchNorm
- 标签平滑 (epsilon=0.05) 效果边际

**Optuna 最优配置（50 轮试验，单种子 seed=72）：**
- AUPR = **0.8023**（hidden=32, n_layers=4, dropout=0.211, 无 BN, step LR）
- **警告：** 多种子验证 (seeds 1-5) 均值仅 0.7424 ± 0.008——0.8023 这个结果**依赖于特定种子，不具鲁棒性**

### 5.3 M3 — 多网络扩展

| 配置 | AUPR | AUROC | 说明 |
|------|------|-------|------|
| 基准 GCN, 仅 CPDB | 0.7479 | 0.8668 | 基线 |
| 基准 GCN, 全 6 网络 | **0.7987** | **0.9114** | 纯数据效应: +0.054 |
| EMGNNImproved, 仅 CPDB | 0.7540 | 0.8615 | 架构仅贡献 +0.006 |
| EMGNNImproved, 2 网络 (IREF_2015+CPDB) | 0.8018 | 0.9000 | 仅 2 网络即接近最优 |
| **EMGNNImproved, 全 6 网络** | **0.8067** | **0.9170** | **最优结果: +5.9%** |

**增益分解：** 数据贡献 +0.054 AUPR；架构仅贡献 +0.008。

**学习到的网络权重 (softmax 归一化)：**
- CPDB: 0.210, MULTINET: 0.201, IREF_2015: 0.166, STRING: 0.166, PCNET: 0.162, IREF: 0.096

### 5.4 M4 — 可解释性

**特征重要性排名（积分梯度）：**
1. METH:LIHC (0.908), 2. GE:BLCA (0.805), 3. GE:BRCA (0.778)
- DNA 甲基化 + 基因表达主导（前 10 中占 8 个）

**预测排名最高的癌症基因：** TP53, MUC16, TTN, CTNNB1, EP300, PIK3CA — 全部为已知癌症驱动基因

**GSEA 通路富集：** 28 条显著的 Hallmark 通路 (FDR < 0.05)：
- 上皮-间质转化 EMT (FDR=1.6×10^-33), PI3K/AKT/mTOR, 凋亡 (Apoptosis), WNT, TGF-beta

### 5.5 M5 — 高级技术

**状态：代码已实现，实验尚未运行。**

全部 11 项改进 (P0-P10) 的代码已编写并提交。消融实验脚本 (`run_m5_ablation.sh`) 已准备就绪。服务器代码已同步并验证通过。

---

## 6. 改进方案 P0–P10

### 已实现 & 可直接运行消融（无需外部数据）

| 编号 | 名称 | CLI 参数 | 难度 | 预期 AUPR 提升 |
|------|------|----------|------|---------------|
| P0 | Focal Loss | `--focal_gamma 2.0 --focal_alpha 0.75` | 低 | +2-4% |
| P1 | 随机游走位置编码 | `--pe_dim 16` | 低 | +1-3% |
| P2 | GraphMAE 自监督预训练 | `--pretrain_graphmae 1 --pretrain_epochs 200` | 中 | +3-5% |
| P3 | GPS Graph Transformer | `--gps_meta 1 --gps_heads 4` | 中 | +2-5% |
| P4 | 跨网络注意力融合 | `--cross_network_attention 1` | 中 | +1-3% |
| P7 | 异质性感知 GNN | `--heterophily_aware 1` | 中 | +2-4% |
| P9 | DropEdge 正则化 | `--drop_edge_rate 0.1` | 低 | +0.5-1.5% |
| P10 | GNNExplainer | 独立脚本 | 低 | 不适用（增强可解释性） |

### 需要外部数据（已推迟）

| 编号 | 名称 | CLI 参数 | 所需数据 | 状态 |
|------|------|----------|----------|------|
| P5 | 超图 (GO/Pathway) | `--hypergraph_gmt <路径>` | MSigDB GMT 文件 | 未下载 |
| P6 | HIPGNN 异常检测 | `--hipgnn_lambda 0.1 --hipgnn_eigvecs 32` | 预计算的拉普拉斯特征向量 | 未计算 |
| P8 | PINNACLE 蛋白质嵌入 | `--pinnacle_path <路径>` | HuggingFace 上的 PINNACLE .npz | 未下载 |

---

## 7. 论文现状

**标题：** "Cancer Driver Gene Prediction via Explainable Multilayer Graph Neural Networks with Multi-Network Integration and Pathway Enrichment Analysis"

**位置：** `LaTeX/main.pdf`（17 页，单栏格式，A4）

**当前章节结构（2026-05-30 修订版）：**

| # | 文件 | 章节 | 状态 |
|---|------|------|------|
| 0 | `00_frontmatter.tex` | 标题、作者、摘要（~200词）、关键词 | ✅ 完成 |
| 1 | `01_introduction.tex` | 引言，含 5 项编号贡献 | ✅ 完成 |
| 2 | `02_related_work.tex` | 相关工作 + 对比表格 | ✅ 完成 |
| 3 | `03_methods.tex` | 方法（合并理论+方法论，全部数学公式，M1–M5 方法描述，**Fig. 1 架构图**） | ✅ 完成 |
| 4 | `04_results.tex` | 实验结果：M1–M4 数据（7 张表、1 张 GSEA 图）、M5 实现状态、讨论 | ⚠️ M5 消融表格为**占位符** |
| 5 | `05_conclusions.tex` | 结论 + 6 项局限性 | ✅ 完成 |
| 6 | `06_declarations.tex` | 致谢、资金、伦理、代码/数据可用性 | ✅ 完成 |
| 7 | `07_references.tex` | 33 篇编号引用（手动 cite 系统） | ✅ 完成 |

**图表：** 2 张（Fig. 1: TikZ 架构图在方法部分；Fig. 2: GSEA 富集分析柱状图在结果部分）

**论文待解决问题：**
- [ ] M5 消融实验结果表格 — `04_results.tex` 中标有 PLACEHOLDER（见下方第 14 节）
- [ ] M3 六网络结果的多种子方差 — 当前仅报告单种子结果
- [ ] 额外可视化（特征重要性热力图、训练曲线）— 锦上添花
- [ ] 在 6 网络模型上重新运行 IG/GSEA — 当前 M4 结果来自单网络模型（论文中已标注局限性）

---

## 8. 运行环境

### 8.1 本地机器

- macOS Darwin 24.6.0, Apple Silicon
- Python 3.10+, PyTorch 2.2.2 (CPU)
- 用途：开发、论文写作、代码管理

### 8.2 云服务器 (AutoDL)

**连接方式：**
```bash
ssh -p 32197 root@connect.westd.seetacloud.com
# 密码: aNeGrlDbcwUu
```

**配置：**
- GPU: NVIDIA RTX 5090 (CUDA 12.8, sm_120)
- 操作系统: Ubuntu, PyTorch 2.7, PyG 2.7
- Conda 环境: `cancer-gnn`
- Python 路径: `/root/miniconda3/envs/cancer-gnn/bin/python`
- 项目路径: `/root/NUS-Capstone/`

**重要注意事项：**
- 服务器按分钟计费 — **实验完成后立即关机**
- 关机命令: `shutdown now`（SSH 执行）
- **GitHub 镜像 (ghproxy.com) 在服务器上不可靠** — 使用 SCP 同步代码而非 git pull
- 服务器代码最后一次 SCP 同步于 2026-05-22（全部 11 个文件已验证）
- 无卡模式可用（非 GPU 准备工作，费用较低）

### 8.3 SCP 代码同步流程

当服务器无法访问 GitHub 时：
```bash
# 从本地机器执行:
sshpass -p 'aNeGrlDbcwUu' scp -P 32197 \
  /Users/ethylene/Learning/NUS/Sem2/Capstone/capstone/NUS-Capstone/src/models/emgnn_improved.py \
  root@connect.westd.seetacloud.com:/root/NUS-Capstone/src/models/emgnn_improved.py
```

### 8.4 依赖列表

```
torch==2.2.2 (本地 CPU) / 2.7 (服务器 GPU)
torch-geometric==2.7.0
captum==0.8.0
numpy==1.26.4
scipy==1.12.0
scikit-learn==1.7.2
pandas==2.2.2
h5py==3.11.0
matplotlib==3.8.4
optuna==4.8.0
gseapy==1.1.13
```

---

## 9. 待完成工作

### 9.1 高优先级 — M5 消融实验

**状态：** 脚本已就绪，服务器代码已同步，**尚未运行**。

消融脚本 `experiments/run_m5_ablation.sh` 测试 8 个实验 × 3 个种子 (72, 1, 2)：

| 实验名 | 改进项 | 额外参数 |
|--------|--------|----------|
| baseline | 无（6 网络参考基线） | — |
| P0_focal_loss | Focal Loss | `--focal_gamma 2.0 --focal_alpha 0.75` |
| P9_dropedge | DropEdge | `--drop_edge_rate 0.1` |
| P7_heterophily | 异质性感知 | `--heterophily_aware 1` |
| P2_graphmae | GraphMAE 预训练 | `--pretrain_graphmae 1 --pretrain_epochs 200` |
| P1_pe_dim16 | 位置编码 | `--pe_dim 16` |
| P3_gps_meta | GPS Transformer | `--gps_meta 1 --gps_heads 4` |
| P4_cross_net_attn | 跨网络注意力 | `--cross_network_attention 1` |

所有实验共享基础配置：GCN 骨干、全 6 网络、norm_type=none、残差=True、cosine 学习率调度、label_smoothing=0.05、hidden=64、n_layers=3。

**预计运行时间：** RTX 5090 上约 8-9 小时。

**启动命令：**
```bash
cd /root/NUS-Capstone
nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &
```

**输出文件：** `results/m5_ablation_summary.csv`（实验名、种子、AUPR、AUROC、耗时）

### 9.2 中优先级 — GNNExplainer (P10)

消融实验完成后，在最优模型上运行：
```bash
bash experiments/run_m5_explain.sh results/my_models/<最优模型目录>
```

### 9.3 中优先级 — 论文更新

M5 实验完成后：
1. 在论文第 4 章（实验结果）添加 M5 消融实验表格
2. 更新摘要中的 M5 发现
3. 添加模型架构图（目前仅 1 张图片）
4. 更新结论章节
5. 重新编译 PDF 并推送至 GitHub

### 9.4 低优先级 — 外部数据实验

| 任务 | 所需数据 | 获取方式 |
|------|----------|----------|
| P5 超图 | MSigDB GMT 文件 | 从 msigdb.org 下载 (Hallmark + C2 集合) |
| P6 HIPGNN 特征向量 | 图拉普拉斯预计算 | 调用 `hipgnn.py` 中的 `precompute_laplacian_eigvecs()` |
| P8 PINNACLE | 128 维蛋白质嵌入 (.npz) | 从 HuggingFace 下载 (Li et al. Nature Methods 2024) |

### 9.5 尚未准备的交付物

- [ ] 期末答辩展示幻灯片（期中幻灯片在 `mid-term/`）
- [ ] 提交用代码打包文件
- [ ] 在 6 网络最优模型上重新运行 IG/GSEA（当前 M4 结果来自单网络模型）

---

## 10. 已知问题与关键发现

### 10.1 BatchNorm 有害

**发现：** BatchNorm1d 在全图批训练中导致 -4.2% AUPR 性能下降。

**原因：** 全图批训练中，整个图就是一个「batch」。训练时在全图上计算的 running statistics 在推理时无法提供有意义的归一化。模型使用全图统计量训练，但使用 running-mean 统计量推理，两者不一致。

**处理：** 始终使用 `--norm_type none` 或 `--norm_type layer`。**绝不使用** `--norm_type batch`。

### 10.2 Optuna 结果依赖种子

Optuna 找到的最优配置 (seed=72 时 AUPR=0.8023) 在 seeds 1-5 上均值仅为 0.7424 ± 0.008。这表明该结果是对特定训练/验证划分的过拟合，**并非鲁棒的超参数改进**。

### 10.3 多网络增益主要来自数据而非架构

基准 GCN 在 6 网络上达到 AUPR=0.7987。EMGNNImproved 架构仅额外贡献 +0.008。+5.9% 的总增益大部分来自整合更多 PPI 数据。

### 10.4 CPDB 循环依赖

CPDB 同时作为输入网络和测试集。CPDB 学到的高权重 (0.210) 可能被这种循环依赖所放大。

### 10.5 服务器 GitHub 镜像不可靠

`mirror.ghproxy.com` 在 AutoDL 服务器上超时。请使用 SCP 同步代码，不要用 `git pull`。

### 10.6 非交互式 Shell 中的 Conda 路径

服务器上的 Shell 脚本必须使用完整 Conda Python 路径（`/root/miniconda3/envs/cancer-gnn/bin/python`），因为 `nohup` 在非交互模式下运行，conda 不会被激活。

---

## 11. 实验运行指南

### 11.1 M5 消融实验（主要待运行实验）

```bash
# 1. 启动 AutoDL 服务器（带 GPU）
# 2. SSH 连接到服务器
ssh -p 32197 root@connect.westd.seetacloud.com

# 3. 启动消融实验
cd /root/NUS-Capstone
nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &

# 4. 监控进度
tail -f results/m5_ablation.log

# 5. 完成后（约 8-9 小时）查看结果
cat results/m5_ablation_summary.csv

# 6. 关闭服务器以停止计费
shutdown now
```

### 11.2 单个实验运行

```bash
/root/miniconda3/envs/cancer-gnn/bin/python experiments/run_improved.py \
  --gcn 1 \
  --dataset IREF_2015 IREF STRING PCNET MULTINET CPDB \
  --norm_type none --use_residual True --use_net_weights True \
  --lr_scheduler cosine --label_smoothing 0.05 \
  --hidden 64 --n_layers 3 --dropout 0.5 \
  --epochs 2000 --patience 250 \
  --seed 72 \
  --focal_gamma 2.0 --focal_alpha 0.75  # 示例: 启用 Focal Loss
```

### 11.3 GNNExplainer

```bash
bash experiments/run_m5_explain.sh results/my_models/<模型目录>
```

### 11.4 GSEA 分析

```bash
python experiments/run_gsea.py \
  --model_dir results/my_models/<模型目录> \
  --mode enrichr \
  --top_n 200
```

---

## 12. 关键参考文献

| 论文 | 年份 | 发表 | 与本项目的关系 |
|------|------|------|---------------|
| EMOGI (Chatzianastasis 等) | 2023 | Bioinformatics | **基线方法** — 多层图神经网络 |
| Focal Loss (Lin 等) | 2017 | ICCV | P0 — 类别不平衡处理 |
| GraphMAE (Hou 等) | 2022 | KDD | P2 — 自监督预训练 |
| GPS (Rampasek 等) | 2022 | NeurIPS | P3 — Graph Transformer 架构 |
| R-GCN (Schlichtkrull 等) | 2018 | ESWC | P4 参考 — 关系图卷积 |
| DISHyper (Deng 等) | 2024 | Bioinformatics (ISMB) | P5 参考 — 疾病超图 |
| HIPGNN | 2025 | AAAI | P6 参考 — 异常检测视角 |
| SGCD | 2024 | Briefings in Bioinformatics | P7 参考 — PPI 中的异质性 |
| PINNACLE (Li 等) | 2024 | Nature Methods | P8 — 预训练蛋白质嵌入 |
| DropEdge (Rong 等) | 2020 | ICLR | P9 — 图结构正则化 |
| GNNExplainer (Ying 等) | 2019 | NeurIPS | P10 — 边级解释 |
| DGHNN (Li 等) | 2025 | Bioinformatics | 近期对比方法：GNN + HyperGNN + FT-Transformer |
| deepCDG | 2025 | Briefings in Bioinformatics | 近期竞品: 跨组学注意力 |

---

## 13. 决策日志

| 日期 | 决策 | 理由 |
|------|------|------|
| 2026-04-04 | 选用 GCN 作为主要骨干网络 | 跨种子最稳定（18 次运行验证） |
| 2026-04-15 | 禁用 BatchNorm | 在全图批 GNN 中有害：-4.2% AUPR |
| 2026-05-07 | 优先解决 G1（在 6 网络模型上重跑 IG/GSEA） | 对论文质量提升最大 |
| 2026-05-21 | 将 P0-P10 实现为模块化开关 | 支持干净的消融实验，无需代码分支 |
| 2026-05-22 | 使用 SCP 而非 git pull 同步服务器代码 | GitHub 镜像在 AutoDL 上不可靠 |
| 2026-05-22 | 在脚本中硬编码 Conda Python 路径 | 非交互式 Shell 无法激活 conda |
| 2026-05-22 | GitHub 仓库排除 LaTeX 源文件 | 仅需 PDF；源文件留在本地 |
| 2026-05-28 | M5 消融测试 8 个实验 × 3 个种子 | 平衡实验覆盖与计算成本 |

---

## 14. 附录

### 附录 A: 模型检查点

服务器上的关键模型目录 (`/root/NUS-Capstone/results/my_models/`)：

| 目录 | 描述 |
|------|------|
| `GCN_['IREF_2015', 'IREF', 'STRING', 'PCNET', 'MULTINET', 'CPDB']_2026_05_09_09_52_59` | 最优 6 网络模型 (AUPR=0.8067) |
| `EMGNNImproved_GCN_CPDB_2026_04_15_10_22_58` | M4 可解释性分析模型（单网络） |

每个目录包含：`model.pkl`, `args.pkl`, `batch.pkl`, `node2idx.pkl`, `predictions.tsv`, `hyper_params.txt`, `final_y.pkl`, `edge_index.pkl`

### 附录 B: Git 提交历史

```
c4d9945 Add M5 ablation experiment scripts for RTX 5090 server
0931429 Remove LaTeX source files from repo, keep only PDF
2050bd4 Update paper with Methodology 5: eleven advanced GNN techniques
e178447 Add LaTeX paper source and PDF to repository
d1650b3 P8: Add PINNACLE pretrained protein embedding support
91df9cd P6: Add HIPGNN-inspired anomaly detection auxiliary head
d65fb1f P5: Add GO/Pathway hypergraph integration
c721a9b P4: Add cross-network attention fusion for gene-level network weighting
3e84f1a P3: Add GPS Graph Transformer option for meta-graph layer
35dfcab P1: Add Random-Walk positional encoding for graph-position awareness
1f25253 P10: Add GNNExplainer for edge-level interpretability
7967d24 P2: Add GraphMAE self-supervised pretraining
c133ffb P7: Add heterophily-aware gated high/low-pass separation
7e27bdd P9: Add DropEdge regularization for graph-structure robustness
07fe2b4 P0: Add Focal Loss for class-imbalanced cancer gene prediction
7b965a9 Add G2/G4 validation results and update GSEA to 6-network model
...
3b45caf Initial commit
```

## 14. 论文-数据映射：缺失内容与对应位置

> **用途：** 本节明确标注每一项缺失实验数据在 LaTeX 论文中的精确位置，以便实验完成后可直接更新论文，无需重新通读全文。

### 14.1 M5 消融实验结果表格（高优先级）

| 项目 | 详情 |
|------|------|
| **缺失内容** | 8 种 M5 技术 × 3 个随机种子的消融实验结果 |
| **论文文件** | `LaTeX/sections/04_results.tex` |
| **论文位置** | 第 4.5 节 "Advanced GNN Techniques: Implementation Status" — 搜索 `%% PLACEHOLDER: M5 ABLATION RESULTS TABLE` |
| **操作步骤** | 1. 取消 PLACEHOLDER 下方 `\begin{table}` 块的注释。 2. 填入各行的 Mean AUPR、std、Mean AUROC、ΔAUPR。 3. 将前面段落从"尚未执行"改为讨论实际结果。 |
| **数据来源** | `results/m5_ablation_summary.csv`（由 `run_m5_ablation.sh` 生成） |
| **服务器命令** | `cd /root/NUS-Capstone && nohup bash experiments/run_m5_ablation.sh > results/m5_ablation.log 2>&1 &` |
| **预计耗时** | RTX 5090 约 8–9 小时 |
| **同步更新** | (a) 摘要 `00_frontmatter.tex`：若有显著提升则补充 M5 结果。 (b) 结论 `05_conclusions.tex`：更新"M5 待评估"局限性。 (c) 讨论 `04_results.tex` 第 4.6 节：增加 M5 讨论子节。 |

### 14.2 M3 最优结果的多种子方差（中等优先级）

| 项目 | 详情 |
|------|------|
| **缺失内容** | 6 网络 EMGNNImproved 模型的多种子 (seeds 1–5) AUPR ± std |
| **论文文件** | `LaTeX/sections/04_results.tex` |
| **论文位置** | 第 4.3 节，表 4 (`tab:multinetwork`) — "EMGNNImproved, All 6 networks" 行当前为单种子结果 |
| **操作步骤** | 跑 5 个种子，计算均值 ± 标准差，更新表格并加脚注。同步更新 `05_conclusions.tex` 中"单种子评估"局限性。 |

### 14.3 GNNExplainer 边级别结果（中等优先级）

| 项目 | 详情 |
|------|------|
| **缺失内容** | 在最优模型上运行 GNNExplainer 的边重要性结果 |
| **论文文件** | `LaTeX/sections/04_results.tex` |
| **论文位置** | 当前无 GNNExplainer 结果章节。在 GSEA 子节后新增第 4.4.4 节 "Edge-Level Interpretability via GNNExplainer"。 |
| **服务器命令** | `bash experiments/run_m5_explain.sh results/my_models/<best_model_dir>` |

### 14.4 外部数据模块（较低优先级）

| 模块 | 缺失数据 | 获取方式 | 训练命令追加参数 |
|------|----------|----------|------------------|
| P5 超图 | MSigDB GMT 文件 | 从 gsea-msigdb.org 下载 | `--hypergraph_gmt data/msigdb_hallmark.gmt` |
| P6 HIPGNN | 拉普拉斯特征向量 | 调用 `hipgnn.py` 中的预计算函数 | `--hipgnn_lambda 0.1 --hipgnn_eigvecs 32` |
| P8 PINNACLE | 128 维蛋白质嵌入 | 从 HuggingFace 下载 | `--pinnacle_path data/pinnacle_embeddings.npz` |

### 14.5 快速参照：论文文件 → 缺失数据

| 论文文件 | 缺失内容 | 优先级 |
|----------|----------|--------|
| `00_frontmatter.tex` | M5 结果出来后更新摘要 | M5 之后 |
| `04_results.tex` 第 4.5 节 | M5 消融表格（PLACEHOLDER 已标记） | **高** |
| `04_results.tex` 第 4.3 节 | 6 网络结果的多种子方差 | 中 |
| `04_results.tex` 第 4.4 节 | GNNExplainer 结果子节 | 中 |
| `05_conclusions.tex` | 更新 "M5 待评估" 和 "单种子" 局限性 | M5 之后 |

---

### 附录 C: 新 Agent 快速入门

1. **首先阅读本文档**以获取完整上下文
2. **核心模型：** `NUS-Capstone/src/models/emgnn_improved.py`
3. **主训练脚本：** `NUS-Capstone/experiments/run_improved.py`
4. **所有实验结果：** `NUS-Capstone/results/experiment_summary.md`
5. **改进方案详述：** `logs/future_improvements.md`
6. **进度评估：** `logs/progress_assessment_2026-05-07.md`
7. **服务器配置：** `config/config_autodl.md`
8. **当前紧急任务：** 在服务器上运行 M5 消融实验，然后用结果更新论文
9. **论文-数据映射：** 见上方第 14 节，明确标注了每项缺失数据的论文位置和填充方法

### 附录 D: 项目关键数据一览

| 指标 | 值 |
|------|-----|
| 最优模型 | EMGNNImproved (GCN, 6 PPI 网络) |
| 最优 AUPR | **0.8067** (+5.9% vs 基线) |
| 最优 AUROC | **0.9170** |
| 正负样本比 | ~700 : ~20,000 (1:30) |
| 节点特征维度 | 64 (4 组学 × 16 癌症类型) |
| GNN 层数 | 3 |
| 隐藏层维度 | 64 |
| 训练设备 | NVIDIA RTX 5090 (CUDA 12.8) |
| 总代码行数 | ~6,756 行 Python |
| 训练模型数 | ~60 个 |
| 论文页数 | 17 页（单栏） |
| 参考文献数 | 33 篇 |
| 图表 | 2 张图 + 8 张表 |
| GitHub 仓库 | github.com/EthyleneC2H4/NUS-Capstone |
