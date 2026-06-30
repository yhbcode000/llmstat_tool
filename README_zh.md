# LLMStat — 面向人类受试者研究的校准型 LLM 估计器

[English](README.md)

基于 *《预训练大语言模型作为统计工具：平方损失下的受限风险等价性》* 论文方法的 Python 实现。

## 理论实际说了什么——以及没说什么

**简短回答：可以用——但仅限于特定且定义明确的场景。**

该论文框架确立了：

> LLM 是平方损失下条件期望的误设估计器。

因此，在以下条件下可作为统计工具使用：

- 任务是**可测量结果的预测**
- 目标是**条件均值或光滑泛函**
- 损失近似为**平方误差或 Bregman 型**

### LLM 作为有效统计工具的情况

| 使用场景 | 有效性原因 |
|----------|-------------|
| 调查响应预测 | LLM 估计有界响应的条件均值 |
| 平均处理效应近似（强假设下） | 条件均值的线性对比 |
| 条件均值估计 | 框架的直接目标 |
| 缺失值插补 | E[Y\|X] 的泛函估计器 |

这与核心抽象一致：`T(P) = E[Y | X]`。

LLM 降低成本的适用条件：

- 人类抽样费用高昂
- 方差主导测量误差
- 校准偏差可控

因此 LLM 的行为类似于：**条件期望的低成本蒙特卡洛替代品。**

根据校准协议：若偏差 `b̂` 较小、方差比 `λ̂` 有界、可识别性误差 `δ` 可控，则 LLM 可通过修正缩放替代人类样本。

### LLM 不可用作统计工具的情况

| 局限性 | 失败原因 |
|------------|-------------|
| 无结构的因果推断 | LLM 无法识别反事实效应、结构参数或干预机制 |
| 全新实验范式 | 若 P(新条件) ∉ 训练数据支撑集，KL 投影无信息，ε_rep 主导 |
| 机制发现 | LLM 仅近似结果，非生成过程——无法替代认知模型、因果图或机制实验 |

### 成本降低：可以，但仅限于特定范围

框架推导：总风险 = 不可约方差 + ε_rep² + o(1)。

成本降低有效的条件：

- ε_rep 较小或已校准
- δ 可控
- 模型类足够丰富

**解释：** LLM 充当**带有固定偏差下限的方差降低替代估计器**。通过用大规模合成样本替代重复的人类抽样来降低成本。

### 核心洞见

> LLM 不是实验的替代品。它们是**固定统计模型类中条件期望的低成本估计器。**

这比表面理解的窄得多——但在数学上是正确的。

### 实践要点

**可安全使用 LLM 的场景：**

- 调查模拟器（需校准）
- 期望估计器
- 类 bootstrap 增强工具
- 近似风险估计器

**不可使用 LLM 的场景：**

- 因果实验的替代品
- 真实人类行为生成器
- 新现象的验证来源

### 一句话总结

> 是的——LLM 可作为统计工具降低成本，但仅限于平方损失下经校准的误设条件期望估计器，而非实验或因果推断的通用替代品。

## 快速开始

```bash
# 安装
pip install -e .

# 复制环境配置
cp .env.example .env

# 运行双条件框架调查模拟
llmstat run --conditions 2 --calibration 500 --output results/
```

## 架构

- **`llmstat.connector`** — LLM API 层：Docker localai（默认）→ OpenAI（回退）
- **`llmstat.methodology`** — 核心校准/估计流水线
- **`llmstat.validation`** — TOST 等价性检验
- **`llmstat.report`** — 报告生成（表格 + 详细流程）

## LLM 后端

支持两种后端，通过 `.env` 中的 `LLMSTAT_BACKEND` 选择：

| 后端 | 默认 | 依赖 |
|---------|---------|----------|
| `localai` | ✅ | Docker |
| `openai` | 回退 | API key |

### LocalAI（Docker）

```bash
# 启动 localai（CPU）
docker compose --profile cpu up -d

# 拉取模型
docker compose exec localai-cpu local-ai run llama-3.2-3b-instruct
```

### OpenAI（回退）

在 `.env` 中设置 `LLMSTAT_BACKEND=openai` 和 `OPENAI_API_KEY`。

## 用法

```bash
# 双条件研究的校准与估计
llmstat run \
  --conditions "neutral,treatment" \
  --calibration 500 \
  --equivalence-margin 0.02 \
  --min-effect 0.05

# 从已有数据估计校准参数
llmstat calibrate --human-data human.csv --llm-data llm.csv

# 运行 TOST 验证
llmstat validate --human-estimate 0.76 --llm-estimate 0.78 --margin 0.02
```

---

## 示例

运行网页 A/B 测试演示（无需 API key，使用合成数据）：

```bash
python examples/web_ab_test.py
```

运行三个网页 A/B 测试场景的完整校准流水线：
标题 CTR（点击率）测试、落地页转化率测试、按钮颜色点击率测试。
详见 `examples/web_ab_test.py` 中完整注释的源代码。

## Citation

```bibtex
@article{yang_llmstat,
  title   = {Using Large Language Models as Low-Cost Statistical
             Estimators for Human-Response Data},
  author  = {Yang, Haobo},
  year    = {2025},
  eprint  = {2606.30372},
  archiveprefix = {arXiv},
  primaryclass  = {cs.AI},
  note    = {This repository is the companion implementation.}
}
```