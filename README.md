# Commerce Agent — AI 电脑硬件智能导购闭环系统

## 产品定位

面向 PC 装机小白的 AI 硬件导购助手。用户输入预算、核心硬件偏好或使用场景，
Agent 一键生成高性价比、兼容的整机搭配方案，并通过 Stripe 沙盒测试模式完成闭环下单。

## 项目结构

```
commerce-agent-project/
├── .env.example
├── .gitignore
├── CLAUDE.md
├── PROJECT.MD
├── README.md
├── requirements.txt
├── data/
│   └── products.json         # 45 件硬件数据库
└── src/
    ├── __init__.py
    ├── config.py             # python-dotenv 配置加载
    ├── agent/
    │   ├── __init__.py
    │   ├── app.py            # Streamlit UI + Agent 管线
    │   ├── prompt.py         # System Prompt + 预算权重
    │   └── tools.py          # 工具函数 + OpenAI schemas
    ├── payment/
    │   ├── __init__.py
    │   └── stripe_client.py  # Stripe Checkout Session
    └── observability/
        ├── __init__.py
        ├── langfuse_ctx.py   # Langfuse v4 追踪 + KPI 指标
        └── eval_metrics.py   # 评分计算 + Langfuse 拉取
```

## 快速开始

```bash
# 1. 创建虚拟环境
python -m venv .venv

# 2. 激活
# Windows PowerShell:  .venv\Scripts\Activate.ps1
# Windows CMD:         .venv\Scripts\activate.bat
# macOS / Linux:       source .venv/bin/activate

# 3. 安装依赖
pip install -r requirements.txt

# 4. 配置密钥
cp .env.example .env
# 编辑 .env 填入真实 API Key

# 5. 启动
streamlit run src/agent/app.py
```

浏览器访问 `http://localhost:8501`。

## 环境变量

| 变量 | 说明 |
|------|------|
| `LLM_API_KEY` | 阿里云百炼 Model Studio API Key |
| `LLM_ENDPOINT` | OpenAI 兼容端点 URL |
| `LLM_MODEL` | 模型名称（默认 `qwen-plus`） |
| `LANGFUSE_PUBLIC_KEY` | Langfuse 公钥 (`pk-lf-...`) |
| `LANGFUSE_SECRET_KEY` | Langfuse 私钥 (`sk-lf-...`) |
| `LANGFUSE_HOST` | Langfuse 地址 (`https://cloud.langfuse.com`) |
| `STRIPE_SECRET_KEY` | Stripe 测试密钥 (`sk_test_...`) |

## 核心技术栈

| 模块 | 方案 |
|------|------|
| LLM | qwen-plus（阿里云百炼，OpenAI 兼容协议） |
| Agent | 确定性管线：数据库预检索 → LLM 文案生成 |
| UI | Streamlit（聊天界面 + 侧边栏 KPI + 评估看板） |
| 可观测性 | Langfuse v4（Trace/Span + 评分） |
| 支付 | Stripe Checkout Session（测试模式，USD） |

## 架构

```
User Input → _auto_build_bundle()  从数据库配 8 件
          → LLM 生成推荐文案（基于数据库结果）
          → Stripe Checkout 链接嵌入回复
          → 评分推送 Langfuse → 侧边栏实时刷新
```

**关键设计决策**：不依赖 LLM 调工具决定配置。改为先由规则引擎从数据库检索最优组合，
注入 Prompt 作为上下文，LLM 仅负责撰写自然语言推荐。这避免了 LLM 跳过工具调用、
编造型号等问题，并确保下单按钮 100% 出现。

## Stripe 测试支付

沙盒模式，不会产生真实交易。在支付页面使用：

- 卡号：`4242 4242 4242 4242`
- 有效期：任意未来日期（如 `12/34`）
- CVC：任意 3 位数字

## 评估看板

侧边栏实时展示 5 项指标：

| 指标 | 范围 |
|------|------|
| 预算约束合规率 | 0–1 |
| 意图对齐度 | 0–5 |
| 回执率（品类覆盖） | 0–1 |
| 精确度（偏好匹配） | 0–1 |
| F1 | 0–1 |

评分在每次推荐后自动计算并推送 Langfuse，`st.rerun()` 后侧边栏即时更新。
