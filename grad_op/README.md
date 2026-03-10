# Mainline grad/LLM game snapshot

这个目录是从主仓库 `analysis/` 中抽取的一份“主线”代码快照，目标是只保留运行：

- `analysis/optimization/grad_game.py`
- `analysis/optimization/llm_game.py`

所需的最小依赖文件集合。

## 运行

建议在本目录下运行，避免输出文件散落到仓库根目录：

```bash
cd mainline_grad_llm

# 梯度博弈
python analysis/optimization/grad_game.py

# LLM 博弈（需要可用的 OpenAI-compatible API；参数在代码里配置）
python analysis/optimization/llm_game.py
```

`analysis/optimization/llm_game.py` 底部 `cfg = LLMGameConfig(...)` 中设置 `llm.preset/llm.model`，并导出对应的 API Key，例如：

```bash
cd mainline_grad_llm
export DEEPSEEK_API_KEY="..."
python analysis/optimization/llm_game.py
```
