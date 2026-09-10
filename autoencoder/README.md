# 子结构离散潜 token

`fragment_vq.py` 是独立的 fragment-level RVQ 自编码器。原有
`graphtransformer.py`、`dataprocess.py` 和 `trainer.py` 保留用于旧实验。
新流程不修改 SISR、FST、GPT 或它们的数据接口。

## 数据与训练

输入必须是**已经分解的子结构**，不是待分解的完整分子。支持每行一个
SMILES 的文本、JSON 列表，以及 `{类型: {SMILES: 次数}}` 或更深层的
统计 JSON。API 也接受含 `raw_mol` / `smiles` 的分解记录。
保留 dummy 原子、连接键、形式电荷、芳香性与氢数，消除 atom-map 和
dummy isotope 的连接编号。真实原子同位素和立体化学不作为重构目标；
不将现有存疑的 symmetry 属性作为监督信号。

```powershell
python -m autoencoder.train_vq train --input data/stru_data.json --checkpoint autoencoder/fragment_vq.pth
python -m autoencoder.train_vq export --input data/stru_data.json --checkpoint autoencoder/fragment_vq.pth --output autoencoder/fragment_tokens.json
python -m unittest autoencoder.test_fragment_vq
```

命令从仓库根目录运行，环境需要 PyTorch 和 RDKit。默认 4 级 RVQ、
每级 256 个码、潜维度 128、最大 70 个原子。训练按规范化 SMILES 去重后
划分训练/验证集，以验证总损失保存最优 checkpoint，并报告各级码使用数。
无效、超长、超出支持范围的输入会报错，不静默截断或丢弃。

编码器采用带键类型偏置的图注意力及掩码池化，没有原子位置编码。
解码器使用可学习位置查询重构规范原子顺序的图；损失包括原子种类、
电荷、芳香性、氢数、有效原子对的键类型（包含无键）、长度及 VQ 损失。
单原子样本的键损失为零。RVQ 使用 straight-through 梯度与逐级残差量化，
不使用高斯采样或 KL 正则。

## 调用

```python
from autoencoder.fragment_vq import FragmentVQAutoencoder

model = FragmentVQAutoencoder.load('autoencoder/fragment_vq.pth')
rows = model.encode_fragments(['[1*]CC(=O)[O-]'])
# rows[0]: {'smiles': ..., 'codes': [整数, ...],
#           'tokens': ['<FZ1_...>', '<FZ2_...>', ...]}
```

导出临时切换 eval 模式并恢复调用前状态。checkpoint 包含模型配置、
特征版本和码本，必须和导出的 token 一起版本化；重新训练后的编号
不具有跨 checkpoint 的语义一致性。旧 VAE 权重不兼容新架构。

这些 token 是学习的子结构表示，不是唯一 ID 或无损结构编码。
`decode(quantizer.lookup(codes))` 可取得重构 logits；当前不提供保证化学有效的
图转 SMILES，也不把随机 RVQ 码组合视为可靠分子生成器。后续在 SISR
中插入前缀属于单独集成步骤。本目录的改动只负责训练与导出。
