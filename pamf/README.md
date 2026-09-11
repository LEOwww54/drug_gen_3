# PAMF：物理信息驱动的分子分解

独立实现聊天方案的 **PAMF-v1**，不修改现有 `decompose/`、`decompose_1/`、
FST、GPT 或 RVQ 的调用路径。输入完整 SMILES，输出携带成对 `[k*]` 的片段
SMILES、连接表、候选键特征和优化结果。第一版不训练神经网络。

## 安装与运行

在项目根目录运行，Python >= 3.10：

```powershell
python -m pip install -r pamf/requirements.txt
python -m pamf --smiles "CCCOCCNCCC" --output pamf_result.json
```

默认 `xtb` 模式直接调用 PATH 中的 **GFN2-xTB 命令行程序**：

```powershell
python -m pamf --smiles "CCCOCCNCCC" --conformers 10
```

XYZ 参数使用绝对路径，Windows 路径分隔符转换为 `/`，通过参数列表传入，
无需手动添加引号或转义反斜杠。读取结果后删除该次运行的整个临时目录，
包括 XYZ、结果及 xTB 中间文件；计算失败、超时或解析错误也会触发清理。
不再提供 `--xtb` / `xtb_executable` 自定义程序路径选项，请通过 PATH 配置命令。

`pip install xtb` 的 Python 接口不等于本模块调用的 CLI。Windows 可使用兼容的
原生 xTB 构建；若通过 WSL 安装，则应在 WSL 内用 Linux Python、RDKit 和 xTB
运行整个模块，不能把 Linux 可执行文件直接交给 Windows Python。
安装入口：[xTB 官方文档](https://xtb-docs.readthedocs.io/en/latest/setup.html)。

无需量化软件的显式基线模式：

```powershell
python -m pamf --smiles "CCCCCC" --mode rules
```

预期 `fragments` 为 `["[1*]CCC", "[1*]CCC"]`。`rules` 不计算 3D/WBO，
输出会明确标记，不能据此声称已完成量子物理分解。
xTB 缺失、超时、不收敛或输出格式不正确时会报错，不静默切换到规则模式。

## Python 接口

只需要输入 SMILES 并取得片段列表时，使用 `fragment_smiles`：

```python
from pamf import fragment_smiles, PAMFConfig

# 默认物理模式，需要 xTB
fragments = fragment_smiles("CCCOCCNCCC")

# 显式规则模式，只需要 RDKit
fragments = fragment_smiles("CCCCCC", PAMFConfig(mode="rules"))
print(fragments)  # ['[1*]CCC', '[1*]CCC']
```

返回值是 `list[str]`；每个连接编号恰好出现两次。相同片段可能重复出现，
不要去重。无法切割时返回原分子组分，不人为添加编号。输入无效或计算失败
会抛出异常。可选 `config` 和 `reference` 与完整接口一致。

需要连接表、评分及计算元数据时，使用完整接口：

```python
from pamf import PAMFConfig, decompose_smiles, reassemble

result = decompose_smiles("CCCOCCNCCC", PAMFConfig(mode="xtb"))
fragments = result["fragments"]
assert reassemble(fragments) == result["parent_smiles"]

# 只需字符串片段时取 result['fragments']，不要丢弃原始完整结果用于审计。
baseline = decompose_smiles("CCCCCC", PAMFConfig(mode="rules"))
```

每条切割键分配一个从 1 开始的 isotope label，在两个片段中各出现一次。
dummy 是切口处的终端原子。例如 `[1*]CC[2*]` 的两个 dummy 分别位于片段两端。
编号只在当前分子内有效，不是跨分子的片段 ID。

## 实际流程与边界

1. RDKit 解析、sanitize、环感知、立体化学识别。保留输入的形式电荷、质子化
   状态、同位素与 atom-map；不猜 pH，不做去盐、中和或互变异构体规范化。
   RDKit 默认 SMILES 解析会合并普通显式氢，原子索引以解析后的分子为准。
   输入已有 dummy 会报错，防止标签碰撞。
2. 硬保护环内键、非单键、共轭键、酰胺/脲/氨基甲酸酯、酯/羧酸盐、磺酰胺、
   脒/胍、硝基、共轭烯酮等。保守保护芳香-杂原子共振连接和已指定立体中心、
   E/Z 相关位置。规则是可审计的化学启发式，不保证识别所有离域体系。
3. 剩余重原子单键作为候选；BRICS、环边界和保护功能团边界提供额外奖励。
   本版使用独立 SMARTS/BRICS，不导入旧项目的 `constant.py` 或其 Torch 依赖，
   也没有把所有 MACCS 匹配直接当作不可拆单元。
   可通过 `extra_protected_smarts=(...)` 扩充保护规则。
4. `xtb` 模式生成默认 10 个 ETKDGv3 构象，MMFF94 优化；参数缺失时尝试 UFF，
   在元数据中记录实际方法。仅保留收敛且位于最低能量以上 10 kcal/mol 内的构象。
   种子固定、单线程构象生成，以支持同环境重复运行。
5. 按重原子邻居索引选择候选键两侧二面角；计算圆方差
   `1 - abs(mean(exp(i * phi)))`。少于两个构象或缺少二面角时记为 `null`，
   不伪装成已测得的刚性。此值是有限采样的启发式，不是热力学收敛证明。
6. 对最低力场能构象做一次 `GFN2-xTB --sp --wbo`。显式加氢后原原子索引保持
   不变，XYZ 中增加 1 对应 xTB 索引；传入总形式电荷和未配对电子数。
   默认未配对数取 RDKit radical electron 总数，可用 `--unpaired` 指定，
   这不等于自动推断耦合自由基的真实基态多重度。
   电子特征不是构象平均值，也没有额外做 xTB 几何优化。
7. 读取独立临时目录中的 `wbo`/`charges` 文件，验证索引、有限值与总电荷。
   保存切口 WBO、切键后两侧半径 2 跳重原子邻域的 cross-WBO 和电荷差；
   若标准输出含原子极化率表则记录，否则为 `null`。电荷差/极化率只作描述符。
   cross-WBO 使用 xTB 稀疏输出，未报告的非键原子对按零处理，因此它是截断近似，
   不是完整密度矩阵意义的精确电子耦合。候选化学键本身缺 WBO 时直接报错。
8. 候选评分为
   `w_rot*rotatable + w_boundary*boundary + w_flex*torsion_variability
   - w_wbo*z_wbo - w_cross*z_cross_wbo`。
   各项及最终得分均输出。权重是工程初值，需要 benchmark 调参；得分不是概率。
9. 对切割集合联合优化：切键得分之和，减去新增片段数、超长片段平方惩罚和
   片段尺寸变异系数平方惩罚。默认新片段至少 3 个重原子；已有小分子/离子可
   原样保留。最大 25 默认是软约束，以免为了尺寸切坏大环或不可拆单元；
   `--strict-max-size` 启用硬上限，不可行会报错。
   <=14 条候选键穷举，给出全局最优；更多候选使用宽度 128 的 beam search，
   明确标记 `optimality_proven=false`。它优化完整切割方案而非逐键阈值，但不能
   保证大分子的全局最优；严格尺寸模式下搜索失败也不等于数学上不存在解。
10. `FragmentOnBonds(dummyLabels=[(k,k), ...])` 生成连接点，通过 isotope `molzip`
    重组，并与原分子的 canonical isomeric SMILES 比较。任何不一致都会报错。
    验证针对 RDKit 支持的标准 SMILES 信息，不包含 CXSMILES 扩展注释。

`xtb` 模式对有可切键的多组分输入报错，避免把分离盐组分当成确定的三维复合物
计算电子耦合；请显式选择要研究的组分。`rules` 保留所有组分及总电荷。
没有候选键时直接返回原分子组分，不进行无用的物理计算。

## 同类键归一化与训练集统计

化学类别是两端元素和杂化类型的无向组合，如 `C(SP3)-O(SP3)`。
绝不使用统一 `WBO < 常数` 的切键阈值。未提供 reference 时，默认按**当前分子
同类候选键**计算均值/标准差；只有一条键或方差为零时，该电子项设为中性并给警告。
这允许先运行采集特征，但小分子上电子评分可能没有区分能力，不能替代数据集校准。

正式比较建议：先只在训练集采集原始物理特征，再拟合参考统计并冻结。

```powershell
python -m pamf --input train.smi --output train_features.jsonl
python -m pamf --fit-reference train_features.jsonl --output reference.json
python -m pamf --input test.smi --reference reference.json --output test_fragments.jsonl
```

统计格式为 `{chemical_class: {count, wbo: {mean, std}, cross_wbo: {mean, std}}}`。
推理遇到参考统计中没有的类别会报错，避免偷偷退回不可比较的量纲。
应使用相同 xTB 方法、几何配置、候选规则和邻域半径拟合与推理；本版不会替你判断
外部参考文件的数据来源是否可信。WBO/cross-WBO 较相关，权重应做消融验证。

批量输入是每行一个 SMILES 的 UTF-8 文本，输出 JSONL；遇错报告输入行号并停止，
不会跳过失败样本。当前批量结果先保存在内存，建议大型数据分批处理。

## 文件结构

| 文件 | 职责 |
|---|---|
| `chemistry.py` | 输入校验、保护规则、候选边界、切割与重组 |
| `geometry.py` | 构象集合、力场优化、圆方差 |
| `electronic.py` | xTB 子进程、输出解析、索引映射、cross-WBO |
| `scoring.py` | 参考统计和候选评分 |
| `optimizer.py` | 穷举/beam 切割集合优化 |
| `pipeline.py` | 配置和公共 API |
| `__main__.py` | 单分子/批量 CLI、统计拟合 |
| `test_pamf.py` | 化学回环、全局选择、物理接口、异常测试 |

输出 `connections` 的原子索引指向 **RDKit 解析 input_smiles 后的原子顺序**，
不是重新解析 `parent_smiles` 后的顺序。`electronics.parent_to_xyz_index` 记录
对应的 1-based XYZ 索引。输出另存 RDKit 版本、完整配置、原始特征、评分项、
搜索方法和警告，便于复现。

## 测试与后续软件

```powershell
python -m unittest pamf.test_pamf -v
# 已安装 xTB 时，可额外运行真实量化端到端测试
$env:PAMF_RUN_XTB = '1'
python -m unittest pamf.test_pamf.PhysicsTests.test_real_xtb -v
```

默认测试执行真实 RDKit 化学/构象计算，xTB 协议测试使用 mock；真实 xTB 测试
需要单独启用。这不能替代对实际 xTB 构建版本的集成验证。

CREST 构象搜索、ORCA/DFT 验证、Multiwfn/AIMAll QTAIM，以及 GNN surrogate
属于聊天方案的后续阶段，本版不依赖也不调用。与既有 FST tokenizer 的自动对接
需单独适配其记录格式；这里提供完整 SMILES 和连接表供该步骤使用。

接口依据：
[RDKit 切割与 molzip](https://www.rdkit.org/docs/source/rdkit.Chem.rdmolops.html)、
[xTB 属性输出](https://xtb-docs.readthedocs.io/en/latest/properties.html)、
[xTB CLI](https://xtb-docs.readthedocs.io/en/latest/commandline.html)。
