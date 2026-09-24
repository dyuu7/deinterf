<h1 align="center">
  <img src="https://raw.githubusercontent.com/dyuu7/deinterf/main/docs/assets/logo.svg" alt="deinterf" width="80%" />
</h1>

在航磁勘探、磁异常目标检测和磁导航等应用中，对外部磁场的观测不可避免地会受到航空器自身磁干扰的影响。机体的永久磁化和感应磁化、航空器机动过程中产生的涡流，以及机载电气系统中的电流都会形成附加磁场，并与环境磁场共同叠加到磁传感器的观测结果中。针对这类平台磁干扰，`deinterf` 提供了一套可扩展的干扰建模与补偿框架，用于对航空器产生的磁场分量进行估计、分离并从观测数据中扣除，从而减弱平台自身对外部磁场测量的影响。

`deinterf` 以 Tolles–Lawson（T–L）模型为基础，通过可组合的模型项描述不同干扰分量，并借助基于 [`dataioc`](https://github.com/dyuu7/dataioc) 的数据容器组织和解析模型变量之间的依赖关系。在此基础上，库进一步提供类似 scikit-learn 的 `fit`/`predict`/`transform` 接口，使干扰模型的构建、标定与应用能够在统一的工作流中完成。

[English](README.md) | **简体中文**

## 安装

```shell
pip install deinterf
```

运行下面的公开航磁数据示例，需要安装数据读取、绘图和 IGRF 相关可选依赖：

```shell
pip install "deinterf[examples]"
```

## 从校准飞行到测线补偿

假设已有同步采集的磁通门和标量磁力仪数据：先用校准航线估计航空器的干扰模型，再对相同传感器配置下的另一条测线进行补偿。

下面通过 [`dafmit-aeromag`](https://github.com/dyuu7/dafmit-aeromag) 读取公开飞行架次 `1002`，用校准航线 `1002.02` 拟合，对 Renfrew 重复测线 `158.00` 进行补偿，采样率均为 10 Hz。`flux_b_x/y/z` 是磁通门三个分量，`mag_3_uc` 是未补偿的标量总场，四列单位均为 nT。

```python
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate


dataset = Dataset()
columns = ["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"]
calibration = dataset.read(
    Selection(1002, lines="1002.02"), columns=columns, split="train"
)
survey = dataset.read(Selection(1002, lines="158.00"), columns=columns, split="train")


def as_inputs(frame):
    X = DataIoC().with_data(
        MagVector(frame["flux_b_x"], frame["flux_b_y"], frame["flux_b_z"])
    )
    return X, Tmi(frame["mag_3_uc"])


calibration_X, calibration_y = as_inputs(calibration)
survey_X, survey_y = as_inputs(survey)

model = TollesLawson(terms=Terms.Terms_16, sampling_rate=10)
model.fit(calibration_X, calibration_y)
tmi_clean = model.transform(survey_X, survey_y)
print(f"improvement_rate={improve_rate(survey_y, tmi_clean, sampling_rate=10):.2f}")
```

`tmi_clean` 是补偿后的测线总场，`model.predict(survey_X)` 则返回估计的干扰。补偿保留输入的均值和单位。默认模型对特征进行标准化，使用岭回归，并在拟合时应用 0.1–0.6 Hz 带通滤波。设置 `filter=None` 可关闭滤波；通过 `estimator` 传入具有 `fit` 和 `predict` 方法的对象，可更换回归器。

读取器首次使用时可能下载并缓存飞行数据。`split="train"` 选择的是数据集已公开的分区，只有传给 `model.fit` 的样本参与补偿器拟合。使用自己的数据时，替换两次读取并调整 `as_inputs` 即可：磁矢量形状为 `(n_samples, 3)`，总场形状为 `(n_samples,)`。样本需对齐、等间隔采样，校准与测线数据需保持单位、坐标轴和采样率一致。

完整的[经典补偿脚本](examples/classic_tmi.py)还会绘制补偿前后的信号。从仓库目录运行：

```shell
uv run --extra examples python examples/classic_tmi.py
```

## 概念快览

| 概念 | 在补偿中的作用 | 本库中的对应 |
| --- | --- | --- |
| 标量总场（TMI） | 待补偿的观测信号，同时含有目标磁场和航空器干扰 | `Tmi`，作为 `y` 传入 |
| 磁矢量 | 三个磁场分量，用于推导建模所需的方向和模长 | `MagVector`，注册在 `DataIoC` 中 |
| 方向余弦 | 描述磁场在传感器坐标系中方向的无量纲 x/y/z 分量 | `DirectionalCosine`，默认由 `MagVector` 推导 |
| 校准机动 | 通过姿态变化帮助估计干扰系数；拟合目标 `y` 是实测总场，并非已知的干净信号 | `fit(calibration_X, calibration_y)` |
| T–L 模型项 | 永磁、感应和涡流贡献，对待拟合系数是线性的 | `Terms.Terms_16` 组合 3 + 5 + 8 个特征 |
| 补偿 | 将拟合得到的干扰估计从另一条测线的观测中扣除 | `transform(survey_X, survey_y)` |

永磁项描述航空器的固定磁化，感应项描述外磁场引起的磁化，涡流项描述航空器机动时磁场变化产生的涡流影响。

`noise_level(y)` 计算 0.1–0.6 Hz 带通滤波后的标准差；`improve_rate(before, after)` 是补偿前后该值的比值，大于 1 表示这一频带内的波动降低。两者默认采样率为 10 Hz，不同时需传入实际的 `sampling_rate`。这些 FOM（figure of merit，性能评价）指标适合用于校准机动数据；普通测线的同一频带也可能包含地质信号。请在独立数据上评估，并结合航测项目的地球物理检查解读补偿比。

## 将业务映射到模型

先区分直接测得的数据、从数据推导的物理量，以及各干扰机制对应的特征。补偿器请求模型项时，`DataIoC` 会按依赖关系取得所需数据。

| 业务要素 | 库中的表达 | 示例 |
| --- | --- | --- |
| 直接观测量 | 注册到 `DataIoC` 的数据类型 | `MagVector`、自定义的 `Current` |
| 派生物理量 | 实现 `__build__` 的数据类型 | 由 `MagVector` 推导 `DirectionalCosine` |
| 一类干扰机制 | 返回 `(n_samples, n_features)` 特征矩阵的 `ComposableTerm` | 永磁项或电缆电流项 |
| 完整特征模型 | 用 `\|` 组合模型项 | `Terms.Terms_16 \| CableCurrent()` |
| 拟合与补偿 | 配置模型项和回归器的 `TollesLawson` | 复用 `fit`、`transform` 流程 |

### 选择传感器数据源

有多个磁通门时，可以用不同索引登记数据，并选择构建特征所用的传感器。例如，将前例中的磁通门登记为 1 号数据源：

```python
sensor_X = DataIoC().with_data(MagVector[1](*calibration_X[MagVector].T))
sensor_model = TollesLawson(terms=Terms.Terms_16[1], sampling_rate=10)
sensor_model.fit(sensor_X, calibration_y)
```

补偿测线时，用相同索引登记对应传感器的数据。这里的索引用于选择数据源，并非特征列号。模型项也可以选择 `Terms.Permanent`、`Terms.Induced_5`、`Terms.Eddy_8`，用 `|` 自由组合，或使用 3 + 6 + 9 形式的 `Terms.Terms_18`。

### 加入实测干扰来源

假设记录了机载电流，并将其产生的磁场建模为：幅值与电流成正比，方向在传感器坐标系中固定。电流与三个方向余弦相乘后得到三列特征，其系数可以和 T–L 系数一起拟合。继续使用上面的校准数据：

```python
from dataioc import DataNDArray

from deinterf.foundation import ComposableTerm
from deinterf.foundation.sensors import DirectionalCosine


class Current(DataNDArray):
    pass


class CableCurrent(ComposableTerm):
    def __build__(self, container: DataIoC):
        current = container[Current]
        direction = container[DirectionalCosine]
        return current[:, None] * direction


current_data = dataset.read(
    Selection(1002, lines="1002.02"), columns=["cur_com_1"], split="train"
)
current_X = DataIoC().with_data(
    calibration_X[MagVector], Current(current_data["cur_com_1"])
)
current_model = TollesLawson(terms=Terms.Terms_16 | CableCurrent(), sampling_rate=10)
current_model.fit(current_X, calibration_y)
```

`Current` 表达观测量，`CableCurrent` 表达它如何进入模型；`DirectionalCosine` 会由已注册的 `MagVector` 自动推导。用该模型补偿测线时，也需提供同步的电流数据。[电缆电流示例](examples/extended_cable_current_tmi.py)进一步展示了多路电流和导数项，[载荷振动示例](examples/extended_load_vibration_tmi.py)展示了更高阶的特征。

### 用惯导推导磁场方向

如果希望由惯性导航系统（INS）提供方向余弦，可以用位置、时间和地磁模型得到磁场，再结合姿态旋转到传感器坐标系。将这段计算实现为 `InsDirectionalCosine` 数据提供者，通过 `container.add_provider(DirectionalCosine, InsDirectionalCosine)` 注册，现有 T–L 项就会使用替换后的方向来源。

[惯导示例](examples/replace_direction_cosine_source_tmi.py)使用 IGRF 和姿态数据实现了这条计算链。该示例替换方向余弦，磁场模长仍由 `MagVector` 提供；使用时需根据仪器匹配旋转顺序、角度单位和坐标系约定。

请在求值前注册输入和数据提供者。容器会缓存计算结果，更换输入或提供者时应新建 `DataIoC`。参数和返回值细节可查阅 [`TollesLawson`](deinterf/compensator/tmi/linear/tolles_lawson.py)、[传感器](deinterf/foundation/sensors.py)和[评价指标](deinterf/metrics/fom.py)的 docstring。

## 参与贡献

欢迎贡献干扰模型、传感器数据源、示例、测试和修复。新增模型项时，请说明物理假设、输入及单位，并提供具有已知预期行为的示例或测试。修改用法时，请同步更新中英文 README。

使用 Ruff 格式化和静态检查 Python 代码，使用 ty 检查类型。从仓库目录运行：

```shell
uv sync --extra examples --group lint
uv run --no-sync ruff check .
uv run --no-sync ty check
uv run --no-sync pytest
```

提交前，用 `uv run --no-sync ruff format <files>` 格式化本次修改的 Python 文件，将格式调整限定在这些文件内。测试使用本地合成飞行数据，不下载公开数据；未安装 `examples` 可选依赖时，会跳过相应示例测试。

## 贡献者

感谢所有贡献代码、示例、问题报告和反馈的贡献者。

<a href="https://github.com/dyuu7/deinterf/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dyuu7/deinterf" alt="deinterf contributors" />
</a>

## 许可证

[MIT License](LICENSE)。
