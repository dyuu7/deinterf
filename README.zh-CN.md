<h1 align="center">
  <img src="https://raw.githubusercontent.com/dyuu7/deinterf/main/docs/assets/logo.svg" alt="deinterf" width="80%" />
</h1>

`deinterf` 基于 [Tolles–Lawson（T–L）模型](https://en.wikipedia.org/wiki/Tolles%E2%80%93Lawson_equation)，用于根据磁场测量估计航空器磁干扰，并从标量总场数据中扣除干扰。它提供类似 scikit-learn 的 `fit`/`predict`/`transform` 接口，也支持组合模型项以加入其他干扰来源。

当前版本聚焦于经典线性 T–L 项的标量 TMI 补偿。补偿指标降低并不等同于地质场估计更准确；正式使用前，请在独立飞行架次或航线上验证模型。

[English](README.md) | **简体中文**

## 安装

`deinterf` 支持 CPython 3.10–3.14。

```shell
pip install deinterf
```

核心包接受 NumPy 兼容的数组。运行下面的真实航磁数据示例，需要安装可选依赖：

```shell
pip install "deinterf[examples]"
```

## 快速开始

`MagVector` 的形状为 `(n_samples, 3)`，`Tmi` 的形状为 `(n_samples,)`。样本必须对齐、等间隔采样，并使用一致的单位和坐标约定。请使用 calibration 数据拟合，再在独立的飞行架次或航线上进行补偿：

```python
import numpy as np
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate


def flight(phase):
    time = np.arange(400) / 10  # 10 Hz
    field = np.column_stack(
        (
            20000 + 800 * np.sin(2 * np.pi * 0.17 * time + phase),
            3000 + 700 * np.cos(2 * np.pi * 0.23 * time + phase),
            45000 + 900 * np.sin(2 * np.pi * 0.31 * time + phase),
        )
    )
    direction = field / np.linalg.norm(field, axis=1)[:, None]
    interference = direction @ np.array([1200.0, -700.0, 300.0])
    return (
        DataIoC().with_data(MagVector(*field.T)),
        Tmi(50000 + interference),
    )


calibration_X, calibration_y = flight(0.0)
survey_X, survey_y = flight(0.7)

model = TollesLawson(terms=Terms.Terms_16, sampling_rate=10)
model.fit(calibration_X, calibration_y)
tmi_clean = model.transform(survey_X, survey_y)

print(f"improvement_rate={improve_rate(survey_y, tmi_clean):.2f}")
```

`TollesLawson` 默认对特征进行标准化，使用 10 折 `RidgeCV`，并在设定的采样率下应用四阶 0.1–0.6 Hz 带通滤波。若不需要这项预处理，可以设置 `filter=None`。

`predict(X)` 返回去除常量分量后的干扰估计。`transform(X, y)` 返回与 `y` 使用相同单位的补偿后 TMI，并保留信号均值。

## 真实航磁数据

下面的示例通过 [`dafmit-aeromag`](https://github.com/dyuu7/dafmit-aeromag) 读取飞行架次 `1002` 的校准航线 `1002.02`，采样率为 10 Hz。首次使用时，数据读取器可能会下载并校验数据；网络不可用时，可以使用已有缓存并设置 `offline=True`。

```python
from dafmit_aeromag import Dataset, Selection
from dataioc import DataIoC

from deinterf.compensator.tmi.linear import Terms, TollesLawson
from deinterf.foundation.sensors import MagVector, Tmi
from deinterf.metrics.fom import improve_rate, noise_level


flight_data = Dataset().read(
    Selection(1002, lines="1002.02"),
    columns=["flux_b_x", "flux_b_y", "flux_b_z", "mag_3_uc"],
    split="train",
)

fom_data = DataIoC().with_data(
    MagVector(
        bx=flight_data["flux_b_x"],
        by=flight_data["flux_b_y"],
        bz=flight_data["flux_b_z"],
    )
)
tmi_with_interf = Tmi(flight_data["mag_3_uc"])

# 这是校准航线示例。评估泛化能力时，请使用独立航线或飞行架次。
tmi_clean = TollesLawson(terms=Terms.Terms_16).fit_transform(
    fom_data, tmi_with_interf
)
print(f"noise_level={noise_level(tmi_clean):.4f}")
print(f"improvement_rate={improve_rate(tmi_with_interf, tmi_clean):.2f}")
```

## 模型项与扩展

| 模型项 | 特征数 | 典型用途 |
| --- | ---: | --- |
| `Terms.Permanent` | 3 | 永磁项 |
| `Terms.Induced_5` | 5 | 感应磁化项 |
| `Terms.Eddy_8` | 8 | 涡流项 |
| `Terms.Terms_16` | 16 | 标准 T–L 模型，默认值 |
| `Terms.Terms_18` | 18 | 扩展的感应项和涡流项 |

可选索引用于选择传感器数据源。例如，`Terms.Terms_16[1]` 使用 `MagVector[1]`；不带索引时使用默认的 `MagVector` 数据源。

自定义模型项需要实现 `ComposableTerm.__build__`，并可使用 `|` 组合，例如 `Terms.Terms_16 | MyInterferenceTerm()`。也可以替换 `DirectionalCosine` 等派生数据源。示例见：

- [`examples/extended_load_vibration_tmi.py`](examples/extended_load_vibration_tmi.py)
- [`examples/extended_cable_current_tmi.py`](examples/extended_cable_current_tmi.py)
- [`examples/replace_direction_cosine_source_tmi.py`](examples/replace_direction_cosine_source_tmi.py)

请在读取数据前注册所有输入和 provider；容器会缓存已经计算的值。

## 评价指标

`noise_level(y)` 是信号经过 0.1–0.6 Hz 带通滤波后的标准差。`improve_rate(y_uncomp, y_comped)` 定义为：

`noise_level(y_uncomp) / noise_level(y_comped)`

两者默认采样率均为 10 Hz；采样率不同时请传入实际值。

请在未参与拟合的数据上进行评估。示例指标适合比较处理方案，但不能替代地球物理验证或具体航测项目的质量控制流程。

## 迁移与开发

当前版本使用 [`dataioc`](https://github.com/dyuu7/dataioc) 管理数据容器，并使用 `dafmit-aeromag` 作为可选的航磁数据读取器。旧接口 `deinterf.utils.data_ioc` 和 `sgl2020.Sgl2020` 已不再支持。

从代码仓库开发：

```shell
uv sync --extra examples
uv run --extra examples pytest
```

## 贡献者

感谢所有贡献代码、示例、问题报告和反馈的贡献者。

<a href="https://github.com/dyuu7/deinterf/graphs/contributors">
  <img src="https://contrib.rocks/image?repo=dyuu7/deinterf" alt="deinterf contributors" />
</a>

目前独立维护的 [`dataioc`](https://github.com/dyuu7/dataioc) 最初主要由 [yanang007](https://github.com/yanang007) 实现，概念由 [dyuu7](https://github.com/dyuu7) 提出。

## 许可证

本项目采用 [MIT License](LICENSE) 授权。
