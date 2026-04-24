<div align="center">
  <img src="https://raw.githubusercontent.com/llm4time/llm4series/main/docs/assets/llm4series.svg" width="150" />

# llm4series
**A library for time series forecasting using Large Language Models (LLMs)**

[![PyPI version](https://img.shields.io/pypi/v/llm4series.svg)](https://pypi.org/project/llm4series/)
![Python versions](https://img.shields.io/badge/python-3.12+-blue)
[![License](https://img.shields.io/github/license/llm4time/llm4series.svg)](LICENSE)

[![YouTube Video](https://img.shields.io/badge/YouTube-Tutorial-red?logo=youtube)](https://www.youtube.com/)
</div>

<p align="center">
  <a href="#-get-started">Get Started</a> •
  <a href="#-features">Features</a> •
  <a href="#-full-pipeline-example">Full Pipeline Example</a> •
  <a href="#-citation">Citation</a> •
  <a href="#-collaborators">Collaborators</a> •
  <a href="#-license">License</a>
</p>

---

## 🧩 Get Started

llm4series is a Python library for time series forecasting using Large Language Models (LLMs).
It provides a modular architecture that includes:
- Data preprocessing and handling
- Prompt generation
- Forecasting with LLMs (OpenAI, Azure, LMStudio)
- Metric evaluation
- Interactive visualization

### Installation
```bash
pip install llm4series
```

Or clone the repository and install dependencies:
```bash
git clone https://github.com/llm4time/llm4series.git
cd llm4series
pip install -r requirements-dev.txt
```

### Requirements
- Python >= 3.12
- numpy >= 1.23.0, <2.2.5
- pandas >= 2.0.0, <2.2.3
- openai >= 1.40.0, <1.90.0
- openpyxl >= 3.1.0, <3.2.0
- permetrics >= 1.5.0, <2.0.0
- plotly >= 5.15.0, <6.1.0
- scikit-learn >= 1.3.0, <1.7.1
- scipy >= 1.10.0, <1.15.3
- statsmodels >= 0.14.0, <0.14.5
- nbformat >= 4.2.0
- python-toon == 0.1.3
- gepa == 0.1.1

---

## 🚀 Features

- Read and preprocess time series data from CSV, Excel, JSON, and Parquet files
- Support for univariate and multivariate time series
- Built-in models for OpenAI, Azure, and LMStudio APIs
- Prompt engineering utilities (zero-shot, few-shot, chain-of-thought)
- Multiple output formats: CSV, JSON, Markdown, TSV, and more
- Visualization tools for time series and forecasts
- Evaluation metrics: SMAPE, MAE, RMSE, and more
- Extensible and modular design

---

## 🔥 Full Pipeline Example

See the [full pipeline notebook](examples/full%20pipeline.ipynb) for a complete workflow.

```python
import llm4series as l4s

# Load and preprocess data
ts = l4s.read_file('llm4series/examples/data/busline.csv', index_col='date')
ts = ts.impute_interpolate('linear') if ts.isna().sum() > 0 else ts
ts = ts.agg_duplicates(method="sum") if ts.index.duplicated().sum() > 0 else ts

# Split data
train, test = ts.split(start="2018-04-06 14:00:00", end="2018-06-16 13:00:00", periods=24)

# Prompt construction
prompt = l4s.prompt(
    ts=train,
    forecast_horizon=24,
    tsformat="plain",
    tstype="textual",
    sampling="uniform",
    type="cot"
)

# Model initialization and prediction
import os
model = l4s.OpenAI(model="gpt-5", api_key=os.getenv("OPENAI_API_KEY"), base_url="https://api.openai.com/v1")
data = train.to_str(format="plain", type="textual")
response = model.predict(prompt, data, temperature=1)

# Parse and plot results
predicted = l4s.from_str(response.predicted, format="plain")
l4s.lineplot(test, predicted, groups=["Original series", "Forecast"])

# Metrics
metrics = test.metrics(predicted)["value"]
print(metrics["smape"], metrics["mae"], metrics["rmse"])
```

---

## 📚 Citation

This library was first presented at the **1st Time Series Age with LLMs** workshop, ICLR 2026.

You can read and discuss our paper on [OpenReview](https://openreview.net/forum?id=6fbcYFRoUL).

If you use llm4series in your research, please cite:

```latex
@inproceedings{silva2026llm4series,
  title={LLM4series: Structured prompting for time series forecasting with LLMs},
  author={Silva, Wesley Barbosa and Scarcela, Maria Fernanda Aquino Freitas and Viana, Luiz Zairo Bastos and Caminha, Carlos and do Vale Madeiro, Jo{\~a}o Paulo and da Silva, Jos{\'e} Wellington Franco},
  booktitle={1st ICLR Workshop on Time Series in the Age of Large Models},
  year={2026}
}
```

---

## 👥 Team
<div align="center">
<table>
  <tr>
    <td align="center" nowrap>
      <a href="https://github.com/zairobastos"><img src="https://github.com/zairobastos.png" style="width: 80px; height: 80px;" alt="Zairo Bastos"/></a>
      <br />
      <sub><b>Zairo Bastos</b></sub>
      <br />
      <sub><i>Master’s student - UFC</i></sub>
      <br />
      <a href="mailto:zairobastos@gmail.com" title="Email">📧</a>
      <a href="https://www.linkedin.com/in/zairobastos/" title="LinkedIn">🔗</a>
    </td>
    <td align="center" nowrap>
      <a href="https://github.com/wesleey"><img src="https://github.com/wesleey.png" style="width: 80px; height: 80px;" alt="Wesley Barbosa"/></a>
      <br />
      <sub><b>Wesley Barbosa</b></sub>
      <br />
      <sub><i>Undergraduate student - UFC</i></sub>
      <br />
      <a href="mailto:wesley.barbosa.developer@gmail.com" title="Email">📧</a>
      <a href="https://www.linkedin.com/in/wesleybarbosasilva/" title="LinkedIn">🔗</a>
    </td>
    <td align="center" nowrap>
      <a href="https://github.com/fernandascarcela"><img src="https://github.com/fernandascarcela.png" style="width: 80px; height: 80px;" alt="Fernanda Scarcela"/></a>
      <br />
      <sub><b>Fernanda Scarcela</b></sub>
      <br />
      <sub><i>Undergraduate student - UFC</i></sub>
      <br />
      <a href="mailto:fernandascla@alu.ufc.br" title="Email">📧</a>
      <a href="https://www.linkedin.com/in/fernanda-scarcela-a95543220/" title="LinkedIn">🔗</a>
    </td>
    <td align="center" nowrap>
      <a href="https://lattes.cnpq.br/4380023778677961"><img src="https://raw.githubusercontent.com/zairobastos/LLM4Time/main/docs/assets/carlos.png" style="width: 80px; height: 80px;" alt="Carlos Caminha"/></a>
      <br />
      <sub><b>Carlos Caminha</b></sub>
      <br />
      <sub><i>Academic advisor - UFC</i></sub>
      <br />
      <a href="mailto:caminha@ufc.br" title="Email">📧</a>
      <a href="https://lattes.cnpq.br/4380023778677961" title="Lattes">🔗</a>
    </td>
    <td align="center" nowrap>
      <a href="https://lattes.cnpq.br/5168415467086883"><img src="https://raw.githubusercontent.com/zairobastos/LLM4Time/main/docs/assets/wellington.png" style="width: 80px; height: 80px;" alt="José Wellington Franco"/></a>
      <br />
      <sub><b>José Wellington Franco</b></sub>
      <br />
      <sub><i>Academic advisor - UFC</i></sub>
      <br />
      <a href="mailto:wellington@crateus.ufc.br" title="Email">📧</a>
      <a href="https://lattes.cnpq.br/5168415467086883" title="Lattes">🔗</a>
    </td>
  </tr>
</table>
</div>

## 🎥 Video Tutorial

Watch the full pipeline tutorial on [YouTube](https://www.youtube.com/watch?v=uIYT28ncVmA):

[![YouTube Video](https://img.shields.io/badge/YouTube-Tutorial-red?logo=youtube)](https://www.youtube.com/watch?v=uIYT28ncVmA)

---

## 📄 License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
