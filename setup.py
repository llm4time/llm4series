import os
from setuptools import setup, find_packages


def get_version():
  with open(os.path.join(os.path.dirname(__file__), "src/llm4series/_version.py")) as f:
    exec(f.read(), globals())
  return globals()["__version__"]

with open("README.md", "r") as arq:
  readme = arq.read()

version = get_version()

setup(name="llm4series",
      version=version,
      license="MIT License",
      author="Wesley Barbosa",
      author_email="wesley.barbosa.developer@gmail.com",
      url="https://github.com/llm4time/llm4series",
      long_description=readme,
      long_description_content_type="text/markdown",
      keywords=["Time Series Forecasting", "Large Language Model", "Prompt Engineering"],
      description="A library for time series forecasting with language models (LLMs)",
      python_requires=">=3.12, <3.15",
      packages=find_packages(where="src"),
      package_dir={"": "src"},
      install_requires=[
        "colorlog>=6.0.0,<7.0.0",
        "litellm>=1.83.4,<2.0.0",
        "numpy>=1.23.0,<3.0.0",
        "pandas>=2.0.0,<3.0.0",
        "scikit-learn>=1.3.0,<2.0.0",
        "openpyxl>=3.1.0,<4.0.0",
        "permetrics>=1.5.0,<2.0.0",
        "plotly>=5.15.0,<7.0.0",
        "kaleido>=1.3.0",
        "scipy>=1.10.0,<2.0.0",
        "statsmodels>=0.14.0,<1.0.0",
        "nbformat>=4.2.0",
        "python-toon==0.1.3",
        "gepa==0.1.1"
      ])
