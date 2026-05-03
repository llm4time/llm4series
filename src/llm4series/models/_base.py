from typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass
from ..data import TimeSeries


@dataclass(kw_only=True)
class ModelResponse:
  prediction: TimeSeries
  input_tokens: int
  output_tokens: int
  time: float


class Model(ABC):

  @abstractmethod
  def predict(self: Self, prompt: str, data: str, temperature: float | None, **kwargs) -> ModelResponse:
    ...
