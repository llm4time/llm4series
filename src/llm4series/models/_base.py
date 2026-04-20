from typing import Self
from abc import ABC, abstractmethod
from dataclasses import dataclass
import re


@dataclass(kw_only=True)
class ModelResponse:
  raw: str
  predicted: str
  input_tokens: int
  output_tokens: int
  time: float


class Model(ABC):

  @abstractmethod
  def predict(self: Self, prompt: str, data: str, temperature: float | None, **kwargs) -> ModelResponse:
    """
    Sends a request to the model and returns the response.

    Args:
        prompt (str): The prompt to send to the model.
        data (str): The data to send to the model.
        temperature (float | None): Degree of randomness in the response.
        **kwargs: Additional arguments passed to `client.chat.completions.create`.

    Returns:
        ModelResponse: Model response with detailed information.
    """
    ...

  def _output(self, response: str) -> str | None:
    matches = re.findall(r"<out>(.*?)</out>", response, re.DOTALL)
    return matches[-1].strip() if matches else None
