from openai import OpenAI as Client
from ._base import Model, ModelResponse
from ._rules import RULES
from typing import override
import time


class LMStudio(Model):

  def __init__(self, model: str, base_url: str | None = None) -> None:
    self.model = model
    self.base_url = base_url or "http://localhost:1234/v1"

  @override
  def predict(self, prompt: str, data: str, temperature: float = 0.7, **kwargs):
    client = Client(api_key="not-needed", base_url=self.base_url)
    start_time = time.time()
    response = client.chat.completions.create(
      model=self.model,
      messages=[
        {"role": "system", "content": RULES},
        {"role": "user", "content": prompt},
        {"role": "user", "content": data}
      ],
      temperature=temperature,
      **kwargs,
    )
    end_time = time.time()
    return self._build_response(response, end_time - start_time)

  def _build_response(self, response, execution_time):
    raw = response.choices[0].message.content or ""
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None) if usage else None
    output_tokens = getattr(usage, "completion_tokens", None) if usage else None
    predicted = self._output(raw) or ""
    return ModelResponse(raw=raw, predicted=predicted, input_tokens=input_tokens,
                         output_tokens=output_tokens, time=execution_time)
