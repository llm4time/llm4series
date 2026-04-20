from openai import AzureOpenAI as Client
from ._base import Model, ModelResponse
from ._rules import RULES
from typing import override
import time


class Azure(Model):

  def __init__(self, model: str, api_key: str, azure_endpoint: str, api_version: str) -> None:
    self.model = model
    self.client = Client(api_key=api_key, azure_endpoint=azure_endpoint, api_version=api_version)

  @override
  def predict(self, prompt: str, data: str, temperature: float = 0.7, **kwargs):
    start_time = time.time()
    response = self.client.chat.completions.create(
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
