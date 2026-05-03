from openai import OpenAI as Client
from openai import AzureOpenAI as AzureClient
from ._base import Model, ModelResponse
from ..prompts._generator import PromptConfig
from ..data import UniTimeSeries, MultiTimeSeries
from typing import override
import time


class LLM(Model):

  def __init__(self, model: str, api_key: str = None, base_url: str = None,
               local: bool = False, azure: bool = False, **kwargs) -> None:
    self.model = model
    if local and azure:
      raise ValueError("Model cannot be both local and Azure.")
    if not local and not api_key:
      raise ValueError("API key must be provided for non-local models.")
    if azure:
      self.client = AzureClient(api_key=api_key, azure_endpoint=base_url, **kwargs)
    elif local:
      self.client = Client(api_key="local", base_url=base_url or "http://localhost:1234/v1", **kwargs)
    else:
      self.client = Client(api_key=api_key, base_url=base_url or "https://api.openai.com/v1", **kwargs)

  @override
  def predict(self, prompt: PromptConfig, **kwargs):
    if not isinstance(prompt, PromptConfig):
      raise ValueError(f"Expected PromptConfig, got {type(prompt).__name__}.")

    start_time = time.time()
    response = self.chat(
      messages=[
        {"role": "system", "content": prompt.system},
        {"role": "user", "content": prompt.text},
        {"role": "user", "content": prompt.data}
      ],
      response_format=prompt.response_format,
      **kwargs
    )
    end_time = time.time()
    return self._build_response(response, end_time - start_time)

  def chat(self, messages, response_format=None, **kwargs):
    if response_format:
      return self.client.beta.chat.completions.parse(model=self.model, messages=messages, response_format=response_format, **kwargs)
    else:
      return self.client.beta.chat.completions.create(model=self.model, messages=messages, **kwargs)

  def _build_response(self, response, execution_time):
    usage = getattr(response, "usage", None)
    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)
    choice = response.choices[0]
    message = getattr(choice, "message", None)
    parsed = getattr(message, "parsed", None) if message else None
    if parsed is None:
      raise ValueError("Response does not contain a parsable message.")
    data = parsed.model_dump()
    index = data.pop("date")
    if len(data) > 1:
      prediction = MultiTimeSeries(data, index=index)
      prediction.index.name = "date"
    else:
      column_name, values = next(iter(data.items()))
      prediction = UniTimeSeries(values, index=index, name=column_name)
      prediction.index.name = "date"
    return ModelResponse(prediction=prediction, input_tokens=input_tokens,
        output_tokens=output_tokens, time=execution_time)
