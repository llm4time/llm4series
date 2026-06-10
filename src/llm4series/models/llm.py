from litellm import completion
from ._base import Model, ModelResponse
from ..prompts._generator import PromptConfig
from ..data import UniTimeSeries, MultiTimeSeries
from .._internal import logger
from typing import override
import time
import json


class LLM(Model):

  def __init__(self, model: str, api_key: str = None, base_url: str = None, **kwargs) -> None:
    self.model = model
    self.api_key = api_key
    self.base_url = base_url
    self.kwargs = kwargs

  @override
  def predict(self, prompt: PromptConfig, **kwargs):
    if not isinstance(prompt, PromptConfig):
      logger.error(f"Expected PromptConfig, got {type(prompt).__name__}.")
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

    return self._build_response(
      response=response,
      response_model=prompt.response_format,
      execution_time=end_time - start_time
    )

  def chat(self, messages, response_format=None, **kwargs):
    params = {
        "model": self.model,
        "messages": messages,
        **self.kwargs,
        **kwargs
    }

    if self.api_key:
        params["api_key"] = self.api_key

    if self.base_url:
        params["api_base"] = self.base_url

    if response_format:
        params["response_format"] = {
            "type": "json_schema",
            "json_schema": {
                "name": response_format.__name__,
                "schema": response_format.model_json_schema()
            }
        }
    logger.info(f"[{params['model']}] Sending request...")
    start_time = time.time()
    response = completion(**params)
    end_time = time.time() - start_time
    logger.info(f"[{params['model']}] Response received in {end_time:.2f}s.")
    return response

  def _build_response(self, response, response_model, execution_time):
    usage = getattr(response, "usage", None)

    if usage is None:
      logger.warning("LLM response does not contain usage information.")

    input_tokens = getattr(usage, "prompt_tokens", None)
    output_tokens = getattr(usage, "completion_tokens", None)

    if input_tokens is None:
      logger.warning("LLM response does not contain prompt_tokens.")

    if output_tokens is None:
      logger.warning("LLM response does not contain completion_tokens.")

    content = response.choices[0].message.content

    if content is None:
      logger.warning("LLM response content is None.")
      raise ValueError("LLM response content is None.")

    try:
      parsed = response_model.model_validate_json(content)
    except Exception as e:
      parsed = response_model.model_validate(json.loads(content))

    data = parsed.model_dump()

    index = data.pop("date", None)

    if index is None:
      logger.warning("Response model does not contain required 'date' field.")
      raise ValueError("Response model does not contain required 'date' field.")

    if not data:
      logger.warning("Response model contains no prediction columns.")
      raise ValueError("Response model contains no prediction columns.")

    if len(data) > 1:
      prediction = MultiTimeSeries(data, index=index)
      prediction.index.name = "date"
    else:
      column_name, values = next(iter(data.items()))
      prediction = UniTimeSeries(values, index=index, name=column_name)
      prediction.index.name = "date"

    return ModelResponse(
      prediction=prediction,
      input_tokens=input_tokens,
      output_tokens=output_tokens,
      time=execution_time
    )
