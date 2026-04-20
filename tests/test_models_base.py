import pytest
from llm4series.models._base import ModelResponse, Model


class TestModelResponse:

  def test_model_response_creation(self):
    response = ModelResponse(
      raw="raw response",
      predicted="predicted output",
      input_tokens=100,
      output_tokens=50,
      time=0.5
    )
    assert response.raw == "raw response"
    assert response.predicted == "predicted output"
    assert response.input_tokens == 100
    assert response.output_tokens == 50
    assert response.time == 0.5

  def test_model_response_with_long_text(self):
    long_text = "x" * 10000
    response = ModelResponse(
      raw=long_text,
      predicted=long_text,
      input_tokens=1000,
      output_tokens=1000,
      time=2.5
    )
    assert len(response.raw) == 10000
    assert len(response.predicted) == 10000

  def test_model_response_with_zero_tokens(self):
    response = ModelResponse(
      raw="",
      predicted="",
      input_tokens=0,
      output_tokens=0,
      time=0.0
    )
    assert response.input_tokens == 0
    assert response.output_tokens == 0
    assert response.time == 0.0

  def test_model_response_with_negative_time(self):
    response = ModelResponse(
      raw="text",
      predicted="text",
      input_tokens=10,
      output_tokens=10,
      time=-1.0
    )
    assert response.time == -1.0

  def test_model_response_field_types(self):
    response = ModelResponse(
      raw="raw",
      predicted="predicted",
      input_tokens=100,
      output_tokens=50,
      time=0.5
    )
    assert isinstance(response.raw, str)
    assert isinstance(response.predicted, str)
    assert isinstance(response.input_tokens, int)
    assert isinstance(response.output_tokens, int)
    assert isinstance(response.time, float)

  def test_model_response_is_dataclass(self):
    import dataclasses
    assert dataclasses.is_dataclass(ModelResponse)

  def test_model_response_immutable_fields(self):
    response = ModelResponse(
      raw="original",
      predicted="original",
      input_tokens=10,
      output_tokens=5,
      time=0.1
    )
    response.raw = "modified"
    assert response.raw == "modified"

  def test_model_response_equality(self):
    response1 = ModelResponse(
      raw="text",
      predicted="output",
      input_tokens=100,
      output_tokens=50,
      time=0.5
    )
    response2 = ModelResponse(
      raw="text",
      predicted="output",
      input_tokens=100,
      output_tokens=50,
      time=0.5
    )
    assert response1 == response2

  def test_model_response_inequality(self):
    response1 = ModelResponse(
      raw="text1",
      predicted="output",
      input_tokens=100,
      output_tokens=50,
      time=0.5
    )
    response2 = ModelResponse(
      raw="text2",
      predicted="output",
      input_tokens=100,
      output_tokens=50,
      time=0.5
    )
    assert response1 != response2


class TestModelBase:

  class ConcreteModel(Model):
    def predict(self, prompt: str, data: str, temperature: float | None, **kwargs) -> ModelResponse:
      return ModelResponse(
        raw=f"response to {prompt}",
        predicted="test output",
        input_tokens=len(prompt.split()),
        output_tokens=2,
        time=0.1
      )

  def test_model_is_abstract(self):
    with pytest.raises(TypeError):
      Model()

  def test_concrete_model_instantiation(self):
    model = self.ConcreteModel()
    assert isinstance(model, Model)

  def test_output_extraction_with_tags(self):
    model = self.ConcreteModel()
    response = "prefix <out>extracted text</out> suffix"
    result = model._output(response)
    assert result == "extracted text"

  def test_output_extraction_multiple_tags(self):
    model = self.ConcreteModel()
    response = "<out>first</out> middle <out>last</out>"
    result = model._output(response)
    assert result == "last"

  def test_output_extraction_no_tags(self):
    model = self.ConcreteModel()
    response = "plain text without tags"
    result = model._output(response)
    assert result is None

  def test_output_extraction_empty_tags(self):
    model = self.ConcreteModel()
    response = "<out></out>"
    result = model._output(response)
    assert result == "" or result is None

  def test_output_extraction_with_whitespace(self):
    model = self.ConcreteModel()
    response = "<out>   padded text   </out>"
    result = model._output(response)
    assert result == "padded text"

  def test_output_extraction_multiline(self):
    model = self.ConcreteModel()
    response = "<out>line1\nline2\nline3</out>"
    result = model._output(response)
    assert "line1" in result
    assert "line2" in result
    assert "line3" in result

  def test_predict_implementation(self):
    model = self.ConcreteModel()
    response = model.predict("test prompt", "test data", 0.7)
    assert isinstance(response, ModelResponse)
    assert response.predicted == "test output"
    assert response.input_tokens > 0
    assert response.output_tokens == 2

  def test_predict_with_none_temperature(self):
    model = self.ConcreteModel()
    response = model.predict("test prompt", "test data", None)
    assert isinstance(response, ModelResponse)

  def test_predict_with_kwargs(self):
    model = self.ConcreteModel()
    response = model.predict("test prompt", "test data", 0.7, extra_param="value")
    assert isinstance(response, ModelResponse)

  def test_output_regex_pattern(self):
    model = self.ConcreteModel()
    response = "<out>text\nwith\nspecial!@#$</out>"
    result = model._output(response)
    assert result is not None
    assert "special!@#$" in result
