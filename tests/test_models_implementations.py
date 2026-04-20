import pytest
from unittest.mock import Mock, patch
from llm4series.models.openai import OpenAI
from llm4series.models.azure import Azure
from llm4series.models.lmstudio import LMStudio
from llm4series.models._base import ModelResponse


class TestOpenAIModel:

  @pytest.fixture
  def mock_openai_client(self):
      with patch("llm4series.models.openai.Client") as mock_client:
          yield mock_client

  @pytest.fixture
  def openai_model(self, mock_openai_client):
      return OpenAI(model="gpt-4", api_key="test-key")

  def test_openai_initialization(self, mock_openai_client):
      model = OpenAI(model="gpt-4", api_key="test-key")
      assert model.model == "gpt-4"
      assert model.api_key == "test-key"
      assert model.base_url == "https://api.openai.com/v1"

  def test_openai_initialization_custom_base_url(self, mock_openai_client):
      model = OpenAI(
          model="gpt-4",
          api_key="test-key",
          base_url="https://custom.com/v1"
      )
      assert model.base_url == "https://custom.com/v1"

  def test_openai_predict_basic(self, openai_model, mock_openai_client):
      mock_response = Mock()
      mock_response.choices = [Mock(message=Mock(content="<out>test output</out>"))]
      mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
      openai_model.client.chat.completions.create.return_value = mock_response
      result = openai_model.predict("test prompt", "test data", temperature=0.7)
      assert isinstance(result, ModelResponse)
      assert result.predicted == "test output"
      assert result.input_tokens == 10
      assert result.output_tokens == 5

  def test_openai_predict_message_structure(self, openai_model, mock_openai_client):
      mock_response = Mock()
      mock_response.choices = [Mock(message=Mock(content=""))]
      mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
      openai_model.client.chat.completions.create.return_value = mock_response
      openai_model.predict("my prompt", "my data", 0.5)
      call_args = openai_model.client.chat.completions.create.call_args
      messages = call_args.kwargs["messages"]
      assert len(messages) == 3
      assert messages[0]["role"] == "system"
      assert messages[1]["role"] == "user"
      assert messages[1]["content"] == "my prompt"
      assert messages[2]["role"] == "user"
      assert messages[2]["content"] == "my data"

  def test_openai_predict_with_kwargs(self, openai_model, mock_openai_client):
      mock_response = Mock()
      mock_response.choices = [Mock(message=Mock(content=""))]
      mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
      openai_model.client.chat.completions.create.return_value = mock_response
      openai_model.predict("prompt", "data", 0.7, max_tokens=100)
      call_args = openai_model.client.chat.completions.create.call_args
      assert call_args.kwargs["max_tokens"] == 100

  def test_openai_build_response_without_usage(self, openai_model):
      mock_response = Mock()
      mock_response.choices = [Mock(message=Mock(content="<out>result</out>"))]
      mock_response.usage = None
      result = openai_model._build_response(mock_response, 0.1)
      assert result.input_tokens is None
      assert result.output_tokens is None
      assert result.predicted == "result"

  def test_openai_build_response_empty_content(self, openai_model):
      mock_response = Mock()
      mock_response.choices = [Mock(message=Mock(content=None))]
      mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
      result = openai_model._build_response(mock_response, 0.1)
      assert result.raw == ""

  def test_openai_execution_time_recorded(self, openai_model, mock_openai_client):
      mock_response = Mock()
      mock_response.choices = [Mock(message=Mock(content=""))]
      mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
      openai_model.client.chat.completions.create.return_value = mock_response
      result = openai_model.predict("p", "d", 0.7)
      assert result.time > 0


class TestAzureModel:

  @pytest.fixture
  def mock_azure_client(self):
    with patch("llm4series.models.azure.Client") as mock_client:
        yield mock_client

  @pytest.fixture
  def azure_model(self, mock_azure_client):
    return Azure(
        model="gpt-4",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-01-01"
    )

  def test_azure_initialization(self, mock_azure_client):
    model = Azure(
        model="gpt-4",
        api_key="test-key",
        azure_endpoint="https://test.openai.azure.com",
        api_version="2024-01-01"
    )
    assert model.model == "gpt-4"
    mock_azure_client.assert_called_once()

  def test_azure_predict_basic(self, azure_model, mock_azure_client):
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="<out>result</out>"))]
    mock_response.usage = Mock(prompt_tokens=10, completion_tokens=5)
    azure_model.client.chat.completions.create.return_value = mock_response
    result = azure_model.predict("prompt", "data", 0.7)
    assert isinstance(result, ModelResponse)
    assert result.predicted == "result"

  def test_azure_predict_message_structure(self, azure_model):
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=""))]
    mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
    azure_model.client.chat.completions.create.return_value = mock_response
    azure_model.predict("test", "data", 0.5)
    call_args = azure_model.client.chat.completions.create.call_args
    messages = call_args.kwargs["messages"]
    assert len(messages) == 3
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"
    assert messages[2]["role"] == "user"

  def test_azure_build_response(self, azure_model):
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="output"))]
    mock_response.usage = Mock(prompt_tokens=5, completion_tokens=3)
    result = azure_model._build_response(mock_response, 0.2)
    assert result.raw == "output"
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.time == 0.2


class TestLMStudioModel:

  @pytest.fixture
  def mock_lmstudio_client(self):
    with patch("llm4series.models.lmstudio.Client") as mock_client:
      yield mock_client

  def test_lmstudio_initialization_default(self, mock_lmstudio_client):
    model = LMStudio(model="neural-chat")
    assert model.model == "neural-chat"
    assert model.base_url == "http://localhost:1234/v1"

  def test_lmstudio_initialization_custom_url(self, mock_lmstudio_client):
    model = LMStudio(model="neural-chat", base_url="http://custom:5000/v1")
    assert model.base_url == "http://custom:5000/v1"

  def test_lmstudio_predict_creates_client_per_call(self, mock_lmstudio_client):
    model = LMStudio(model="neural-chat")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=""))]
    mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_lmstudio_client.return_value = mock_client_instance
    model.predict("prompt", "data", 0.7)
    mock_lmstudio_client.assert_called_with(
        api_key="not-needed",
        base_url="http://localhost:1234/v1"
    )

  def test_lmstudio_predict_response(self, mock_lmstudio_client):
    model = LMStudio(model="neural-chat")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="<out>output</out>"))]
    mock_response.usage = Mock(prompt_tokens=8, completion_tokens=4)
    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_lmstudio_client.return_value = mock_client_instance
    result = model.predict("prompt", "data", 0.7)
    assert isinstance(result, ModelResponse)
    assert result.predicted == "output"
    assert result.input_tokens == 8
    assert result.output_tokens == 4

  def test_lmstudio_no_api_key_required(self, mock_lmstudio_client):
    model = LMStudio(model="local-model")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=""))]
    mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)
    mock_client_instance = Mock()
    mock_client_instance.chat.completions.create.return_value = mock_response
    mock_lmstudio_client.return_value = mock_client_instance
    model.predict("p", "d", 0.7)
    call_args = mock_lmstudio_client.call_args
    assert call_args.kwargs["api_key"] == "not-needed"

  def test_lmstudio_build_response(self):
    model = LMStudio(model="neural-chat")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content="test output"))]
    mock_response.usage = Mock(prompt_tokens=5, completion_tokens=3)
    result = model._build_response(mock_response, 0.15)
    assert result.raw == "test output"
    assert result.input_tokens == 5
    assert result.output_tokens == 3
    assert result.time == 0.15

  def test_lmstudio_message_structure(self):
    model = LMStudio(model="neural-chat")
    mock_response = Mock()
    mock_response.choices = [Mock(message=Mock(content=""))]
    mock_response.usage = Mock(prompt_tokens=0, completion_tokens=0)

    with patch("llm4series.models.lmstudio.Client") as mock_client:
      mock_instance = Mock()
      mock_instance.chat.completions.create.return_value = mock_response
      mock_client.return_value = mock_instance
      model.predict("my prompt", "my data", 0.6)
      call_args = mock_instance.chat.completions.create.call_args
      messages = call_args.kwargs["messages"]
      assert len(messages) == 3
      assert messages[1]["content"] == "my prompt"
      assert messages[2]["content"] == "my data"
