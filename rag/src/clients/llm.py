import openai
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import SecretStr

from errors import LLMError
from models import LLMConfig


class LLMClient:
    """
    Thin client over an OpenAI-style chat completions endpoint.
    """

    def __init__(self, config: LLMConfig) -> None:
        """
        Build the chat model from the generation settings.

        :param config: Answer generation service settings.
        """
        self._model = ChatOpenAI(
            base_url=config.url,
            api_key=config.api_key or SecretStr("-"),
            model=config.model,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
            timeout=config.timeout_sec,
        )

    async def complete(self, prompt: ChatPromptTemplate, values: dict[str, str]) -> str:
        """
        Render a prompt template and complete it with the chat model.

        :param prompt: The prompt template to render.
        :param values: Values for the template placeholders.
        :raises LLMError: If the service is unreachable or returns an unusable response.
        :return: The generated text.
        """
        try:
            message = await (prompt | self._model).ainvoke(values)
        except openai.OpenAIError as e:
            raise LLMError(f"completion request failed: {e}") from e
        return message.text()
