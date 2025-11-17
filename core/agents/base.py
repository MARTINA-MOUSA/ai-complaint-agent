"""Base agent using LlamaIndex ReActAgent framework."""

from __future__ import annotations

from typing import Any, Optional

from llama_index.core.agent import ReActAgent
from llama_index.core.base.llms.types import CompletionResponse
from llama_index.core.tools import FunctionTool


class LlamaIndexAgent:
    """Base agent wrapper using LlamaIndex ReActAgent."""

    def __init__(
        self,
        *,
        llm: Any,
        system_prompt: str,
        tools: Optional[list[FunctionTool]] = None,
        verbose: bool = False,
    ) -> None:
        self.llm = llm
        self.system_prompt = system_prompt
        self.tools = tools or []
        self.verbose = verbose

        # Create a simple LLM adapter for LlamaIndex
        self.llm_adapter = self._create_llm_adapter(llm)

        # Create ReActAgent
        self.agent = ReActAgent.from_tools(
            tools=self.tools,
            llm=self.llm_adapter,
            system_prompt=system_prompt,
            verbose=verbose,
        )

    def _create_llm_adapter(self, llm: Any):
        """Create LlamaIndex LLM adapter from custom LLM."""
        from llama_index.core.llms import CustomLLM
        from llama_index.core.base.llms.types import LLMMetadata

        class GeminiLLMAdapter(CustomLLM):
            def __init__(self, wrapped_llm):
                super().__init__(
                    model="gemini-2.5-flash",
                    metadata=LLMMetadata(
                        num_output=4096,
                        is_chat_model=False,
                    ),
                )
                self._wrapped = wrapped_llm

            def complete(self, prompt: str, **kwargs):
                result = self._wrapped.complete(prompt)
                text = self._extract_text(result)
                return CompletionResponse(text=text)

            async def acomplete(self, prompt: str, **kwargs):
                result = await self._wrapped.acomplete(prompt)
                text = self._extract_text(result)
                return CompletionResponse(text=text)

            @staticmethod
            def _extract_text(raw: Any) -> str:
                if isinstance(raw, str):
                    return raw
                if hasattr(raw, "text"):
                    return getattr(raw, "text", "")
                if hasattr(raw, "response"):
                    return getattr(raw, "response", "")
                return str(raw)

        return GeminiLLMAdapter(llm)

    async def achat(self, message: str) -> str:
        """Chat with the agent asynchronously."""
        try:
            response = await self.agent.achat(message)
            return str(response)
        except Exception as e:
            # Fallback: use LLM directly if agent fails
            if self.verbose:
                print(f"Agent chat failed, using direct LLM: {e}")
            full_prompt = f"{self.system_prompt}\n\n{message}"
            result = await self.llm.acomplete(full_prompt)
            return self._extract_text(result)

    def chat(self, message: str) -> str:
        """Chat with the agent synchronously."""
        try:
            response = self.agent.chat(message)
            return str(response)
        except Exception as e:
            # Fallback: use LLM directly if agent fails
            if self.verbose:
                print(f"Agent chat failed, using direct LLM: {e}")
            full_prompt = f"{self.system_prompt}\n\n{message}"
            result = self.llm.complete(full_prompt)
            return self._extract_text(result)

    @staticmethod
    def _extract_text(raw: Any) -> str:
        """Extract text from LLM response."""
        if isinstance(raw, str):
            return raw.strip()
        if hasattr(raw, "text"):
            return getattr(raw, "text", "").strip()
        if hasattr(raw, "response"):
            return getattr(raw, "response", "").strip()
        return str(raw).strip()
