from litellm.llms.deepseek.chat.transformation import DeepSeekChatConfig


class TestDeepSeekChatTransformation:
    def test_fill_reasoning_content_adds_content_for_reasoning_only_assistant(self):
        config = DeepSeekChatConfig()
        messages = config._fill_reasoning_content(
            [
                {
                    "role": "assistant",
                    "content": None,
                    "reasoning_content": "thinking step",
                }
            ]
        )

        assert messages[0]["content"] == " "
        assert messages[0]["reasoning_content"] == "thinking step"
