# Copyright 2025 Bytedance Ltd. and/or its affiliates
import logging
import os

logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


def initialize_system_prompt(tokenizer, **apply_chat_template_kwargs) -> list[int]:
    """
    Initialize system prompt tokens for chat templates that support them.

    Args:
        tokenizer: The tokenizer with a chat template
        **apply_chat_template_kwargs: Additional arguments for apply_chat_template

    Returns:
        List of token IDs for the system prompt, or empty list if not supported
    """
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    return system_prompt


def extract_system_prompt_and_generation(tokenizer):
    token1 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}], add_generation_prompt=False, tokenize=True
    )
    token2 = tokenizer.apply_chat_template(
        [{"role": "user", "content": ""}] * 2, add_generation_prompt=False, tokenize=True
    )
    # get system prompt tokens
    system_prompt = token1[: -(len(token2) - len(token1))]
    # get generate prompt tokens
    token3 = tokenizer.apply_chat_template([{"role": "user", "content": ""}], add_generation_prompt=True, tokenize=True)
    generate_prompt = token3[len(token1) :]

    return system_prompt, generate_prompt


def apply_chat_template_single_turn(
    processor,
    messages: list[dict],
    full_conversation: list[dict],
    turn_index: int,
    tools=None,
    **kwargs,
):
    """Apply chat_template to a single turn's ``messages``, with automatic
    fallback for templates that require a user message (e.g. Qwen 3.5's
    "No user query found" error).

    When the direct call fails, the function tokenises the full conversation up
    to *turn_index* and subtracts the prefix produced by everything before
    *turn_index*, yielding only this turn's tokens.

    Args:
        processor: tokenizer or processor that has ``apply_chat_template``.
        messages: the message(s) to tokenise (typically ``[single_msg]``).
        full_conversation: the complete conversation list for fallback context.
        turn_index: 0-based position of the **last** message of ``messages``
            inside ``full_conversation``.
        tools: tool schemas forwarded to ``apply_chat_template``.
        **kwargs: extra keyword arguments forwarded to ``apply_chat_template``
            (e.g. ``add_generation_prompt``, ``return_tensors``, ``tokenize``).

    Returns:
        Same type as ``processor.apply_chat_template`` — typically a ``dict``
        (when ``return_dict=True``) or a ``list[int]``.
    """
    try:
        return processor.apply_chat_template(messages, tools=tools, **kwargs)
    except Exception as e:
        if "No user query" not in str(e):
            raise

        inputs_full = processor.apply_chat_template(
            full_conversation[: turn_index + 1],
            tools=tools,
            **kwargs,
        )
        prefix_len = 0
        if turn_index > 0:
            prefix_tools = tools if turn_index == 1 else None
            inputs_prev = processor.apply_chat_template(
                full_conversation[:turn_index],
                tools=prefix_tools,
                **kwargs,
            )
            if isinstance(inputs_prev, dict):
                prefix_len = inputs_prev["input_ids"].shape[-1]
            else:
                prefix_len = len(inputs_prev)

        if isinstance(inputs_full, dict):
            return {k: v[..., prefix_len:] for k, v in inputs_full.items()}
        return inputs_full[prefix_len:]
