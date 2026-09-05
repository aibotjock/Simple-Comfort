"""Simple-Comfort custom nodes for ComfyUI.

Ships the in-graph LLM prompt enhancer used by the enhanced Wan 2.2
workflows: a short idea goes in, a rich cinematic video-model prompt comes
out, rewritten by a local OpenAI-compatible LLM server (LM Studio on this
box). Falls back to passing the text through untouched if the LLM server is
unreachable, so graphs never hard-fail because the enhancer is down.
"""
from .llm_prompt_enhancer import LLMPromptEnhancer, WAN22_SYSTEM_PROMPT

NODE_CLASS_MAPPINGS = {
    "LLMPromptEnhancer": LLMPromptEnhancer,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LLMPromptEnhancer": "LLM Prompt Enhancer (Simple-Comfort)",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS",
           "LLMPromptEnhancer", "WAN22_SYSTEM_PROMPT"]
