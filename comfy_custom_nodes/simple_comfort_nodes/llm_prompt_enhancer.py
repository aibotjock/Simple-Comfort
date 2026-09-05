"""LLM-backed prompt enhancer node.

Rewrites a short user idea into a rich, cinematic prompt tuned for the
Wan 2.2 video models by calling a local OpenAI-compatible chat endpoint
(LM Studio's server, http://127.0.0.1:1234). Uses only the Python standard
library so the node has no dependencies beyond ComfyUI itself.

Design notes:
- ``enabled`` off, a dead endpoint, a timeout, or a malformed response all
  degrade to returning ``text`` unchanged (with a console warning) so the
  graph still renders. Enhancement is an upgrade, never a hard dependency.
- ``reroll`` exists only to bust ComfyUI's input-hash cache: bump it to
  re-enhance the same text and get a different take.
- The cleanup pass strips the artifacts LLMs love to add (markdown fences,
  surrounding quotes, "Enhanced prompt:" preambles) so downstream
  CLIPTextEncode receives pure prompt text.
"""
from __future__ import annotations

import json
import re
import urllib.error
import urllib.request

# System prompt tuned for Wan 2.2: the model responds best to flowing
# descriptive prose that covers subject, chronological motion, camera,
# lighting and style — not tag soup, and no invented content.
WAN22_SYSTEM_PROMPT = (
    "You rewrite short video ideas into rich, cinematic prompts for the "
    "Wan 2.2 video generation model.\n"
    "Rules:\n"
    "- Expand the idea into one flowing descriptive paragraph of 2-4 "
    "sentences, at most 90 words.\n"
    "- Cover, only when implied by the idea: subject appearance, the "
    "action and how it unfolds over time (describe motion "
    "chronologically), camera framing and movement (e.g. slow dolly-in, "
    "tracking shot, handheld), lighting (e.g. soft golden-hour light, "
    "dramatic rim light), atmosphere and mood, environment detail, and "
    "film-style terms (cinematic, shallow depth of field, 35mm film).\n"
    "- Never invent new subjects, people, brands or locations the idea "
    "does not imply.\n"
    "- Never add on-screen text, captions, watermarks or logos.\n"
    "- Do not mention resolutions, quality scores or technical settings.\n"
    "- Output ONLY the rewritten prompt: no preamble, no quotes, no "
    "explanation."
)

_PREFIXES = [
    "enhanced prompt:", "prompt:", "here is the prompt:",
    "here's the prompt:", "rewritten prompt:", "output:",
]


def _clean(reply: str) -> str:
    """Strip common LLM artifacts from the reply."""
    text = reply.strip()
    # ```markdown fences (optionally wrapped around the whole reply)
    if text.startswith("```"):
        text = re.sub(r"^```[a-zA-Z0-9]*\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Matching surrounding quotes
    for a, b in (('"', '"'), ("'", "'"), ("“", "”")):
        if len(text) >= 2 and text.startswith(a) and text.endswith(b):
            text = text[1:-1].strip()
    low = text.lower()
    for prefix in _PREFIXES:
        if low.startswith(prefix):
            text = text[len(prefix):].lstrip("-: ").strip()
            break
    # Collapse internal runs of whitespace/newlines into single spaces
    return re.sub(r"\s+", " ", text).strip()


class LLMPromptEnhancer:
    CATEGORY = "simple_comfort"
    DESCRIPTION = (
        "Rewrite a short idea into a rich cinematic video prompt using a "
        "local OpenAI-compatible LLM (LM Studio). Falls back to passing "
        "the text through unchanged if the server is unreachable."
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {
                    "multiline": True, "default": "",
                    "tooltip": "The raw idea to expand into a full prompt",
                }),
                "model": ("STRING", {
                    "default": "qwen/qwen3.8-27b",
                    "tooltip": "Model id as served by the LLM endpoint "
                               "(see the endpoint's /v1/models)",
                }),
                "system_prompt": ("STRING", {
                    "multiline": True, "default": WAN22_SYSTEM_PROMPT,
                    "tooltip": "Instructions the LLM rewrites with",
                }),
                "endpoint": ("STRING", {
                    "default": "http://127.0.0.1:1234/v1/chat/completions",
                    "tooltip": "OpenAI-compatible chat completions URL",
                }),
                "temperature": ("FLOAT", {
                    "default": 0.7, "min": 0.0, "max": 2.0, "step": 0.05,
                    "tooltip": "Creativity of the rewrite",
                }),
                "max_tokens": ("INT", {
                    "default": 512, "min": 64, "max": 4096, "step": 32,
                    "tooltip": "Reply length cap",
                }),
                "timeout": ("INT", {
                    "default": 120, "min": 5, "max": 600, "step": 5,
                    "tooltip": "Seconds to wait for the LLM before "
                               "falling back to the raw text",
                }),
                "reroll": ("INT", {
                    "default": 0, "min": 0, "max": 999999,
                    "tooltip": "Bump to re-enhance the same text "
                               "(busts the execution cache)",
                }),
                "enabled": ("BOOLEAN", {
                    "default": True,
                    "tooltip": "Off = pass text through unchanged",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("text",)
    FUNCTION = "enhance"
    OUTPUT_NODE = False

    def enhance(self, text, model, system_prompt, endpoint, temperature,
                max_tokens, timeout, reroll, enabled):
        text = (text or "").strip()
        if not text:
            return (text,)
        if not enabled:
            print("[simple_comfort] LLMPromptEnhancer: disabled, "
                  "passing text through")
            return (text,)

        payload = json.dumps({
            "model": model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": text},
            ],
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False,
        }).encode("utf-8")

        request = urllib.request.Request(
            endpoint,
            data=payload,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as resp:
                body = json.loads(resp.read().decode("utf-8"))
            reply = body["choices"][0]["message"]["content"]
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError,
                KeyError, IndexError, ValueError, json.JSONDecodeError,
                OSError) as exc:
            print(f"[simple_comfort] LLMPromptEnhancer: LLM call failed "
                  f"({exc!r}) — passing the raw text through")
            return (text,)

        enhanced = _clean(reply or "")
        if not enhanced:
            print("[simple_comfort] LLMPromptEnhancer: empty reply — "
                  "passing the raw text through")
            return (text,)

        print(f"[simple_comfort] LLMPromptEnhancer: {text[:60]!r} -> "
              f"{enhanced[:60]!r}...")
        return (enhanced,)


NODE_CLASS_MAPPINGS = {
    "LLMPromptEnhancer": LLMPromptEnhancer,
}
