"""Chat tokenisation for the Qwen3.5 prompt-enhancer node.

Uses the `tokenizers` fast tokenizer shipped with the checkpoint plus the model's
own Jinja chat template, so the exact prompt format matches the reference model.
"""
import json
import os

from jinja2 import Template
from tokenizers import Tokenizer

DEFAULT_SYSTEM = (
    """You are an expert image prompt generation system. Your sole task is to take the user's input and expand it into a detailed, descriptive prompt using complete, grammatically correct sentences.

    Rules & Guidelines:
    1. Always retain and rich-up all original adjectives, adverbs, and key attributes provided by the user. Do not shorten, truncate, or simplify the input.
    2. Formulate the expanded prompt as fluid, fully constructed sentences. Do not use comma-separated tag lists, short phrases, or bullet points.
    3. Elaborate naturally on visual details such as lighting, atmosphere, textures, colors, composition, and mood while maintaining sentence structure.
    4. Output ONLY the final expanded caption text. Do not include introductory remarks, explanations, conversational filler, conversational greetings, quotes, tags, or markdown headers."""
)

# Tokens that end a generation turn.
STOP_STRINGS = ("<|im_end|>", "<|endoftext|>")


class ChatTokenizer:
    def __init__(self, model_dir):
        self.model_dir = model_dir
        self.tok = Tokenizer.from_file(os.path.join(model_dir, "tokenizer.json"))
        tmpl_path = os.path.join(model_dir, "chat_template.jinja")
        with open(tmpl_path, encoding="utf-8") as f:
            self.template = Template(f.read())
        self.stop_ids = tuple(
            self.tok.token_to_id(s) for s in STOP_STRINGS
            if self.tok.token_to_id(s) is not None
        )

    def render_prompt(self, system_prompt, user_text, enable_thinking=False):
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_text})
        return self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    def render_prompt_mm(self, system_prompt, user_text, n_images,
                         enable_thinking=False):
        """Render the chat template with `n_images` image placeholders leading
        the user turn. The template expands each to
        `<|vision_start|><|image_pad|><|vision_end|>`; the single `<|image_pad|>`
        marker per image must be expanded to the grid-derived count by the
        caller (same contract as the HF processor)."""
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        content = [{"type": "image"} for _ in range(n_images)]
        content.append({"type": "text", "text": user_text})
        messages.append({"role": "user", "content": content})
        return self.template.render(
            messages=messages,
            add_generation_prompt=True,
            enable_thinking=enable_thinking,
        )

    @staticmethod
    def expand_image_pads(text, pad_counts):
        """Replace the n-th single `<|image_pad|>` marker in `text` with
        `pad_counts[n]` copies of it, in order."""
        for count in pad_counts:
            text = text.replace("<|image_pad|>", "<|image_pad|>" * int(count), 1)
        return text

    def encode_chat_mm(self, system_prompt, user_text, pad_counts,
                       enable_thinking=False):
        """Render + expand image markers and tokenize. Returns (ids, spans)
        where spans is a list of (start, end) absolute token positions of the
        `<|image_pad|>` runs, one per image, in order."""
        text = self.render_prompt_mm(system_prompt, user_text, len(pad_counts),
                                     enable_thinking=enable_thinking)
        text = self.expand_image_pads(text, pad_counts)
        ids = self.tok.encode(text).ids
        pad_id = self.tok.token_to_id("<|image_pad|>")
        spans = []
        pos = 0
        for count in pad_counts:
            start = ids.index(pad_id, pos)
            spans.append((start, start + int(count)))
            pos = start + int(count)
        return ids, spans

    def encode_chat(self, system_prompt, user_text, enable_thinking=False):
        text = self.render_prompt(system_prompt, user_text, enable_thinking)
        return self.tok.encode(text).ids

    def encode_chat_prefix_len(self, system_prompt, ids):
        """Length of the system-turn prefix within an `encode_chat` id list.

        The prefix is everything up to the start of the user turn, so the
        engine can snapshot the state after the (expensive, identical) system
        prompt and restore it instead of re-running it.  Falls back to the
        end of the system turn (the first `<|im_end|>`) if BPE boundary
        effects make the empty-user render differ from a prefix of the full
        encoding.
        """
        if not system_prompt:
            return 0
        pre = self.tok.encode(self.render_prompt(system_prompt, "")).ids
        if len(pre) <= len(ids) and ids[: len(pre)] == pre:
            return len(pre)
        end = self.tok.token_to_id("<|im_end|>")
        if end is not None and ids:
            for i, t in enumerate(ids):
                if t == end:
                    return i + 1
        return 0

    def encode(self, text):
        return self.tok.encode(text).ids

    def token_to_id(self, token):
        """Token id for a literal string (e.g. a newline).

        `Tokenizer.token_to_id` looks up byte-mapped vocab entries ("Ċ"), so
        it misses "\n"/" " literals; fall back to a single-token encode.
        """
        tid = self.tok.token_to_id(token)
        if tid is not None:
            return tid
        enc = self.tok.encode(token).ids
        return enc[0] if len(enc) == 1 else None

    def decode(self, ids):
        return self.tok.decode(ids, skip_special_tokens=True)

    def decode_stream(self, ids):
        """Decode a token list for progressive display (special tokens stripped)."""
        return self.tok.decode(ids, skip_special_tokens=True)
