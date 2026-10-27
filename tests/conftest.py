"""Bộ đồ nghề cho test: tokenizer đồ chơi (id đặc biệt trùng Llama-3) và các mô hình nhỏ khởi tạo ngẫu nhiên.

Không tải gì từ mạng. Mô hình thật (Llama-3-8B-Instruct, Qwen3.5-9B, Gemma 4 12B) chỉ chạy trên GPU thuê;
test ở đây kiểm logic của code trên mô hình nhỏ cùng kiến trúc.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pytest
import torch

GOC = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(GOC / "src"))
sys.path.insert(0, str(GOC / "scripts"))

DU_LIEU = GOC / "data" / "arditi"

# Chat template của meta-llama/Meta-Llama-3-8B-Instruct (tokenizer_config.json).
TEMPLATE_LLAMA3 = (
    "{% set loop_messages = messages %}{% for message in loop_messages %}"
    "{% set content = '<|start_header_id|>' + message['role'] + '<|end_header_id|>\n\n'+ message['content'] | trim + '<|eot_id|>' %}"
    "{% if loop.index0 == 0 %}{% set content = bos_token + content %}{% endif %}{{ content }}{% endfor %}"
    "{% if add_generation_prompt %}{{ '<|start_header_id|>assistant<|end_header_id|>\n\n' }}{% endif %}"
)

# Template kiểu Qwen3.5 rút gọn: thinking tắt → khối "<think>\n\n</think>\n\n" rỗng ở cuối.
TEMPLATE_QWEN = (
    "{% for message in messages %}<|im_start|>{{ message['role'] }}\n{{ message['content'] }}<|im_end|>\n{% endfor %}"
    "{% if add_generation_prompt %}<|im_start|>assistant\n"
    "{% if enable_thinking is defined and enable_thinking is false %}<think>\n\n</think>\n\n{% endif %}{% endif %}"
)


def _tu_trong_du_lieu() -> list[str]:
    tu = set()
    for f in DU_LIEU.glob("*.jsonl"):
        for d in f.read_text(encoding="utf-8").splitlines():
            if d.strip():
                tu.update(re.findall(r"\w+|[^\w\s]", json.loads(d)["instruction"]))
    tu.update(["user", "assistant", "system", "Sure", "sorry", "cannot", "can't", "As", "an", "AI"])
    tu.discard("I")
    return sorted(tu)


def _tao_tokenizer_llama3_do_choi():
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers, processors
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0}
    for i in range(1, 40):
        vocab[f"<f{i}>"] = i
    vocab["I"] = 40                      # token mở đầu từ chối của Llama-3
    i = 41
    for t in _tu_trong_du_lieu():
        if i == 271:
            vocab["\n\n"] = 271
            i += 1
        vocab[t] = i
        i += 1
        if i >= 120000:
            break
    if "\n\n" not in vocab:
        vocab["\n\n"] = i
        i += 1
    while i < 128000:
        vocab[f"<f{i}>"] = i
        i += 1
    dac_biet = {128000: "<|begin_of_text|>", 128001: "<|end_of_text|>", 128006: "<|start_header_id|>",
                128007: "<|end_header_id|>", 128009: "<|eot_id|>"}
    for j in range(128000, 128256):
        vocab[dac_biet.get(j, f"<|reserved_special_token_{j}|>")] = j
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in dac_biet.values()])
    tk.add_tokens([AddedToken("\n\n", normalized=False, special=False)])
    tk.post_processor = processors.TemplateProcessing(single="<|begin_of_text|> $A",
                                                      special_tokens=[("<|begin_of_text|>", 128000)])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, bos_token="<|begin_of_text|>", eos_token="<|eot_id|>",
                                  unk_token="[UNK]")
    tok.chat_template = TEMPLATE_LLAMA3
    tok.padding_side = "left"
    tok.pad_token = tok.eos_token
    return tok


def _tao_tokenizer_qwen_do_choi(so_tu: int = 400):
    from tokenizers import AddedToken, Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"[UNK]": 0}
    for t in _tu_trong_du_lieu()[: so_tu]:
        vocab[t] = len(vocab)
    them = ["\n", "\n\n", "<|im_start|>", "<|im_end|>", "<think>", "</think>", "<|endoftext|>"]
    for t in them:
        vocab[t] = len(vocab)
    tk = Tokenizer(models.WordLevel(vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = pre_tokenizers.Whitespace()
    tk.add_special_tokens([AddedToken(t, special=True, normalized=False) for t in them[2:]])
    tk.add_tokens([AddedToken(t, normalized=False, special=False) for t in them[:2]])
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>",
                                  unk_token="[UNK]")
    tok.chat_template = TEMPLATE_QWEN
    tok.padding_side = "left"
    return tok


@pytest.fixture(scope="session")
def tok_llama():
    return _tao_tokenizer_llama3_do_choi()


@pytest.fixture(scope="session")
def tok_qwen():
    return _tao_tokenizer_qwen_do_choi()


def tao_llama_nho(so_lop: int = 32, dtype=torch.float32, seed: int = 0):
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(seed)
    cfg = LlamaConfig(vocab_size=128256, hidden_size=32, intermediate_size=64, num_hidden_layers=so_lop,
                      num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=4096,
                      bos_token_id=128000, eos_token_id=128001, pad_token_id=128009, tie_word_embeddings=False,
                      initializer_range=0.2)
    m = LlamaForCausalLM(cfg).to(dtype).eval()
    m.requires_grad_(False)
    # generation_config "gây nhiễu": lấy mẫu + phạt lặp, để kiểm code luôn đặt tham số greedy tường minh.
    m.generation_config.do_sample = True
    m.generation_config.temperature = 0.7
    m.generation_config.top_k = 5
    m.generation_config.repetition_penalty = 1.5
    m.generation_config.eos_token_id = [128001, 128009]
    return m


@pytest.fixture(scope="session")
def llama_f32():
    return tao_llama_nho(dtype=torch.float32)


@pytest.fixture(scope="session")
def llama_bf16():
    return tao_llama_nho(dtype=torch.bfloat16)


def tao_qwen35_nho(vocab: int, dtype=torch.float32, seed: int = 0):
    from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=8,
                            num_attention_heads=4, num_key_value_heads=2, head_dim=8,
                            linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=8,
                            linear_value_head_dim=8,
                            layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"] * 2,
                            initializer_range=0.2)
    m = Qwen3_5ForCausalLM(cfg).to(dtype).eval()
    m.requires_grad_(False)
    return m


def tao_gemma4_nho(vocab: int, dtype=torch.float32, seed: int = 0):
    from transformers import Gemma4UnifiedForCausalLM, Gemma4UnifiedTextConfig

    torch.manual_seed(seed)
    cfg = Gemma4UnifiedTextConfig(vocab_size=vocab, hidden_size=32, intermediate_size=64, num_hidden_layers=6,
                                  num_attention_heads=2, num_key_value_heads=1, head_dim=16, sliding_window=8,
                                  layer_types=["sliding_attention"] * 5 + ["full_attention"],
                                  initializer_range=0.2, pad_token_id=0, eos_token_id=1, bos_token_id=2)
    m = Gemma4UnifiedForCausalLM(cfg).to(dtype).eval()
    m.requires_grad_(False)
    return m


def doc_cau(ten: str, n: int | None = None) -> list[str]:
    ds = [json.loads(d)["instruction"] for d in (DU_LIEU / f"{ten}.jsonl").read_text(encoding="utf-8").splitlines() if d.strip()]
    return ds[:n] if n else ds
