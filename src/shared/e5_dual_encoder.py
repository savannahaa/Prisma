"""
E5 双塔编码器封装（query / passage）。

主要目标：
1) 提供统一编码接口：`encode_queries` 与 `encode_passages`。
2) 统一做池化 + L2 归一化，确保余弦检索口径一致。

注意：
- query 侧会自动加 `query: ` 前缀，passage 侧加 `passage: ` 前缀。
"""

import os
from typing import List, Tuple

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer
from shared.gpu_accel import resolve_device_spec


EPS = 1e-12


def _load_from_pretrained_local_first(loader, model_name: str, *, log_prefix: str, component_name: str):
    # Fresh runs should prefer the already-populated local HF cache so transient network
    # issues do not break the paperfaithful mainline workflow.
    try:
        print(f"[{log_prefix}] loading {component_name} from local cache: {model_name}", flush=True)
        return loader.from_pretrained(model_name, local_files_only=True)
    except Exception as exc:
        print(
            f"[{log_prefix}] local cache load for {component_name} failed "
            f"({exc.__class__.__name__}: {exc}); falling back to online lookup",
            flush=True,
        )
        return loader.from_pretrained(model_name)


#向量编码器总入口
#归一化和平均池化，即把一句话的所有 token 向量平均成一个向量，并且归一化到单位球面上
def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norms = np.linalg.norm(x, axis=1, keepdims=True)
    norms = np.maximum(norms, EPS)
    return (x / norms).astype(np.float32)


def normalize_vec(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    norm = float(np.linalg.norm(x))
    if norm <= EPS:
        raise ValueError("向量范数过小。")
    return (x / norm).astype(np.float32)


def average_pool(last_hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    masked = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    denom = attention_mask.sum(dim=1)[..., None].clamp(min=1)
    return masked.sum(dim=1) / denom

#初始化编码器：加载分词器、E5模型、几何校正层
class E5DualEncoder:
    def __init__(
        self,
        model_name: str,
        log_prefix: str = "e5-encoder",
        device: str | None = None,
        device_role: str = "default",
    ):
        self.log_prefix = log_prefix
        self._configure_torch_threads()
        requested_device = str(device).strip() if device is not None else str(resolve_device_spec(device_role))
        if torch.cuda.is_available() and requested_device:
            self.device = torch.device(requested_device)
        else:
            self.device = torch.device("cpu")
        self.tokenizer = _load_from_pretrained_local_first(
            AutoTokenizer,
            model_name,
            log_prefix=self.log_prefix,
            component_name="tokenizer",
        )
        self.model = _load_from_pretrained_local_first(
            AutoModel,
            model_name,
            log_prefix=self.log_prefix,
            component_name="AutoModel",
        )
        self.model.to(self.device)
        self.model.eval()
        print(f"[{self.log_prefix}] model ready on {self.device}", flush=True)

    def _configure_torch_threads(self) -> None:
        num_threads_raw = (
            os.getenv("TORCH_NUM_THREADS")
            or os.getenv("OMP_NUM_THREADS")
            or os.getenv("CPU_NUM_THREADS")
            or "32"
        )
        interop_threads_raw = os.getenv("TORCH_NUM_INTEROP_THREADS", "1")
        if num_threads_raw:
            try:
                torch.set_num_threads(max(1, int(num_threads_raw)))
            except Exception as exc:
                print(
                    f"[{self.log_prefix}] failed to set torch num threads from {num_threads_raw}: "
                    f"{exc.__class__.__name__}: {exc}",
                    flush=True,
                )
        if interop_threads_raw:
            try:
                torch.set_num_interop_threads(max(1, int(interop_threads_raw)))
            except Exception as exc:
                print(
                    f"[{self.log_prefix}] failed to set torch interop threads from {interop_threads_raw}: "
                    f"{exc.__class__.__name__}: {exc}",
                    flush=True,
                )
        print(
            f"[{self.log_prefix}] torch threads={torch.get_num_threads()} "
            f"interop_threads={torch.get_num_interop_threads()}",
            flush=True,
        )

    #底层编码一批文本，查询前面加query，文档前面加passage
    def encode_batch(self, texts: List[str], prefix: str, max_length: int = 512) -> np.ndarray:
        if len(texts) == 0:
            raise ValueError("batch texts 为空。")

        # 统一前缀是 E5 检索效果稳定的关键约定。
        batch_texts = [f"{prefix}{t}" for t in texts]
        batch = self.tokenizer(
            batch_texts,
            max_length=max_length,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        batch = {k: v.to(self.device) for k, v in batch.items()}

        # 编码阶段禁用梯度，降低显存占用并提升推理速度。
        with torch.inference_mode():
            outputs = self.model(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
            )
            pooled = average_pool(outputs.last_hidden_state, batch["attention_mask"])

        return pooled.detach().cpu().numpy().astype(np.float32)

    def encode_texts(
        self,
        texts: List[str],
        prefix: str,
        batch_size: int = 8,
        max_length: int = 512,
        progress_name: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        if len(texts) == 0:
            raise ValueError("texts 为空。")

        raw_out = []
        total = len(texts)
        for i in range(0, total, batch_size):
            raw_out.append(self.encode_batch(texts[i:i + batch_size], prefix=prefix, max_length=max_length))
            if progress_name is not None:
                done = min(i + batch_size, total)
                print(f"[{progress_name}] done {done}/{total}", flush=True)

        # 统一输出 raw + normalized 两种表示，便于下游按需选择。
        raw = np.vstack(raw_out).astype(np.float32)
        normalized = normalize_rows(raw)
        return raw, normalized

    def encode_queries(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        progress_name: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw, normalized = self.encode_texts(
            texts=texts,
            prefix="query: ",
            batch_size=batch_size,
            max_length=max_length,
            progress_name=progress_name,
        )
        return raw, normalized

    def encode_passages(
        self,
        texts: List[str],
        batch_size: int = 8,
        max_length: int = 512,
        progress_name: str | None = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        raw, normalized = self.encode_texts(
            texts=texts,
            prefix="passage: ",
            batch_size=batch_size,
            max_length=max_length,
            progress_name=progress_name,
        )
        return raw, normalized
