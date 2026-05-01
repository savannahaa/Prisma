from __future__ import annotations

import os
from pathlib import Path

import numpy as np


class LazySyntheticDocIds:
    def __init__(self, *, prefix: str, num_docs: int) -> None:
        self._prefix = str(prefix).strip() or "synthetic_doc"
        self._num_docs = int(max(0, num_docs))

    def __len__(self) -> int:
        return int(self._num_docs)

    def __getitem__(self, index):
        if isinstance(index, slice):
            start, stop, step = index.indices(int(self._num_docs))
            return [self[ii] for ii in range(start, stop, step)]
        idx = int(index)
        if idx < 0:
            idx += int(self._num_docs)
        if idx < 0 or idx >= int(self._num_docs):
            raise IndexError(idx)
        return f"{self._prefix}_{idx}"


class LazySyntheticTexts:
    def __init__(self, *, doc_ids) -> None:
        self._doc_ids = doc_ids

    def __len__(self) -> int:
        return int(len(self._doc_ids))

    def __getitem__(self, index):
        return str(self._doc_ids[index])


def load_doc_ids_or_synthetic(
    path: str | os.PathLike[str],
    *,
    num_docs: int,
    allow_synthetic: bool,
    default_prefix: str = "",
):
    if os.path.exists(path):
        return np.load(path, allow_pickle=True), False
    if not bool(allow_synthetic):
        raise FileNotFoundError(f"missing required file: {path}")
    prefix = str(default_prefix).strip() or Path(path).stem or "synthetic_doc"
    return LazySyntheticDocIds(prefix=prefix, num_docs=int(num_docs)), True
