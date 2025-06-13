from typing import List
import os

import json
import jsonlines

from trinity.buffer.buffer_writer import BufferWriter
from trinity.common.config import BufferConfig, StorageConfig
from trinity.common.constants import StorageType
from trinity.common.experience import Experience
from trinity.common.workflows import Task


class _Encoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, Experience):
            return o.to_dict()
        if isinstance(o, Task):
            return o.to_dict()
        return super().default(o)


class JSONWriter(BufferWriter):

    def __init__(self, meta: StorageConfig, config: BufferConfig):
        assert meta.storage_type == StorageType.FILE
        if meta.path is None:
            raise ValueError("File path cannot be None for RawFileWriter")
        ext = os.path.splitext(meta.path)[-1]
        if ext != ".jsonl" and ext != ".json":
            raise ValueError(f"File path must end with .json or .jsonl, got {meta.path}")
        self.writer = jsonlines.open(
            meta.path, mode="a", dumps=_Encoder(ensure_ascii=False).encode, flush=True
        )

    def write(self, data: List) -> None:
        self.writer.write_all(data)

    def finish(self):
        self.writer.close()
