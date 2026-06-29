import unittest

import torch

from trinity.buffer.store import MemoryStore, parse_record_key
from trinity.common.experience import EID, Experience
from trinity.common.models.recording.recorder import Recorder


def make_turn(
    *,
    request_id: str,
    record_key: str,
    tokens: list[int],
    prompt_length: int,
    logprobs: list[float],
    sample_id: str | None = None,
    sample_index: int = 0,
) -> Experience:
    batch, task, run = parse_record_key(record_key)
    info = {"sample_index": sample_index}
    if sample_id is not None:
        info["sample_id"] = sample_id
    return Experience(
        eid=EID(batch=batch, task=task, run=run, suffix=request_id),
        tokens=tokens,
        prompt_length=prompt_length,
        logprobs=logprobs,
        info=info,
    )


class RecorderPrefixMergeTest(unittest.IsolatedAsyncioTestCase):
    async def test_prefix_experiences_merge_and_keep_final_sample_id(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        first = make_turn(
            request_id="req-1",
            record_key=record_key,
            tokens=[10, 11, 20, 21],
            prompt_length=2,
            logprobs=[-0.2, -0.3],
            sample_id="sample-old",
        )
        second = make_turn(
            request_id="req-2",
            record_key=record_key,
            tokens=[10, 11, 20, 21, 12, 13, 30, 31, 32],
            prompt_length=6,
            logprobs=[-0.4, -0.5, -0.6],
            sample_id="sample-final",
        )

        await recorder._safe_append(first)
        await recorder._safe_append(second)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 1)
        merged = recorded[0]
        self.assertEqual(merged.info["sample_id"], "sample-final")
        self.assertEqual(merged.eid.suffix, "req-2")
        self.assertEqual(merged.prompt_length, 2)
        self.assertTrue(torch.equal(merged.tokens, second.tokens))
        self.assertTrue(
            torch.equal(
                merged.action_mask,
                torch.tensor([True, True, False, False, True, True, True]),
            )
        )
        self.assertTrue(
            torch.allclose(
                merged.logprobs,
                torch.tensor([-0.2, -0.3, 0.0, 0.0, -0.4, -0.5, -0.6]),
            )
        )
        self.assertEqual(merged.info["merged_eid_suffixes"], ["req-1", "req-2"])
        self.assertEqual(merged.info["merged_sample_ids"], ["sample-old", "sample-final"])

        store.update(record_key, reward=1.0, info=None, sample_ids=["sample-final"])
        self.assertEqual(store.get(record_key)[0].reward, 1.0)
        with self.assertRaises(KeyError):
            store.update(record_key, reward=2.0, info=None, sample_ids=["sample-old"])

    async def test_non_prefix_experiences_do_not_merge(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"

        await recorder._safe_append(
            make_turn(
                request_id="req-1",
                record_key=record_key,
                tokens=[10, 11, 20],
                prompt_length=2,
                logprobs=[-0.2],
            )
        )
        await recorder._safe_append(
            make_turn(
                request_id="req-2",
                record_key=record_key,
                tokens=[10, 12, 30],
                prompt_length=2,
                logprobs=[-0.3],
            )
        )

        self.assertEqual(len(store.get(record_key)), 2)

    async def test_merge_head_replaces_only_matching_sample_stream(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        sample_zero = make_turn(
            request_id="req-1",
            record_key=record_key,
            tokens=[10, 11, 20],
            prompt_length=2,
            logprobs=[-0.2],
            sample_id="sample-zero",
            sample_index=0,
        )
        sample_one_first = make_turn(
            request_id="req-2",
            record_key=record_key,
            tokens=[10, 11, 21],
            prompt_length=2,
            logprobs=[-0.3],
            sample_id="sample-one-old",
            sample_index=1,
        )
        sample_one_final = make_turn(
            request_id="req-3",
            record_key=record_key,
            tokens=[10, 11, 21, 12, 31],
            prompt_length=4,
            logprobs=[-0.4],
            sample_id="sample-one-final",
            sample_index=1,
        )

        await recorder._safe_append(sample_zero)
        await recorder._safe_append(sample_one_first)
        await recorder._safe_append(sample_one_final)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 2)
        self.assertEqual(recorded[0].info["sample_id"], "sample-zero")
        self.assertEqual(recorded[1].info["sample_id"], "sample-one-final")
        self.assertTrue(
            torch.equal(
                recorded[1].action_mask,
                torch.tensor([True, False, True]),
            )
        )

    async def test_stale_merge_head_falls_back_to_append(self):
        store = MemoryStore()
        recorder = Recorder(
            store=store,
            build_experiences=lambda *_args, **_kwargs: [],
            enabled=True,
        )
        record_key = "0/task_a/1"
        first = make_turn(
            request_id="req-1",
            record_key=record_key,
            tokens=[10, 11, 20],
            prompt_length=2,
            logprobs=[-0.2],
            sample_id="sample-old",
        )
        second = make_turn(
            request_id="req-2",
            record_key=record_key,
            tokens=[10, 11, 20, 12, 30],
            prompt_length=4,
            logprobs=[-0.3],
            sample_id="sample-final",
        )

        await recorder._safe_append(first)
        store.remove(record_key)
        await recorder._safe_append(second)

        recorded = store.get(record_key)
        self.assertEqual(len(recorded), 1)
        self.assertEqual(recorded[0].info["sample_id"], "sample-final")
        self.assertEqual(recorded[0].prompt_length, 4)


if __name__ == "__main__":
    unittest.main()
