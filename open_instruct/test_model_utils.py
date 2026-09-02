import contextlib
import pathlib
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

import torch

import open_instruct.model_utils
from open_instruct.model_utils import Batch, TensorCache


class TestPortableModelStateDict(unittest.TestCase):
    def test_save_and_load_vanilla_model(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = pathlib.Path(tmpdir) / "model.bin"
            open_instruct.model_utils.save_model_state_dict(source, checkpoint_path, rank=0)
            open_instruct.model_utils.maybe_load_checkpoint(target, str(checkpoint_path), torch.device("cpu"), rank=0)

        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_parameter, target_parameter))

    def test_zero3_save_gathers_each_parameter(self):
        model = torch.nn.Linear(3, 2)
        for parameter in model.parameters():
            parameter.ds_id = 0
            parameter.ds_numel = parameter.numel()

        gather_calls = []

        def gathered_parameters(parameters, **kwargs):
            gather_calls.append((list(parameters), kwargs))
            return contextlib.nullcontext()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = pathlib.Path(tmpdir) / "model.bin"
            with patch("open_instruct.model_utils.deepspeed.zero.GatheredParameters", side_effect=gathered_parameters):
                open_instruct.model_utils.save_model_state_dict(model, checkpoint_path, rank=0)
            state_dict = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

        self.assertEqual(len(gather_calls), len(list(model.parameters())))
        self.assertTrue(all(call_kwargs["enabled"] for _, call_kwargs in gather_calls))
        self.assertEqual(set(state_dict), set(model.state_dict()))
        self.assertTrue(all(tensor.numel() > 0 for tensor in state_dict.values()))

    def test_zero3_load_gathers_full_model_on_rank_zero(self):
        source = torch.nn.Linear(3, 2)
        target = torch.nn.Linear(3, 2)
        for parameter in target.parameters():
            parameter.ds_id = 0

        gather_calls = []

        def gathered_parameters(parameters, **kwargs):
            gather_calls.append((list(parameters), kwargs))
            return contextlib.nullcontext()

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = pathlib.Path(tmpdir) / "model.bin"
            torch.save(source.state_dict(), checkpoint_path)
            with patch("open_instruct.model_utils.deepspeed.zero.GatheredParameters", side_effect=gathered_parameters):
                open_instruct.model_utils.maybe_load_checkpoint(
                    SimpleNamespace(module=target), str(checkpoint_path), torch.device("cpu"), rank=0
                )

        self.assertEqual(len(gather_calls), 1)
        self.assertEqual(gather_calls[0][1]["modifier_rank"], 0)
        for source_parameter, target_parameter in zip(source.parameters(), target.parameters()):
            self.assertTrue(torch.equal(source_parameter, target_parameter))


class TestBatchSlicing(unittest.TestCase):
    def test_batch_slicing_with_all_fields(self):
        batch = Batch(
            queries=[[1, 2], [3, 4], [5, 6]],
            ground_truths=[[7, 8], [9, 10], [11, 12]],
            datasets=["ds1", "ds2", "ds3"],
            raw_queries=["q1", "q2", "q3"],
            decoded_responses=["r1", "r2", "r3"],
            indices=[0, 1, 2],
            scores=[0.1, 0.2, 0.3],
            model_steps=[0, 0, 0],
        )

        sliced = batch[[0, 2]]
        self.assertEqual(len(sliced.queries), 2)
        self.assertEqual(sliced.queries, [[1, 2], [5, 6]])
        self.assertEqual(sliced.decoded_responses, ["r1", "r3"])
        self.assertEqual(sliced.scores, [0.1, 0.3])

    def test_batch_slicing_with_none_fields(self):
        batch = Batch(
            queries=[[1, 2], [3, 4], [5, 6]],
            ground_truths=[[7, 8], [9, 10], [11, 12]],
            datasets=["ds1", "ds2", "ds3"],
            raw_queries=None,
            decoded_responses=None,
            indices=None,
            scores=None,
            model_steps=[0, 0, 0],
        )

        sliced = batch[[0, 2]]
        self.assertEqual(len(sliced.queries), 2)
        self.assertEqual(sliced.queries, [[1, 2], [5, 6]])
        self.assertIsNone(sliced.decoded_responses)
        self.assertIsNone(sliced.scores)


class TestLogSoftmaxAndGather(unittest.TestCase):
    def test_log_softmax_and_gather_sliced_logits(self):
        batch_size, seq_len, vocab_size = 2, 160, 151936
        logits_full = torch.randn(batch_size, seq_len + 1, vocab_size)
        logits = logits_full[:, :-1, :]
        index_full = torch.randint(0, vocab_size, (batch_size, seq_len + 1))
        index = index_full[:, 1:].clone()

        self.assertFalse(logits.is_contiguous())
        self.assertTrue(index.is_contiguous())

        result = open_instruct.model_utils.log_softmax_and_gather(logits, index)

        self.assertEqual(result.shape, (batch_size, seq_len))
        self.assertTrue(torch.all(result <= 0))
        self.assertTrue(torch.all(torch.isfinite(result)))


class TestTensorCache(unittest.TestCase):
    def test_getitem_returns_correct_tensors(self):
        chosen_logps = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        rejected_logps = torch.tensor([[0.5, 1.5], [2.5, 3.5]])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        result = cache[torch.tensor([0])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[1.0, 2.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[0.5, 1.5]])))

        result = cache[torch.tensor([1])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[3.0, 4.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[2.5, 3.5]])))

    def test_getitem_with_multiple_indices(self):
        chosen_logps = torch.tensor([[1.0], [2.0], [3.0]])
        rejected_logps = torch.tensor([[0.5], [1.5], [2.5]])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        result = cache[torch.tensor([0, 2])]
        self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([[1.0], [3.0]])))
        self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([[0.5], [2.5]])))

    def test_to_disk_and_from_disk(self):
        chosen_logps = torch.tensor([1.0, 2.0, 3.0])
        rejected_logps = torch.tensor([0.5, 1.5, 2.5])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)

            self.assertTrue(cache_path.exists())

            loaded_cache = TensorCache.from_disk(cache_path, device="cpu")

            self.assertTrue(torch.allclose(loaded_cache.tensors["chosen_logps"], chosen_logps))
            self.assertTrue(torch.allclose(loaded_cache.tensors["rejected_logps"], rejected_logps))

    def test_from_disk_preserves_indexing(self):
        chosen_logps = torch.tensor([1.0, 2.0, 3.0, 4.0])
        rejected_logps = torch.tensor([0.1, 0.2, 0.3, 0.4])

        cache = TensorCache(tensors={"chosen_logps": chosen_logps, "rejected_logps": rejected_logps})

        with tempfile.TemporaryDirectory() as tmpdir:
            cache_path = pathlib.Path(tmpdir) / "cache.pt"
            cache.to_disk(cache_path)
            loaded_cache = TensorCache.from_disk(cache_path, device="cpu")

            result = loaded_cache[torch.tensor([1, 3])]
            self.assertTrue(torch.allclose(result["chosen_logps"], torch.tensor([2.0, 4.0])))
            self.assertTrue(torch.allclose(result["rejected_logps"], torch.tensor([0.2, 0.4])))


if __name__ == "__main__":
    unittest.main()
