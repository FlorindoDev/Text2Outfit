"""Test local checkpoint reuse and Hub fallback without downloading model weights."""

import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from unittest.mock import call, patch

from huggingface_hub.errors import LocalEntryNotFoundError

from model.llm.download import main, resolve_model_path


class ModelDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.checkpoint = self.root / "checkpoint"
        self.checkpoint.mkdir()
        for name in ("config.json", "tokenizer_config.json", "tokenizer.json"):
            (self.checkpoint / name).write_text("{}", encoding="utf-8")
        (self.checkpoint / "model.safetensors").write_bytes(b"test weights")

    def test_local_directory_is_reused_without_contacting_the_hub(self):
        with patch("model.llm.download.snapshot_download") as download:
            for source in (str(self.checkpoint), self.checkpoint):
                self.assertEqual(resolve_model_path(source), self.checkpoint.resolve())
        download.assert_not_called()

    def test_complete_hub_cache_is_reused_without_network_lookup(self):
        with patch("model.llm.download.snapshot_download", return_value=str(self.checkpoint)) as download:
            self.assertEqual(resolve_model_path("Qwen/Qwen3-Embedding-8B"), self.checkpoint)
        download.assert_called_once_with(
            repo_id="Qwen/Qwen3-Embedding-8B", repo_type="model", cache_dir=None,
            local_files_only=True,
        )

    def test_cache_miss_downloads_from_hugging_face_into_the_selected_cache(self):
        cache = self.root / "cache"
        with patch("model.llm.download.snapshot_download", side_effect=[
            LocalEntryNotFoundError("not cached"), str(self.checkpoint),
        ]) as download:
            self.assertEqual(resolve_model_path("Qwen/Qwen3-Embedding-8B", cache_dir=cache), self.checkpoint)
        arguments = {"repo_id": "Qwen/Qwen3-Embedding-8B", "repo_type": "model", "cache_dir": cache}
        self.assertEqual(download.call_args_list, [
            call(**arguments, local_files_only=True), call(**arguments),
        ])

    def test_partial_sharded_cache_downloads_the_missing_shard(self):
        (self.checkpoint / "model.safetensors").unlink()
        index = {"weight_map": {"layer1": "part1.safetensors", "layer2": "part2.safetensors"}}
        (self.checkpoint / "model.safetensors.index.json").write_text(json.dumps(index), encoding="utf-8")
        (self.checkpoint / "part1.safetensors").write_bytes(b"first shard")

        def snapshot(**kwargs):
            if not kwargs.get("local_files_only"):
                (self.checkpoint / "part2.safetensors").write_bytes(b"second shard")
            return str(self.checkpoint)

        with patch("model.llm.download.snapshot_download", side_effect=snapshot) as download:
            self.assertEqual(resolve_model_path("Qwen/Qwen3-Embedding-8B"), self.checkpoint)
        self.assertEqual(download.call_count, 2)
        with patch("model.llm.download.snapshot_download", return_value=str(self.checkpoint)) as download:
            resolve_model_path("Qwen/Qwen3-Embedding-8B")
        self.assertEqual(download.call_count, 1)

    def test_metadata_only_cache_is_completed_before_loading(self):
        partial = self.root / "partial"
        partial.mkdir()
        (partial / "config.json").write_text("{}", encoding="utf-8")
        with patch("model.llm.download.snapshot_download", side_effect=[str(partial), str(self.checkpoint)]) as download:
            self.assertEqual(resolve_model_path("Qwen/Qwen3-Embedding-8B"), self.checkpoint)
        self.assertEqual(download.call_count, 2)

    def test_incomplete_download_is_rejected(self):
        (self.checkpoint / "model.safetensors").unlink()
        with patch("model.llm.download.snapshot_download", return_value=str(self.checkpoint)):
            with self.assertRaisesRegex(ValueError, "incomplete model checkpoint"):
                resolve_model_path("Qwen/Qwen3-Embedding-8B")

    def test_invalid_local_paths_do_not_trigger_downloads(self):
        with patch("model.llm.download.snapshot_download") as download:
            for source in (self.root / "missing", str(self.root / "missing")):
                with self.subTest(source=source), self.assertRaises(FileNotFoundError):
                    resolve_model_path(source)
            with self.assertRaises(NotADirectoryError):
                resolve_model_path(self.checkpoint / "config.json")
            (self.checkpoint / "model.safetensors").unlink()
            with self.assertRaisesRegex(ValueError, "incomplete model checkpoint"):
                resolve_model_path(self.checkpoint)
            for source in ("", " ", None):
                with self.subTest(source=source), self.assertRaises(ValueError):
                    resolve_model_path(source)
        download.assert_not_called()

    def test_invalid_or_missing_shards_are_rejected(self):
        (self.checkpoint / "model.safetensors").unlink()
        index_path = self.checkpoint / "model.safetensors.index.json"
        for content in ("{", "[]", "{}", '{"weight_map": {}}',
                        '{"weight_map": {"w": "missing.safetensors"}}',
                        '{"weight_map": {"w": []}}',
                        '{"weight_map": {"w": "../outside.safetensors"}}'):
            index_path.write_text(content, encoding="utf-8")
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "incomplete"):
                resolve_model_path(self.checkpoint)

    def test_hub_download_errors_are_preserved(self):
        with patch("model.llm.download.snapshot_download", side_effect=[
            LocalEntryNotFoundError("not cached"), ConnectionError("network unavailable"),
        ]):
            with self.assertRaisesRegex(ConnectionError, "network unavailable"):
                resolve_model_path("Qwen/Qwen3-Embedding-8B")

    def test_cli_prints_the_local_checkpoint_path(self):
        output = io.StringIO()
        with redirect_stdout(output), patch("model.llm.download.snapshot_download") as download:
            main([str(self.checkpoint)])
        self.assertEqual(output.getvalue().strip(), str(self.checkpoint.resolve()))
        download.assert_not_called()


if __name__ == "__main__":
    unittest.main()
