"""CP assets reuse local files/cache and download only the requested best checkpoint."""

import io
import tempfile
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from huggingface_hub.errors import LocalEntryNotFoundError

from model import DEFAULT_MODEL_CONFIG
from model.compatibility import resolve_cp_assets


class CPDownloadTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "source"
        self.checkpoint = self.root / "best.pt"
        self.checkpoint.write_bytes(b"checkpoint fixture")
        self.config = replace(DEFAULT_MODEL_CONFIG, cp_repo=self.repo, cp_cache=self.root / "hub")

    def local_source(self):
        (self.repo / "model").mkdir(parents=True)
        (self.repo / "model/__init__.py").write_text("# model fixture")

    @staticmethod
    def archive(files):
        output = io.BytesIO()
        with zipfile.ZipFile(output, "w") as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        output.seek(0)
        return output

    def test_local_source_and_checkpoint_require_no_network(self):
        self.local_source()
        with patch("model.compatibility.download.urllib.request.urlopen") as source, patch("model.compatibility.download.hf_hub_download") as weights:
            repo, checkpoint = resolve_cp_assets(replace(self.config, cp_checkpoint=self.checkpoint))
        self.assertEqual((repo, checkpoint), (self.repo.resolve(), self.checkpoint.resolve()))
        source.assert_not_called()
        weights.assert_not_called()

    def test_cached_weights_require_no_online_download(self):
        self.local_source()
        with patch("model.compatibility.download.hf_hub_download", return_value=str(self.checkpoint)) as download:
            _, checkpoint = resolve_cp_assets(self.config)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(download.call_count, 1)
        self.assertTrue(download.call_args.kwargs["local_files_only"])

    def test_siglip_requires_its_own_checkpoint_before_any_download(self):
        with patch("model.compatibility.download.urllib.request.urlopen") as source, patch("model.compatibility.download.hf_hub_download") as weights:
            with self.assertRaisesRegex(ValueError, "SigLIP.*--cp-checkpoint"):
                resolve_cp_assets(replace(self.config, encoder="siglip"))
        source.assert_not_called()
        weights.assert_not_called()

    def test_siglip_accepts_explicit_local_or_remote_checkpoint(self):
        self.local_source()
        config = replace(self.config, encoder="siglip", cp_checkpoint=self.checkpoint)
        with patch("model.compatibility.download.hf_hub_download") as download:
            _, checkpoint = resolve_cp_assets(config)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        download.assert_not_called()
        config = replace(config, cp_checkpoint=None, cp_weights_file="checkpoints/cp_marqo/best.pt")
        with patch("model.compatibility.download.hf_hub_download", return_value=str(self.checkpoint)) as download:
            _, checkpoint = resolve_cp_assets(config)
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(download.call_args.kwargs["filename"], config.cp_weights_file)

    def test_downloads_model_code_and_one_best_checkpoint_only(self):
        archive = self.archive({
            "repo/model/__init__.py": "# package",
            "repo/model/cp/transformer.py": "# CP",
            "repo/training/train.py": "# unused",
            "repo/checkpoints/epoch_001.pt": "unused weights",
        })
        with (
            patch("model.compatibility.download.urllib.request.urlopen", return_value=archive),
            patch("model.compatibility.download.hf_hub_download", side_effect=[LocalEntryNotFoundError("missing"), str(self.checkpoint)]) as download,
        ):
            repo, checkpoint = resolve_cp_assets(self.config)
        self.assertTrue((repo / "model/cp/transformer.py").is_file())
        self.assertFalse((repo / "training").exists())
        self.assertFalse((repo / "checkpoints").exists())
        self.assertEqual(checkpoint, self.checkpoint.resolve())
        self.assertEqual(download.call_count, 2)
        for call in download.call_args_list:
            self.assertEqual(call.kwargs["filename"], self.config.cp_weights_file)
            self.assertTrue(call.kwargs["filename"].endswith("/best.pt"))
        self.assertNotIn("local_files_only", download.call_args.kwargs)

    def test_invalid_local_paths_fail_without_downloading(self):
        with patch("model.compatibility.download.urllib.request.urlopen") as source, patch("model.compatibility.download.hf_hub_download") as weights:
            with self.assertRaises(FileNotFoundError):
                resolve_cp_assets(replace(self.config, cp_checkpoint=self.root / "missing.pt"))
            self.repo.mkdir()
            with self.assertRaisesRegex(ValueError, "no model/__init__.py"):
                resolve_cp_assets(self.config)
        source.assert_not_called()
        weights.assert_not_called()

    def test_invalid_archive_does_not_publish_partial_source(self):
        for files in ({"repo/../outside.py": "bad"}, {"repo/README.md": "no model"}):
            with self.subTest(files=files), patch("model.compatibility.download.urllib.request.urlopen", return_value=self.archive(files)):
                with self.assertRaises(ValueError):
                    resolve_cp_assets(self.config)
                self.assertFalse(self.repo.exists())
                self.assertFalse((self.root / "outside.py").exists())

    def test_remote_weights_must_be_best_checkpoints(self):
        for name in ("../best.pt", "/best.pt", "checkpoints/epoch_001.pt", "checkpoints\\best.pt"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "best.pt"):
                replace(self.config, cp_weights_file=name).validate()


if __name__ == "__main__":
    unittest.main()
