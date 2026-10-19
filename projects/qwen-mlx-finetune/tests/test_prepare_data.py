import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[1]
PREPARE_SCRIPT = PROJECT_DIR / "prepare_data.py"


class PrepareDataTests(unittest.TestCase):
    def run_prepare(self, source, output, *options):
        return subprocess.run(
            [
                sys.executable,
                str(PREPARE_SCRIPT),
                "--input",
                str(source),
                "--output-dir",
                str(output),
                *options,
            ],
            capture_output=True,
            text=True,
            check=False,
        )

    def test_valid_messages_are_split_deterministically(self):
        samples = [
            {
                "messages": [
                    {"role": "user", "content": f"input {i}"},
                    {"role": "assistant", "content": f"answer {i}"},
                ]
            }
            for i in range(10)
        ]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "incoming.jsonl"
            source.write_text(
                "".join(json.dumps(sample) + "\n" for sample in samples),
                encoding="utf-8",
            )
            output = root / "processed"

            first = self.run_prepare(source, output)
            first_train = (output / "train.jsonl").read_text(encoding="utf-8")
            first_valid = (output / "valid.jsonl").read_text(encoding="utf-8")
            second = self.run_prepare(source, output)

            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(first_train, (output / "train.jsonl").read_text())
            self.assertEqual(first_valid, (output / "valid.jsonl").read_text())
            self.assertEqual(len(first_train.splitlines()), 9)
            self.assertEqual(len(first_valid.splitlines()), 1)
            self.assertEqual(
                {json.loads(row)["messages"][0]["content"] for row in first_train.splitlines()}
                | {json.loads(row)["messages"][0]["content"] for row in first_valid.splitlines()},
                {f"input {i}" for i in range(10)},
            )

    def test_single_sample_does_not_create_empty_validation_file(self):
        sample = {
            "messages": [
                {"role": "user", "content": "input"},
                {"role": "assistant", "content": "answer"},
            ]
        }
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "incoming.jsonl"
            source.write_text(json.dumps(sample) + "\n", encoding="utf-8")
            output = root / "processed"

            result = self.run_prepare(source, output)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue((output / "train.jsonl").is_file())
            self.assertFalse((output / "valid.jsonl").exists())

    def test_invalid_message_shape_reports_line(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root / "incoming.jsonl"
            source.write_text(
                '{"messages":[{"role":"tool","content":"not supported"}]}\n',
                encoding="utf-8",
            )

            result = self.run_prepare(source, root / "processed")

            self.assertNotEqual(result.returncode, 0)
            self.assertIn("line 1", result.stderr)
            self.assertIn("unsupported role", result.stderr)


if __name__ == "__main__":
    unittest.main()
