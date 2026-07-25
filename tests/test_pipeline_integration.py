"""End-to-end pipeline test. Slow, and skipped unless you opt in.

The fast suite covers the subsetting logic with a synthetic fixture. This one runs the
real TxGNN stack (DGL graph construction, pretrain, finetune, evaluation) against the
mini KG, which takes about a minute on CPU.

Run:  TXGNN_RUN_SLOW=1 python -m unittest discover -s tests -v

Requires data_mini/ to exist (build it with make_mini_kg.py first).
"""

import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_MINI = os.path.join(REPO, "data_mini")

SLOW_ENABLED = os.environ.get("TXGNN_RUN_SLOW") == "1"


@unittest.skipUnless(SLOW_ENABLED, "set TXGNN_RUN_SLOW=1 to run the slow pipeline test")
@unittest.skipUnless(os.path.isdir(DATA_MINI),
                     "data_mini/ not found, build it with make_mini_kg.py")
class TestPipelineEndToEnd(unittest.TestCase):

    def test_pipeline_runs_to_completion(self):
        proc = subprocess.run(
            [sys.executable, os.path.join(REPO, "test_mini_pipeline.py"),
             "--data", DATA_MINI, "--device", "cpu",
             "--n-hid", "32", "--n-inp", "32", "--n-out", "32", "--num-walks", "10",
             "--pretrain-epochs", "1", "--finetune-epochs", "2", "--valid-per-n", "1",
             "--save-dir", os.path.join(REPO, "saved_models", "TxGNN_pytest")],
            capture_output=True, text=True, cwd=REPO)

        self.assertEqual(proc.returncode, 0,
                         f"pipeline failed:\n{proc.stdout[-3000:]}\n{proc.stderr[-3000:]}")
        # every stage must have reported in
        for marker in ("Load KG + build split", "Initialize model", "Pretrain",
                       "Finetune", "Save model", "Disease-centric evaluation", "DONE"):
            self.assertIn(marker, proc.stdout, f"stage missing from output: {marker}")

    def test_checkpoint_is_written(self):
        ckpt = os.path.join(REPO, "saved_models", "TxGNN_pytest")
        if not os.path.isdir(ckpt):
            self.skipTest("run test_pipeline_runs_to_completion first")
        for name in ("model.pt", "config.pkl"):
            self.assertTrue(os.path.exists(os.path.join(ckpt, name)),
                            f"checkpoint is missing {name}")


@unittest.skipUnless(SLOW_ENABLED, "set TXGNN_RUN_SLOW=1 to run the slow pipeline test")
@unittest.skipUnless(os.path.isdir(DATA_MINI),
                     "data_mini/ not found, build it with make_mini_kg.py")
class TestQueryEndToEnd(unittest.TestCase):

    def _run_query(self, *args):
        return subprocess.run(
            [sys.executable, os.path.join(REPO, "query_txgnn.py"),
             "--data", DATA_MINI, *args],
            capture_output=True, text=True, cwd=REPO)

    def test_list_diseases_finds_known_names(self):
        proc = self._run_query("--list-diseases", "asthma")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("asthma", proc.stdout)
        self.assertIn("idx=", proc.stdout)

    def test_query_returns_ranked_drugs(self):
        ckpt = os.path.join(REPO, "model_ckpt_mini")
        if not os.path.isdir(ckpt):
            self.skipTest("no checkpoint, run: python query_txgnn.py --train "
                          "--disease asthma")
        proc = self._run_query("--disease", "asthma", "--topk", "5")
        self.assertEqual(proc.returncode, 0, proc.stderr[-2000:])
        self.assertIn("Top 5 predicted drugs", proc.stdout)
        # the ranked list should actually contain entries
        numbered = [ln for ln in proc.stdout.splitlines()
                    if ln.strip().startswith(("1.", "2.", "3."))]
        self.assertGreaterEqual(len(numbered), 3, "no ranked drugs in output")


if __name__ == "__main__":
    unittest.main()
