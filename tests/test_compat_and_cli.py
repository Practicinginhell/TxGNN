"""Environment and command-line contract tests.

Fast, no data required. These guard the things that are easy to break by accident
when editing: Python 3.8 compatibility, requirements files that setup.py can parse,
and CLI flags that other docs and scripts depend on.

Run:  python -m unittest discover -s tests -v
"""

import ast
import os
import subprocess
import sys
import unittest

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPTS = ["make_mini_kg.py", "query_txgnn.py", "test_mini_pipeline.py"]

# str methods added after 3.8. DGL 0.5.2 has no wheels above Python 3.8, so the whole
# project is pinned there and these must not appear.
PY39_STR_METHODS = {"removeprefix", "removesuffix"}


class TestPython38Compatibility(unittest.TestCase):

    def test_scripts_parse_under_this_interpreter(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                path = os.path.join(REPO, name)
                with open(path, encoding="utf-8") as f:
                    ast.parse(f.read(), filename=path)

    def test_no_python_39_only_string_methods(self):
        """Regression: str.removeprefix() was used and crashes on Python 3.8.

        It sat at the end of make_mini_kg.py, after kg.csv had been written and the
        summary printed, so the run looked successful while node.csv and edges.csv
        were silently never created.
        """
        for name in SCRIPTS:
            with self.subTest(script=name):
                path = os.path.join(REPO, name)
                with open(path, encoding="utf-8") as f:
                    tree = ast.parse(f.read(), filename=path)
                for node in ast.walk(tree):
                    if isinstance(node, ast.Attribute):
                        self.assertNotIn(
                            node.attr, PY39_STR_METHODS,
                            f"{name} line {node.lineno} uses .{node.attr}(), "
                            f"which needs Python 3.9 but this project targets 3.8")

    def test_interpreter_is_python_38(self):
        """Not a hard failure elsewhere, but worth surfacing when it drifts."""
        if sys.version_info[:2] != (3, 8):
            self.skipTest(
                f"running on {sys.version_info.major}.{sys.version_info.minor}, "
                f"the project targets 3.8")


class TestRequirementsFiles(unittest.TestCase):

    def _lines(self, filename):
        with open(os.path.join(REPO, filename), encoding="utf-8") as f:
            return f.read().splitlines()

    def test_setup_py_can_parse_requirements(self):
        """setup.py feeds requirements.txt straight into install_requires.

        Blank lines and comments are tolerated by the parser, but a malformed pin
        would break `pip install .` for everyone.
        """
        from pkg_resources import parse_requirements
        lines = self._lines("requirements.txt")
        list(parse_requirements("\n".join(lines)))

    def test_core_pins_are_exact(self):
        """Loose pins let pandas 2.x in, which removes DataFrame.append and breaks
        txgnn/utils.py."""
        lines = [ln.strip() for ln in self._lines("requirements.txt")]
        pins = [ln for ln in lines if ln and not ln.startswith("#")]
        self.assertTrue(pins, "requirements.txt has no requirements")
        for pin in pins:
            self.assertIn("==", pin, f"{pin!r} is not pinned to an exact version")

    def test_mac_cpu_requirements_include_torch_and_dgl(self):
        """Regression: requirements.txt alone cannot import txgnn.

        torch and dgl are deliberately absent there (the Colab path needs CUDA
        wheels from a custom index), so the macOS file must carry them or a fresh
        environment fails with ModuleNotFoundError: No module named 'dgl'.
        """
        text = "\n".join(self._lines("requirements-mac-cpu.txt"))
        pins = [ln.strip() for ln in text.splitlines()
                if ln.strip() and not ln.strip().startswith("#")]
        joined = " ".join(pins)
        self.assertIn("torch==", joined)
        self.assertIn("dgl==", joined)

    def test_pyg_lines_carry_their_wheel_index(self):
        """PyPI only ships source tarballs for torch-scatter / torch-sparse at these
        versions, so without the -f index pip tries to compile from source."""
        lines = [ln.strip() for ln in self._lines("requirements-mac-cpu.txt")]
        active = [ln for ln in lines if ln and not ln.startswith("#")]
        needs_index = any(p.startswith(("torch-scatter", "torch-sparse"))
                          for p in active)
        if needs_index:
            self.assertTrue(
                any(ln.startswith("-f ") or ln.startswith("--find-links")
                    for ln in active),
                "torch-scatter/torch-sparse are active but no -f wheel index is set")


class TestCommandLineInterfaces(unittest.TestCase):

    def _run(self, script, *args):
        return subprocess.run([sys.executable, os.path.join(REPO, script), *args],
                              capture_output=True, text=True)

    def test_help_works_for_every_script(self):
        for name in SCRIPTS:
            with self.subTest(script=name):
                proc = self._run(name, "--help")
                self.assertEqual(proc.returncode, 0,
                                 f"{name} --help failed:\n{proc.stderr}")
                self.assertIn("usage", proc.stdout.lower())

    def test_documented_flags_exist(self):
        """The README documents these, so removing one silently breaks the docs."""
        expected = {
            "make_mini_kg.py": ["--src", "--out", "--n-diseases", "--max-ppi",
                                "--max-per-rel", "--seed"],
            "query_txgnn.py": ["--disease", "--disease-idx", "--relation", "--topk",
                               "--train", "--list-diseases"],
            "test_mini_pipeline.py": ["--data", "--device", "--n-hid", "--num-walks",
                                      "--pretrain-epochs", "--finetune-epochs",
                                      "--valid-per-n", "--split"],
        }
        for script, flags in expected.items():
            help_text = self._run(script, "--help").stdout
            for flag in flags:
                with self.subTest(script=script, flag=flag):
                    self.assertIn(flag, help_text,
                                  f"{script} no longer accepts {flag}")

    def test_query_without_a_target_fails_fast(self):
        """Regression: it used to load the whole KG and then die with
        AttributeError: 'NoneType' object has no attribute 'lower'."""
        proc = self._run("query_txgnn.py")
        self.assertNotEqual(proc.returncode, 0)
        combined = (proc.stdout + proc.stderr).lower()
        self.assertIn("nothing to query", combined)
        self.assertNotIn("traceback", combined)

    def test_query_rejects_an_unknown_relation(self):
        proc = self._run("query_txgnn.py", "--disease", "asthma",
                         "--relation", "not-a-relation")
        self.assertNotEqual(proc.returncode, 0)
        self.assertIn("invalid choice", (proc.stdout + proc.stderr).lower())


if __name__ == "__main__":
    unittest.main()
