"""Tests for make_mini_kg.py.

These lock down the structural invariants the TxGNN pipeline depends on. Several of
them are regression tests for bugs that actually occurred, and each such test says so.

Run:  python -m unittest discover -s tests -v
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kg_fixture import build_synthetic_kg, undirected_keys, COLUMNS  # noqa: E402

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SCRIPT = os.path.join(REPO, "make_mini_kg.py")
DD_RELS = ["indication", "contraindication", "off-label use"]


class MiniKGTestBase(unittest.TestCase):
    """Builds a synthetic KG once, runs make_mini_kg.py on it once."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="txgnn_test_")
        cls.src = os.path.join(cls.tmp, "kg_full.csv")
        cls.out = os.path.join(cls.tmp, "mini")
        cls.full = build_synthetic_kg(cls.src, n_diseases=40, drugs_per_disease=4)

        cls.proc = subprocess.run(
            [sys.executable, SCRIPT, "--src", cls.src, "--out", cls.out,
             "--n-diseases", "30", "--max-ppi", "40", "--max-per-rel", "40",
             "--seed", "42"],
            capture_output=True, text=True,
        )
        if cls.proc.returncode != 0:
            raise AssertionError(
                f"make_mini_kg.py failed:\nSTDOUT:\n{cls.proc.stdout}\n"
                f"STDERR:\n{cls.proc.stderr}")
        cls.kg = pd.read_csv(os.path.join(cls.out, "kg.csv"), low_memory=False)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)


class TestOutputFiles(MiniKGTestBase):

    def test_writes_all_three_files(self):
        for name in ("kg.csv", "node.csv", "edges.csv"):
            path = os.path.join(self.out, name)
            self.assertTrue(os.path.exists(path), f"{name} was not written")
            self.assertGreater(os.path.getsize(path), 0, f"{name} is empty")

    def test_kg_csv_keeps_the_full_schema(self):
        # preprocess_kg() selects specific columns by name, so the schema must survive
        self.assertEqual(list(self.kg.columns), COLUMNS)

    def test_node_csv_is_tab_separated(self):
        """Regression: node.csv was written comma-separated.

        DataSplitter reads it with sep='\\t' (txgnn/data_splits/datasplit.py), so a
        comma-separated file collapses into a single column and every
        `node_source == "MONDO"` query raises UndefinedVariableError. This only shows
        up on the disease-area splits, so it hid for a long time.
        """
        nodes = pd.read_csv(os.path.join(self.out, "node.csv"), sep="\t",
                            low_memory=False)
        self.assertEqual(
            list(nodes.columns),
            ["node_index", "node_id", "node_type", "node_name", "node_source"])
        # the query DataSplitter actually runs must not raise
        nodes.query('node_source == "MONDO"')

    def test_edges_csv_is_comma_separated_with_expected_columns(self):
        # DataSplitter reads edges.csv with the default comma separator and pulls
        # x_index / y_index out of it
        edges = pd.read_csv(os.path.join(self.out, "edges.csv"), low_memory=False)
        self.assertEqual(list(edges.columns),
                         ["relation", "display_relation", "x_index", "y_index"])

    def test_node_csv_covers_every_node_in_kg(self):
        nodes = pd.read_csv(os.path.join(self.out, "node.csv"), sep="\t",
                            low_memory=False)
        in_kg = set(self.kg.x_index) | set(self.kg.y_index)
        self.assertTrue(
            in_kg.issubset(set(nodes.node_index)),
            "node.csv is missing nodes that appear in kg.csv")


class TestGraphInvariants(MiniKGTestBase):

    def test_every_edge_kept_in_both_directions(self):
        """The invariant preprocess_kg() depends on.

        It de-duplicates each relation to one orientation via
        `d_off[d_off.x_type == d_off.x_type.iloc[0]]`. If only one direction of an
        edge survives subsetting, that relation silently loses edges.
        """
        for relation, group in self.kg.groupby("relation"):
            pairs = {}
            for a, b in zip(group.x_index, group.y_index):
                pairs.setdefault(tuple(sorted((a, b))), set()).add((a, b))
            missing = [p for p, seen in pairs.items() if len(seen) != 2]
            self.assertEqual(
                missing, [],
                f"relation {relation!r} has {len(missing)} edge(s) present in only "
                f"one direction, e.g. {missing[:3]}")

    def test_drug_disease_edges_are_never_downsampled(self):
        """Drug-disease edges are the prediction task, so caps must not touch them."""
        kept_nodes = set(self.kg.x_index) | set(self.kg.y_index)
        for relation in DD_RELS:
            expected = self.full[
                (self.full.relation == relation)
                & self.full.x_index.isin(kept_nodes)
                & self.full.y_index.isin(kept_nodes)
            ]
            actual = self.kg[self.kg.relation == relation]
            self.assertEqual(
                len(actual), len(expected),
                f"{relation} was downsampled: kept {len(actual)} of {len(expected)}")

    def test_subset_is_a_subset(self):
        """Every output edge must exist in the source, no invented edges."""
        self.assertTrue(
            undirected_keys(self.kg).issubset(undirected_keys(self.full)))

    def test_enough_treated_diseases_for_a_nonempty_test_split(self):
        """complex_disease_fold() gives the test bucket whatever np.split leaves.

        The second cut is at int((frac[0] + frac[1]) * n), so the bucket holds
        n - int(0.95 * n), not int(0.05 * n). The two disagree whenever 0.95 * n
        has a fractional part: at n = 1957 the real bucket is 98, not 97.
        Evaluation has nothing to score if the bucket comes out empty.
        """
        dd = self.kg[self.kg.relation.isin(DD_RELS)]
        n_diseases = dd[dd.y_type == "disease"].y_index.nunique()
        n_test = n_diseases - int((0.83125 + 0.11875) * n_diseases)
        self.assertGreaterEqual(
            n_test, 1,
            f"only {n_diseases} treated diseases, the test split would be empty")

    def test_all_node_types_survive(self):
        types = set(self.kg.x_type) | set(self.kg.y_type)
        for expected in ("disease", "drug", "gene/protein"):
            self.assertIn(expected, types)

    def test_relation_caps_are_respected_within_mirroring_factor(self):
        """Caps apply before mirroring, so the final count lands at about 2x.

        This guards against a cap being ignored outright.
        """
        cap = 40
        for relation, group in self.kg.groupby("relation"):
            if relation in DD_RELS:
                continue  # intentionally uncapped
            self.assertLessEqual(
                len(group), cap * 2,
                f"{relation} has {len(group)} edges, more than 2x the cap of {cap}")


class TestDeterminism(unittest.TestCase):

    def test_same_seed_gives_identical_output(self):
        tmp = tempfile.mkdtemp(prefix="txgnn_seed_")
        try:
            src = os.path.join(tmp, "kg_full.csv")
            build_synthetic_kg(src, n_diseases=30, drugs_per_disease=4)
            digests = []
            for run in ("a", "b"):
                out = os.path.join(tmp, run)
                proc = subprocess.run(
                    [sys.executable, SCRIPT, "--src", src, "--out", out,
                     "--n-diseases", "20", "--max-ppi", "40", "--max-per-rel", "40",
                     "--seed", "7"],
                    capture_output=True, text=True)
                self.assertEqual(proc.returncode, 0, proc.stderr)
                with open(os.path.join(out, "kg.csv"), "rb") as f:
                    digests.append(f.read())
            self.assertEqual(digests[0], digests[1],
                             "same --seed produced different output")
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    unittest.main()
