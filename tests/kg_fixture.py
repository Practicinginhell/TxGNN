"""Build a tiny synthetic KG that mimics the real Dataverse kg.csv.

Tests use this instead of the real 8.1M-edge / 945 MB download so the suite runs in
seconds. The schema and the structural invariants are faithful to the real file:

  * the same 12 columns, in the same order
  * every edge stored in BOTH directions, which is what preprocess_kg() relies on
    when it de-duplicates each relation down to one orientation
  * a mix of homogeneous relations (protein_protein, disease_disease) and
    heterogeneous ones (indication, disease_protein, ...)
  * node_index is global across node types, as in the real KG
"""

import pandas as pd

COLUMNS = [
    "relation", "display_relation",
    "x_index", "x_id", "x_type", "x_name", "x_source",
    "y_index", "y_id", "y_type", "y_name", "y_source",
]

DISEASE_BASE, DRUG_BASE, PROTEIN_BASE = 0, 1000, 2000


def _node(index, ntype, name, source):
    return (index, str(index), ntype, name, source)


def build_synthetic_kg(path, n_diseases=40, drugs_per_disease=4, n_proteins=30):
    """Write a synthetic kg.csv to `path` and return the DataFrame."""
    rows = []

    def add_both(relation, display, a, b):
        """Append an edge in both directions, exactly like the real kg.csv."""
        rows.append(dict(zip(COLUMNS, (relation, display) + a + b)))
        rows.append(dict(zip(COLUMNS, (relation, display) + b + a)))

    diseases = [_node(DISEASE_BASE + i, "disease", f"disease_{i}", "MONDO")
                for i in range(n_diseases)]
    drugs = [_node(DRUG_BASE + i, "drug", f"drug_{i}", "DrugBank")
             for i in range(n_diseases * drugs_per_disease)]
    proteins = [_node(PROTEIN_BASE + i, "gene/protein", f"protein_{i}", "NCBI")
                for i in range(n_proteins)]

    # drug-disease edges: every disease gets enough treatments to clear the
    # ">= 3 treatments" eligibility filter in make_mini_kg.py
    for d_i, disease in enumerate(diseases):
        for k in range(drugs_per_disease):
            drug = drugs[d_i * drugs_per_disease + k]
            relation = ["indication", "contraindication", "off-label use"][k % 3]
            add_both(relation, relation, drug, disease)

    # homogeneous relation, name halves match ("protein" == "protein")
    for i in range(len(proteins) - 1):
        add_both("protein_protein", "ppi", proteins[i], proteins[i + 1])

    # homogeneous relation between diseases
    for i in range(len(diseases) - 1):
        add_both("disease_disease", "parent-child", diseases[i], diseases[i + 1])

    # heterogeneous relations linking the two halves of the graph together
    for i, disease in enumerate(diseases):
        add_both("disease_protein", "associated with", disease, proteins[i % n_proteins])
    for i, drug in enumerate(drugs):
        add_both("drug_protein", "target", drug, proteins[i % n_proteins])

    df = pd.DataFrame(rows, columns=COLUMNS)
    df.to_csv(path, index=False)
    return df


def undirected_keys(df):
    """Canonical (unordered endpoints, relation) key for every row."""
    return {
        "_".join(sorted([str(a), str(b)])) + "|" + r
        for a, b, r in zip(df.x_index, df.y_index, df.relation)
    }
