"""Turn a LightGBM booster into compact JSON that site/predict.js evaluates, plus a Python evaluator of
that same JSON so tests can prove the export is exact."""

from __future__ import annotations

import numpy as np


def compact(booster) -> dict:
    """Each tree becomes parallel arrays. Internal nodes are numbered by split_index; a child that is a
    leaf is stored as -(leaf_index + 1)."""
    dump = booster.dump_model()
    trees = []
    for t in dump["tree_info"]:
        root = t["tree_structure"]
        if "leaf_value" in root:  # a single-leaf tree
            trees.append({"f": [], "t": [], "l": [], "r": [], "v": [root["leaf_value"]]})
            continue
        n_split, n_leaf = t["num_leaves"] - 1, t["num_leaves"]
        f, thr, left, right, val = [0] * n_split, [0.0] * n_split, [0] * n_split, [0] * n_split, [0.0] * n_leaf

        def ref(node):
            return node["split_index"] if "split_index" in node else -(node["leaf_index"] + 1)

        stack = [root]
        while stack:
            node = stack.pop()
            if "leaf_index" in node:
                val[node["leaf_index"]] = node["leaf_value"]
                continue
            # the features never contain NaN and the model has no categorical splits
            assert node["decision_type"] == "<=", node["decision_type"]
            i = node["split_index"]
            f[i], thr[i] = node["split_feature"], node["threshold"]
            left[i], right[i] = ref(node["left_child"]), ref(node["right_child"])
            stack += [node["left_child"], node["right_child"]]
        trees.append({"f": f, "t": thr, "l": left, "r": right, "v": val})
    return {"feature_names": dump["feature_names"], "trees": trees}


def predict(model: dict, X: np.ndarray) -> np.ndarray:
    """Reference evaluator with the exact traversal predict.js uses."""
    out = np.zeros(len(X))
    for row_i, row in enumerate(X):
        total = 0.0
        for tr in model["trees"]:
            if not tr["f"]:
                total += tr["v"][0]
                continue
            node = 0
            while node >= 0:
                node = tr["l"][node] if row[tr["f"][node]] <= tr["t"][node] else tr["r"][node]
            total += tr["v"][-node - 1]
        out[row_i] = total
    return out
