"""python -m tripdur [train|site|all]"""

from __future__ import annotations

import json
import sys

from tripdur import site_data, train

STAGES = {"train": train.run, "site": site_data.run}


def main(argv: list[str]) -> None:
    names = list(STAGES) if argv[:1] in ([], ["all"]) else argv
    for name in names:
        result = STAGES[name]()
        print(f"[{name}]", json.dumps(result, indent=2)[:1500])


if __name__ == "__main__":
    main(sys.argv[1:])
