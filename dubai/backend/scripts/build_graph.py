"""Build + cache the Dubai road graph. Run from dubai/backend:
    python -m scripts.build_graph [--rebuild]
"""
import argparse

from app.models.graph import build_dubai_graph


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true", help="force rebuild even if cached")
    args = ap.parse_args()
    g = build_dubai_graph(force=args.rebuild)
    print(f"graph: {g.number_of_nodes():,} nodes / {g.number_of_edges():,} edges")


if __name__ == "__main__":
    main()
