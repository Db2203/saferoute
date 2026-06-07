"""Train + persist the severity model. Run from dubai/backend:
    python -m scripts.train_severity
"""
import pandas as pd

from app.models.severity_model import (
    MODEL_PKL,
    PARQUET,
    REPORT_MD,
    save_artifact,
    train,
    write_report,
)


def main() -> None:
    df = pd.read_parquet(PARQUET)
    artifact, _ = train(df)
    save_artifact(artifact)
    write_report(artifact)
    m = artifact.metrics
    print(f"AUC={m['roc_auc']} severe-recall={m['severe']['recall']}")
    print(f"importances: {artifact.importances}")
    print(f"saved {MODEL_PKL.name} + {REPORT_MD.name}")


if __name__ == "__main__":
    main()
