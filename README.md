# KeySI

KeySI is an interactive system for keyword-guided semantic inspection, clustering, triplet/prototype training, and document-level refinement over article text.

This repository contains the code and data for the accepted KeySI paper.

## Files

```text
.
├── 20news_6class_cleaned.csv
├── Keysi_demo_video.mp4
├── pyproject.toml
├── requirements.txt
├── scripts/check_environment.py
└── src/keysi_app/
    ├── __init__.py
    ├── __main__.py
    ├── application.py
    ├── core.py
    ├── training.py
    ├── ui.py
    └── callbacks/
        ├── exploration.py
        ├── refinement.py
        └── training_view.py
```

## Code structure

- `core.py`: shared configuration, data loading, preprocessing, and encoder definitions.
- `training.py`: retrieval, persistence, triplet/prototype training, and evaluation.
- `ui.py`: Dash application instance and page layout.
- `callbacks/exploration.py`: keyword grouping, exploration, and initial training interactions.
- `callbacks/training_view.py`: trained-result inspection interactions.
- `callbacks/refinement.py`: document reassignment and refinement training interactions.
- `application.py`: application assembly and callback registration.

## Data

KeySI uses two datasets:

- `20news_6class_cleaned.csv`
- `CSV/risk_factors.csv`

Both CSV files use the same format:

```text
label,text
```

## Run

```powershell
python -m pip install -r requirements.txt
python scripts/check_environment.py
$env:PYTHONPATH="src"
python -m keysi_app
```

The app runs at:

```text
http://127.0.0.1:47983
```

## Workflow

1. Choose the number of groups.
2. Click **Generate Groups**.
3. Select a group.
4. Click keywords in **Keywords 2D Visualization** to assign them.
5. Use **Exclude** for irrelevant keywords.
6. Click **Train Model**.
7. Inspect Training View.
8. Use Refinement Mode for document-level corrections.
9. Click **Run Refinement Training**.

Generated outputs are written to `KeySI_results/`.
