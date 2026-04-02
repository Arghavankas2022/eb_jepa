"""
Utilities shared between data_pairing.py and main.py.
"""
import pandas as pd


def parse_developmental_edges(xlsx_path: str) -> pd.DataFrame:
    """Extract ONLY 'Developmental progression' edges from the metadata Excel file.

    Mirrors the same filter used in data_pairing.py so that valid_transitions
    is consistent with how the training pairs were generated.

    Returns a DataFrame with columns:
        Subsystem, Cell state name (x), Cell state name (y), Edge type
    """
    raw = pd.read_excel(xlsx_path, sheet_name="Table.22", header=None)

    # Find the header row
    header_row = next(
        (i for i in range(20)
         if "Cell state name (x)" in [str(v) for v in raw.iloc[i]]),
        None,
    )
    if header_row is None:
        raise ValueError("Could not find header row in Table.22")

    df = raw.iloc[header_row + 1:].copy()
    df.columns = raw.iloc[header_row].tolist()

    needed = ["Subsystem", "Cell state name (x)", "Cell state name (y)", "Edge type"]
    df = df[needed].dropna(subset=["Cell state name (x)", "Cell state name (y)"]).copy()
    for col in needed:
        df[col] = df[col].astype(str).str.strip()

    dev_mask = df["Edge type"].str.contains("Developmental progression", case=False)
    return df[dev_mask].drop_duplicates(
        subset=["Subsystem", "Cell state name (x)", "Cell state name (y)"]
    ).reset_index(drop=True)
