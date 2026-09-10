import pandas as pd


def load_dataset(file_path):
    data = pd.read_csv(
        file_path,
        sep="\t",
        header=None,
        names=["label", "message"]
    )

    return data