import pandas as pd


class CompositionProfiler:

    def __init__(self, csv_path):
        self.data = pd.read_csv(csv_path)

    def estimate(self, material):

        records = self.data[
            self.data["Material"].str.lower()
            == material.lower()
        ]

        if records.empty:
            raise ValueError("Material not found in database")

        profile = {}

        for _, row in records.iterrows():

            element = row["Element"]

            minimum = row["Min_wt"]
            maximum = row["Max_wt"]

            profile[element] = {
                "minimum": minimum,
                "maximum": maximum,
                "estimated": (minimum + maximum) / 2
            }

        return profile