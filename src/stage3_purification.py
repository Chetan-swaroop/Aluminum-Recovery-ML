class PurificationModel:

    def __init__(self, models):
        self.models = models

    def predict(self, fe, si, cu):

        routes = [
            "Remelting",
            "Fluxing",
            "Electrorefining"
        ]

        predictions = {}

        for route in routes:

            features = pd.DataFrame(
                [[fe, si, cu] + [
                    int(route == r) for r in routes
                ]],
                columns=[
                    "Fe", "Si", "Cu",
                    "Route_Remelting",
                    "Route_Fluxing",
                    "Route_Electrorefining"
                ]
            )

            purity = self.models["Purity"].predict(features)[0]
            cost = self.models["Cost"].predict(features)[0]
            energy = self.models["Energy"].predict(features)[0]

            predictions[route] = {
                "purity": purity,
                "cost": cost,
                "energy": energy
            }

        return predictions