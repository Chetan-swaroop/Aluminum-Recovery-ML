class BatteryModel:

    def __init__(self, efficiency_model, corrosion_model):
        self.efficiency_model = efficiency_model
        self.corrosion_model = corrosion_model

    def predict(self, purity, fe, si, cu, mg, zn):

        features = pd.DataFrame([[
            purity, fe, si, cu, mg, zn
        ]], columns=[
            "Purity", "Fe", "Si", "Cu", "Mg", "Zn"
        ])

        efficiency = self.efficiency_model.predict(
            features
        )[0]

        corrosion = self.corrosion_model.predict(
            features
        )[0]

        return {
            "discharge_efficiency": efficiency,
            "self_corrosion_rate": corrosion
        }