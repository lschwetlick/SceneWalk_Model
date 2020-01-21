"""
SceneWalk variant that has access to limited history at each fixation
"""

class limited_sw():

    def __init__(self, data_range, n_history):
        super().__init__("subtractive", "zero", "both", 1, "on", data_range, {"coupled_oms": True, "coupled_facil": True})
        self.n_history = n_history

    def whoami(self):
        return "Limited History SW: " + super().whoami()