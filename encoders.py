# ── Shared Encoder ────────────────────────────────────────────────────────────

class OrdinalMapEncoder:
    def __init__(self, mapping):
        self.classes_ = list(mapping.keys())
        self._map     = mapping

    def transform(self, vals):
        return [self._map.get(str(v), 0) for v in vals]
