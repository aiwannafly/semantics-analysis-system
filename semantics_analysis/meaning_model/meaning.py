from typing import Self


class Meaning:
    token: str
    values: list[float]

    def __init__(self, token: str, values: list[float]):
        self.token = token
        self.values = values

    def calculate_similarity(self, other: Self) -> float:
        res = 0
        for a, b in zip(self.values, other.values):
            res += a * b

        return res
