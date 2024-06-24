import json
from typing import List


class ApiKeyLoadBalancer:
    def __init__(self, path: str):
        with open(path, 'r') as file:
            data = json.load(file)
        self.keys: List[str] = data['keys']
        self.next_index: int = 0

    def get_key(self) -> str:
        if not self.keys:
            raise ValueError("No keys defined")

        api_key = self.keys[self.next_index]
        self.next_index = (self.next_index + 1) % len(self.keys)
        return api_key
