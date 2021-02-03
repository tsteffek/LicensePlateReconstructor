from typing import Iterable, List, Any

from . import Character, Text


def distinct(arr: Iterable[Any]):
    return list(dict.fromkeys(arr))


class Vocabulary:
    def __init__(self, chars: Iterable[str], noise=None, special_tokens: Iterable[str] = None):
        if noise is None:
            noise = []
        if special_tokens is None:
            special_tokens = ['_', '°']

        self.special_tokens = special_tokens
        self.noise = set(noise)

        self.chars = distinct(special_tokens + [char for char in chars if char not in self.noise])
        self.noisy_chars = self.chars + noise

        self.blank_idx = 0
        self.noise_idx = 1

        self.str_lookup = {char: idx for idx, char in enumerate(self.chars)}

    def __len__(self):
        return len(self.chars)

    def encode_str(self, string: str) -> int:
        if string in self.noise:
            return self.noise_idx
        return self.str_lookup[string]

    def encode_strings(self, strings: List[str]) -> List[int]:
        return [self.encode_str(s) for s in strings]

    def encode_char(self, char: Character):
        return self.encode_str(str(char))

    def encode_text(self, text: Text) -> List[int]:
        return [self.encode_char(char) for char in text.chars]

    def decode_char(self, char: int) -> str:
        return self.chars[char]

    def decode_text(self, text: Iterable[int]) -> List[str]:
        return [self.decode_char(char) for char in text if char != self.blank_idx]
