from typing import Dict, Iterable, List

from OCR.image_gen.model.Language import Language
from OCR.image_gen.model.Text import Character, Text


class Vocabulary:
    def __init__(self, languages: Dict[int, Language], special_tokens: Iterable[str] = None,
                 noise: Iterable[str] = None):
        if special_tokens is None:
            special_tokens = ['_', '°']
        if noise is None:
            noise = []

        self.languages = languages
        self.special_tokens = special_tokens

        self.blank_idx = 0
        self.noise_idx = 1

        self.noise = set(noise)

        self.chars = [*special_tokens]
        self.noisy_chars = [*special_tokens]
        self.offsets = [len(special_tokens)]
        for idx in range(len(languages)):
            lang = languages[idx]
            assert idx == lang.id

            chars = lang.chars
            self.noisy_chars.extend(chars)

            if self.noise is not None:
                chars = [char for char in lang.chars if char not in self.noise]

            self.chars.extend(chars)
            self.offsets.append(self.offsets[idx] + len(chars))

    def encode_char(self, char: Character):
        if char.to_string() in self.noise:
            return self.noise_idx
        return self.offsets[char.language.id] + char.char_idx

    def encode_text(self, text: Text):
        return [self.encode_char(char) for char in text.chars]

    def decode_char(self, char: int) -> str:
        return self.chars[char]

    def decode_text(self, text: Iterable[int]) -> List[str]:
        return [self.decode_char(char) for char in text if char != self.blank_idx]
