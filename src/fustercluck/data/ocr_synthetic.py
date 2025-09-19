"""Synthetic OCR sample generator for documents and UI snippets."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from PIL import Image, ImageDraw, ImageFont

DEFAULT_STRINGS = [
    "Invoice Total",
    "Due Date",
    "Shipping Address",
    "Subtotal",
    "Taxes",
    "Balance",
    "Account ID",
    "Order #",
    "Customer Success",
    "Payment Received",
    "Transaction",
    "Dashboard",
]

DEFAULT_FONTS = [
    "/System/Library/Fonts/SFNS.ttf",
    "/System/Library/Fonts/Supplemental/Arial.ttf",
    "/System/Library/Fonts/Supplemental/Courier New.ttf",
]


@dataclass
class SynthOCRConfig:
    width: int = 512
    height: int = 320
    background: Tuple[int, int, int] = (255, 255, 255)
    font_size_range: Tuple[int, int] = (18, 36)
    noise_level: float = 0.05
    max_lines: int = 8
    fonts: Iterable[Path] | None = None


class SyntheticOCRGenerator:
    def __init__(self, cfg: SynthOCRConfig | None = None) -> None:
        self.cfg = cfg or SynthOCRConfig()
        fonts = self.cfg.fonts or [Path(path) for path in DEFAULT_FONTS if Path(path).exists()]
        if not fonts:
            raise RuntimeError("No fonts found; supply paths via SynthOCRConfig.fonts")
        self.fonts = fonts

    def _sample_font(self) -> ImageFont.FreeTypeFont:
        font_path = random.choice(list(self.fonts))
        size = random.randint(*self.cfg.font_size_range)
        return ImageFont.truetype(str(font_path), size=size)

    def _draw_noise(self, image: Image.Image) -> None:
        pixels = image.load()
        width, height = image.size
        noise_pixels = int(width * height * self.cfg.noise_level)
        for _ in range(noise_pixels):
            x = random.randint(0, width - 1)
            y = random.randint(0, height - 1)
            value = random.randint(200, 255)
            pixels[x, y] = (value, value, value)

    def generate(self, strings: List[str] | None = None) -> Tuple[Image.Image, List[str]]:
        strings = strings or DEFAULT_STRINGS
        bg = Image.new("RGB", (self.cfg.width, self.cfg.height), color=self.cfg.background)
        draw = ImageDraw.Draw(bg)
        y = random.randint(10, 30)
        used_strings = []
        for _ in range(random.randint(3, self.cfg.max_lines)):
            text = random.choice(strings)
            used_strings.append(text)
            font = self._sample_font()
            draw.text((random.randint(20, 40), y), text, fill=(0, 0, 0), font=font)
            y += font.size + random.randint(6, 20)
            if y > self.cfg.height - 40:
                break
        self._draw_noise(bg)
        return bg, used_strings

    def batch(self, count: int) -> List[Tuple[Image.Image, List[str]]]:
        return [self.generate() for _ in range(count)]
