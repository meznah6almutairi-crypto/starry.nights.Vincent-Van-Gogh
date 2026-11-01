"""
Starry Night as living colored ASCII, in its own window.

Setup:
    py -3.13 -m pip install numpy pillow
Put the painting image in the same folder as this file and set IMAGE_PATH below.
Run:
    py -3.13 starry_ascii.py
Esc or close the window to quit. (Uses tkinter, which comes with Python.)
"""
import os
import time
import tkinter as tk

import numpy as np
from PIL import Image, ImageDraw, ImageFont, ImageTk

IMAGE_PATH = "starry.webp"      # your painting file (webp/jpg/png all work)
TARGET_WIDTH = 1100             # window width in pixels
FONT_SIZE = 10                  # smaller = more detail (try 8 to 14)
FPS = 30
INTRO_SECONDS = 3.5             # noise -> painting reveal
BG_STRENGTH = 0.75              # how much painting color shows behind the letters (0-1)
CHARS = " .`'^\",:;Il!i~+_-?][}{1)(|/tfjrxnuvczXYUJCLQ0OZmwqpdbkhao*#MW&8%B@$"

FONT_CANDIDATES = ["consola.ttf", "Consolas.ttf", "cour.ttf", "DejaVuSansMono.ttf",
                   "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf"]


def load_font(size):
    for name in FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_glyphs(font):
    """Render each character once; sort them from lightest to densest."""
    cw = max(int(round(font.getlength("M"))), 4)
    asc, desc = font.getmetrics()
    ch = asc + desc
    masks = []
    for c in CHARS:
        im = Image.new("L", (cw, ch), 0)
        ImageDraw.Draw(im).text((0, 0), c, fill=255, font=font)
        masks.append(np.asarray(im, dtype=np.float32) / 255.0)
    masks = np.stack(masks)
    cov = masks.mean(axis=(1, 2))
    order = np.argsort(cov)
    masks, cov = masks[order], cov[order]
    cov = cov / cov.max()
    return masks, cov, cw, ch


def prepare_image(path, cols, rows):
    im = Image.open(path).convert("RGB").resize((cols, rows), Image.LANCZOS)
    rgb = np.asarray(im, dtype=np.float32) / 255.0
    # a little extra punch so colors survive being drawn with thin glyphs
    mean = rgb.mean(axis=2, keepdims=True)
    rgb = np.clip(mean + (rgb - mean) * 1.25, 0, 1)
    lum = rgb @ np.array([0.30, 0.59, 0.11], dtype=np.float32)
    lo, hi = np.percentile(lum, [2, 98])           # stretch contrast so glyphs vary
    lum = np.clip((lum - lo) / (hi - lo + 1e-6), 0, 1)
    return rgb, lum


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    path = IMAGE_PATH if os.path.isabs(IMAGE_PATH) else os.path.join(here, IMAGE_PATH)
    if not os.path.exists(path):
        raise SystemExit(f"Can't find the image: {path}\nPut it next to this script and "
                         f"set IMAGE_PATH at the top.")

    font = load_font(FONT_SIZE)
    masks, cov, cw, ch = build_glyphs(font)

    src = Image.open(path)
    aspect = src.width / src.height
    cols = TARGET_WIDTH // cw
    rows = int(round(cols * cw / aspect / ch))
    rgb, lum = prepare_image(path, cols, rows)

    rng = np.random.default_rng()
    yy, xx = np.mgrid[0:rows, 0:cols].astype(np.float32)
    reveal_order = rng.random((rows, cols)).astype(np.float32)   # when each cell resolves
    start = time.perf_counter()

    root = tk.Tk()
    root.title("Starry Night")
    root.configure(bg="black")
    root.resizable(False, False)
    label = tk.Label(root, bd=0, bg="black")
    label.pack()
    root.bind("<Escape>", lambda e: root.destroy())

    state = {"photo": None}

    def frame():
        t = time.perf_counter() - start

        # living shimmer: slow waves of light drifting across the painting
        wave = (0.07 * np.sin(0.18 * xx + 0.12 * yy - 2.2 * t)
                + 0.05 * np.sin(0.11 * xx - 0.21 * yy + 1.6 * t))
        flicker = rng.normal(0, 0.03, (rows, cols)).astype(np.float32)
        l = np.clip(lum + wave + flicker, 0, 1)

        # background: the painting's true color; glyph: same hue, brighter, ink = brightness
        shade = np.clip(1 + wave * 2.2, 0.6, 1.5)[..., None]
        bg = np.clip(rgb * BG_STRENGTH * shade, 0, 1)
        fg = np.clip(rgb * (1.0 + 0.9 * l[..., None]) * shade, 0, 1)
        idx = np.clip(np.searchsorted(cov, 0.10 + 0.55 * l), 0, len(cov) - 1)
        bg, fg = bg * 255, fg * 255

        # intro: random characters resolve into the painting
        if t < INTRO_SECONDS:
            p = t / INTRO_SECONDS
            unresolved = reveal_order > p ** 1.4
            noise_idx = rng.integers(len(CHARS) // 4, len(CHARS), size=(rows, cols))
            idx = np.where(unresolved, noise_idx, idx)
            bg = np.where(unresolved[..., None], 0, bg)
            fg = np.where(unresolved[..., None], fg * 0.45, fg)

        g = masks[idx][..., None]                                # (rows, cols, ch, cw, 1)
        img = bg[:, :, None, None, :] * (1 - g) + fg[:, :, None, None, :] * g
        img = img.transpose(0, 2, 1, 3, 4).reshape(rows * ch, cols * cw, 3)
        state["photo"] = ImageTk.PhotoImage(Image.fromarray(img.astype(np.uint8)))
        label.configure(image=state["photo"])
        root.after(max(1, int(1000 / FPS)), frame)

    frame()
    root.mainloop()


if __name__ == "__main__":
    main()
