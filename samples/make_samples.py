"""Generates the test images/masks in this folder (procedural, no external assets).
These are INPUTS for testing the interface; they are not inpainting results."""
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

rng = np.random.default_rng(7)

# 1. Landscape: sky gradient, sun, hills, small house + grain
W, H = 640, 480
y = np.linspace(0, 1, H)[:, None, None]
sky = (1 - y) * np.array([235, 200, 150]) + y * np.array([250, 235, 205])
img = Image.fromarray(np.broadcast_to(sky, (H, W, 3)).astype(np.uint8))
d = ImageDraw.Draw(img)
d.ellipse((440, 60, 520, 140), fill=(214, 120, 80))
for base, col, amp, ph in ((300, (150, 160, 110), 40, 0.0), (350, (110, 125, 80), 35, 1.5), (410, (80, 95, 60), 25, 3.0)):
    pts = [(x, base + amp * np.sin(x / 90 + ph)) for x in range(0, W + 1, 8)] + [(W, H), (0, H)]
    d.polygon(pts, fill=col)
d.rectangle((120, 330, 210, 400), fill=(190, 110, 80))
d.polygon([(110, 330), (165, 285), (220, 330)], fill=(120, 60, 45))
d.rectangle((152, 360, 178, 400), fill=(70, 50, 40))
arr = np.array(img.filter(ImageFilter.GaussianBlur(0.8)), dtype=np.float32) + rng.normal(0, 4, (H, W, 1))
Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8)).save("sample1_landscape.png")
m = Image.new("L", (W, H), 0); md = ImageDraw.Draw(m)
md.ellipse((400, 40, 540, 170), fill=255); md.rectangle((135, 345, 200, 395), fill=255); md.ellipse((300, 380, 380, 430), fill=255)
m.save("sample1_landscape_mask.png")

# 2. Textured wall (tiles) with scratches mask - good for patch-based methods
W, H = 512, 512
tile = np.zeros((H, W, 3), np.float32)
for i in range(0, H, 64):
    for j in range(0, W, 64):
        shade = 175 + 30 * ((i // 64 + j // 64) % 2)
        tile[i:i + 64, j:j + 64] = np.array([shade, shade - 25, shade - 55])
        tile[i:i + 3, j:j + 64] *= 0.7; tile[i:i + 64, j:j + 3] *= 0.7
tile += rng.normal(0, 6, (H, W, 1))
Image.fromarray(np.clip(tile, 0, 255).astype(np.uint8)).save("sample2_tiles.png")
m = Image.new("L", (W, H), 0); md = ImageDraw.Draw(m)
md.line((40, 60, 460, 140), fill=255, width=9); md.line((90, 400, 420, 300), fill=255, width=7)
md.line((250, 20, 270, 490), fill=255, width=6); md.ellipse((350, 380, 440, 470), fill=255)
m.save("sample2_tiles_mask.png")
print("done")
