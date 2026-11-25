"""Generate app icon assets for Windows (.ico) and macOS (.icns).

Source: a 1024x1024 PNG drawn from scratch with Pillow (radial gradient
indigo -> cyan, rounded square mask, white microphone glyph, 'BT' wordmark).
The .icns is built by handing a .iconset folder to macOS `iconutil` - on
non-macOS hosts we still write the .iconset so the Mac build script can
iconutil-convert it.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parent
ASSETS = ROOT / "assets"
ASSETS.mkdir(exist_ok=True)

# Indigo -> cyan, matching the in-app accent gradient.
GRADIENT_INNER = (99, 102, 241, 255)   # #6366F1
GRADIENT_OUTER = (6, 182, 212, 255)    # #06B6D4
SURFACE_DARK = (10, 14, 39, 255)       # #0A0E27

SIZE = 1024
CORNER_RADIUS = 224                    # ~22% - modern app-icon look


def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def lerp_rgba(c1, c2, t: float):
    return tuple(lerp(c1[i], c2[i], t) for i in range(4))


def draw_gradient(size: int) -> Image.Image:
    """Top-left to bottom-right linear gradient indigo -> cyan."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    px = img.load()
    for y in range(size):
        for x in range(size):
            # diagonal distance 0..1
            t = (x + y) / (2 * size)
            px[x, y] = lerp_rgba(GRADIENT_INNER, GRADIENT_OUTER, t)
    return img


def rounded_mask(size: int, radius: int) -> Image.Image:
    """Solid white rounded-rect on transparent - used as alpha mask."""
    mask = Image.new("L", (size, size), 0)
    d = ImageDraw.Draw(mask)
    d.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def draw_microphone(draw: ImageDraw.ImageDraw, cx: int, cy: int, scale: float = 1.0):
    """Stylized microphone - rounded capsule + arc stand + base line."""
    s = scale
    # Capsule body
    cap_w = int(180 * s)
    cap_h = int(280 * s)
    cap_top = cy - int(170 * s)
    cap_bottom = cap_top + cap_h
    capsule_box = (cx - cap_w // 2, cap_top, cx + cap_w // 2, cap_bottom)
    draw.rounded_rectangle(capsule_box, radius=cap_w // 2, fill=(255, 255, 255, 255))

    # U-shaped arc (stand) around the lower half of the capsule
    arc_pad = int(40 * s)
    arc_box = (capsule_box[0] - arc_pad, capsule_box[1] + int(20 * s),
               capsule_box[2] + arc_pad, capsule_box[3] + int(120 * s))
    arc_w = int(22 * s)
    draw.arc(arc_box, start=20, end=160, fill=(255, 255, 255, 255), width=arc_w)

    # Vertical stem
    stem_top = arc_box[3] - int(10 * s)
    stem_bottom = stem_top + int(70 * s)
    draw.line((cx, stem_top, cx, stem_bottom), fill=(255, 255, 255, 255), width=arc_w)

    # Horizontal base
    base_half = int(90 * s)
    draw.line((cx - base_half, stem_bottom, cx + base_half, stem_bottom),
              fill=(255, 255, 255, 255), width=arc_w)


def draw_bt_wordmark(draw: ImageDraw.ImageDraw, img_size: int):
    """Small 'BT' monogram bottom-right."""
    # Use a default PIL font - we don't bundle Inter into the icon (PNG only).
    try:
        font = ImageFont.truetype("arialbd.ttf", 110)
    except OSError:
        font = ImageFont.load_default()
    text = "BT"
    # Right-align with padding
    pad = 80
    bbox = draw.textbbox((0, 0), text, font=font)
    tw, th = bbox[2] - bbox[0], bbox[3] - bbox[1]
    x = img_size - tw - pad
    y = img_size - th - pad
    # Soft shadow
    draw.text((x + 3, y + 3), text, font=font, fill=(0, 0, 0, 80))
    draw.text((x, y), text, font=font, fill=(255, 255, 255, 230))


def render_source() -> Image.Image:
    """Build the 1024x1024 RGBA source icon."""
    base = draw_gradient(SIZE)
    mask = rounded_mask(SIZE, CORNER_RADIUS)
    base.putalpha(mask)

    # Subtle inner vignette for depth
    vignette = Image.new("RGBA", (SIZE, SIZE), (0, 0, 0, 0))
    vd = ImageDraw.Draw(vignette)
    vd.ellipse((-SIZE // 2, -SIZE // 2, SIZE + SIZE // 2, SIZE + SIZE // 2),
               fill=(255, 255, 255, 30))
    vignette = vignette.filter(ImageFilter.GaussianBlur(80))
    base = Image.alpha_composite(base, vignette)

    draw = ImageDraw.Draw(base)
    # Microphone centered, slightly above center
    draw_microphone(draw, cx=SIZE // 2, cy=int(SIZE * 0.46), scale=1.4)
    # Wordmark
    draw_bt_wordmark(draw, SIZE)
    return base


def write_png(img: Image.Image) -> Path:
    out = ASSETS / "icon.png"
    img.save(out, format="PNG", optimize=True)
    print(f"[ok] wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


def write_ico(img: Image.Image) -> Path:
    out = ASSETS / "app.ico"
    sizes = [(16, 16), (24, 24), (32, 32), (48, 48), (64, 64),
             (128, 128), (256, 256)]
    img.save(out, format="ICO", sizes=sizes)
    print(f"[ok] wrote {out} ({out.stat().st_size // 1024} KB)")
    return out


def write_icns(img: Image.Image) -> Path:
    """Build a .icns via a .iconset folder + macOS `iconutil`.

    On non-macOS hosts we still write the .iconset so build_macos.sh can
    finish the job with `iconutil`.
    """
    iconset = ASSETS / "app.iconset"
    if iconset.exists():
        shutil.rmtree(iconset)
    iconset.mkdir()

    # Apple iconutil spec: png files at exact sizes with these names.
    spec = [
        (16, "16x16"),
        (32, "16x16@2x"),
        (32, "32x32"),
        (64, "32x32@2x"),
        (128, "128x128"),
        (256, "128x128@2x"),
        (256, "256x256"),
        (512, "256x256@2x"),
        (512, "512x512"),
        (1024, "512x512@2x"),
    ]
    for px, name in spec:
        resized = img.resize((px, px), Image.LANCZOS)
        resized.save(iconset / f"icon_{name}.png", format="PNG", optimize=True)
    print(f"[ok] wrote iconset at {iconset} ({len(spec)} sizes)")

    if sys.platform == "darwin":
        out = ASSETS / "app.icns"
        if out.exists():
            out.unlink()
        subprocess.run(["iconutil", "-c", "icns", str(iconset), "-o", str(out)],
                       check=True)
        print(f"[ok] wrote {out} ({out.stat().st_size // 1024} KB)")
        return out
    else:
        print("[skip] not on macOS - wrote .iconset; build_macos.sh will iconutil it")
        return iconset


def main():
    src = render_source()
    write_png(src)
    write_ico(src)
    write_icns(src)


if __name__ == "__main__":
    main()

