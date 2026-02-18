import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


TARGET_WIDTH = 800
TARGET_HEIGHT = 480

CONTRAST = 1.5
COLOR = 1.2
SHARPEN_RADIUS = 1.0
SHARPEN_PERCENT = 140
SHARPEN_THRESHOLD = 3
DITHER_STRENGTH = 1.0  # 0.0 = off, 1.0 = full Floyd-Steinberg, 0.0-1.0 = scaled diffusion

# Keep output as PNG to preserve indexed palette output quality.
OUTPUT_FORMAT = "PNG"

# 6-color palette for the Waveshare 7.3" E panel.
PALETTE = [
    (0, 0, 0),        # black
    (255, 255, 255),  # white
    (0, 255, 0),      # green
    (0, 0, 255),      # blue
    (255, 0, 0),      # red
    (255, 255, 0),    # yellow
]

if hasattr(Image, "Dither"):
    DITHER = Image.Dither.FLOYDSTEINBERG
    DITHER_NONE = Image.Dither.NONE
else:
    DITHER = Image.FLOYDSTEINBERG
    DITHER_NONE = Image.NONE


def make_palette_image(palette_rgb):
    pal = Image.new("P", (1, 1))
    flat = []
    for color in palette_rgb:
        flat.extend(color)
    flat.extend([0, 0, 0] * (256 - len(palette_rgb)))
    pal.putpalette(flat)
    return pal


def make_palette_flat(palette_rgb):
    flat = []
    for color in palette_rgb:
        flat.extend(color)
    flat.extend([0, 0, 0] * (256 - len(palette_rgb)))
    return flat


def clamp_u8(value):
    return max(0.0, min(255.0, value))


def nearest_palette_index(r, g, b, palette_rgb):
    best_idx = 0
    best_dist = float("inf")
    for i, (pr, pg, pb) in enumerate(palette_rgb):
        dr = r - pr
        dg = g - pg
        db = b - pb
        dist = dr * dr + dg * dg + db * db
        if dist < best_dist:
            best_dist = dist
            best_idx = i
    return best_idx


def quantize_with_dither_strength(img_rgb, palette_rgb, dither_strength):
    width, height = img_rgb.size
    data = list(img_rgb.getdata())
    px_count = width * height
    buf = [0.0] * (px_count * 3)
    out_idx = [0] * px_count

    for i, (r, g, b) in enumerate(data):
        bi = i * 3
        buf[bi] = float(r)
        buf[bi + 1] = float(g)
        buf[bi + 2] = float(b)

    def add_error(x, y, er, eg, eb, factor):
        if x < 0 or y < 0 or x >= width or y >= height:
            return
        bi = (y * width + x) * 3
        buf[bi] = clamp_u8(buf[bi] + er * factor)
        buf[bi + 1] = clamp_u8(buf[bi + 1] + eg * factor)
        buf[bi + 2] = clamp_u8(buf[bi + 2] + eb * factor)

    strength = max(0.0, min(1.0, float(dither_strength)))

    for y in range(height):
        for x in range(width):
            pi = y * width + x
            bi = pi * 3
            old_r = buf[bi]
            old_g = buf[bi + 1]
            old_b = buf[bi + 2]

            pal_idx = nearest_palette_index(old_r, old_g, old_b, palette_rgb)
            new_r, new_g, new_b = palette_rgb[pal_idx]
            out_idx[pi] = pal_idx

            if strength <= 0.0:
                continue

            er = (old_r - new_r) * strength
            eg = (old_g - new_g) * strength
            eb = (old_b - new_b) * strength

            add_error(x + 1, y, er, eg, eb, 7.0 / 16.0)
            add_error(x - 1, y + 1, er, eg, eb, 3.0 / 16.0)
            add_error(x, y + 1, er, eg, eb, 5.0 / 16.0)
            add_error(x + 1, y + 1, er, eg, eb, 1.0 / 16.0)

    out = Image.new("P", (width, height))
    out.putpalette(make_palette_flat(palette_rgb))
    out.putdata(out_idx)
    return out


class ImageConverter:
    """
    Class to convert images for display on the e-Paper screen.
    """

    def __init__(self, source_dir, output_dir):
        self.source_dir = source_dir
        self.output_dir = output_dir

    # Finds valid image files in the source directory to process.
    def process_images(self):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tiff')

        for img in os.listdir(self.source_dir):

            if img.startswith('.'):
                continue
            
            print(f"Found file: {img}")
            img_path = os.path.join(self.source_dir, img)

            if os.path.isfile(img_path) and img.lower().endswith(valid_extensions):
                print(f"Preprocessing image: {img_path}")
                self.preprocess_image(img_path, img)
            

    # Uses fixed resize, contrast/color boost, unsharp mask, then panel palette quantization.
    def preprocess_image(self, img_path, file_name):
        palette = make_palette_image(PALETTE)
        base_name = os.path.splitext(file_name)[0]
        out_name = f"{base_name}_waveshare6.png"
        out_path = os.path.join(self.output_dir, out_name)

        with Image.open(img_path) as img:
            img = ImageOps.exif_transpose(img)
            img = img.convert("RGB")
            img = img.resize((TARGET_WIDTH, TARGET_HEIGHT), Image.Resampling.LANCZOS)

            img = ImageEnhance.Contrast(img).enhance(CONTRAST)
            img = ImageEnhance.Color(img).enhance(COLOR)
            img = img.filter(
                ImageFilter.UnsharpMask(
                    radius=SHARPEN_RADIUS,
                    percent=SHARPEN_PERCENT,
                    threshold=SHARPEN_THRESHOLD,
                )
            )

            if DITHER_STRENGTH <= 0.0:
                out = img.quantize(palette=palette, dither=DITHER_NONE)
            elif DITHER_STRENGTH >= 1.0:
                out = img.quantize(palette=palette, dither=DITHER)
            else:
                out = quantize_with_dither_strength(img, PALETTE, DITHER_STRENGTH)

            if OUTPUT_FORMAT.upper() == "PNG":
                out.save(out_path, format="PNG")
            else:
                out.convert("RGB").save(out_path, format="BMP")
