import os
from PIL import Image, ImageEnhance, ImageFilter, ImageOps


TARGET_WIDTH = 800
TARGET_HEIGHT = 480

CONTRAST = 1.7
COLOR = 1.2
SHARPEN_RADIUS = 1.2
SHARPEN_PERCENT = 140
SHARPEN_THRESHOLD = 3
DENOISE_SIZE = 3

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

# Speckle control: NONE gives cleaner flat areas on e-ink than FLOYDSTEINBERG.
if hasattr(Image, "Dither"):
    DITHER = Image.Dither.NONE
else:
    DITHER = Image.NONE


def make_palette_image(palette_rgb):
    pal = Image.new("P", (1, 1))
    flat = []
    for color in palette_rgb:
        flat.extend(color)
    flat.extend([0, 0, 0] * (256 - len(palette_rgb)))
    pal.putpalette(flat)
    return pal


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
            img = img.filter(ImageFilter.MedianFilter(size=DENOISE_SIZE))
            img = img.filter(
                ImageFilter.UnsharpMask(
                    radius=SHARPEN_RADIUS,
                    percent=SHARPEN_PERCENT,
                    threshold=SHARPEN_THRESHOLD,
                )
            )

            out = img.quantize(palette=palette, dither=DITHER)

            if OUTPUT_FORMAT.upper() == "PNG":
                out.save(out_path, format="PNG")
            else:
                out.convert("RGB").save(out_path, format="BMP")

