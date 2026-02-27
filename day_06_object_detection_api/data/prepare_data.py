"""
Day 06 — Object Detection API: Data Preparation Pipeline
Downloads sample images for testing and demonstration.
"""

import io
import json
import struct
import zlib
from pathlib import Path
from loguru import logger

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import config


# ─── Sample Images ───────────────────────────────────────────────────
# We generate simple synthetic images for testing (no external downloads needed)

def _create_minimal_png(width: int, height: int, color: tuple[int, int, int]) -> bytes:
    """
    Create a minimal valid PNG image in memory.

    Args:
        width: Image width in pixels.
        height: Image height in pixels.
        color: RGB color tuple.

    Returns:
        PNG file bytes.
    """

    def _chunk(chunk_type: bytes, data: bytes) -> bytes:
        c = chunk_type + data
        crc = struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)
        return struct.pack(">I", len(data)) + c + crc

    # PNG signature
    signature = b"\x89PNG\r\n\x1a\n"

    # IHDR
    ihdr_data = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    ihdr = _chunk(b"IHDR", ihdr_data)

    # IDAT — raw pixel data
    raw_rows = b""
    for _ in range(height):
        raw_rows += b"\x00"  # filter byte (None)
        for _ in range(width):
            raw_rows += bytes(color)

    compressed = zlib.compress(raw_rows)
    idat = _chunk(b"IDAT", compressed)

    # IEND
    iend = _chunk(b"IEND", b"")

    return signature + ihdr + idat + iend


SAMPLE_IMAGES = [
    {
        "name": "sample_street.png",
        "description": "Synthetic street scene placeholder (red-ish)",
        "color": (180, 60, 40),
        "width": 640,
        "height": 480,
    },
    {
        "name": "sample_park.png",
        "description": "Synthetic park scene placeholder (green-ish)",
        "color": (50, 160, 60),
        "width": 640,
        "height": 480,
    },
    {
        "name": "sample_office.png",
        "description": "Synthetic office scene placeholder (blue-ish)",
        "color": (40, 80, 180),
        "width": 640,
        "height": 480,
    },
    {
        "name": "sample_kitchen.png",
        "description": "Synthetic kitchen scene placeholder (orange-ish)",
        "color": (200, 140, 40),
        "width": 640,
        "height": 480,
    },
    {
        "name": "sample_small.png",
        "description": "Small test image (gray)",
        "color": (128, 128, 128),
        "width": 320,
        "height": 240,
    },
]


def prepare_sample_images() -> Path:
    """
    Create sample synthetic PNG images for testing and demonstration.

    Returns:
        Path to the raw images directory.
    """
    logger.info("Preparing sample images...")

    manifest = []
    for img_info in SAMPLE_IMAGES:
        img_path = config.RAW_DIR / img_info["name"]

        png_bytes = _create_minimal_png(
            width=img_info["width"],
            height=img_info["height"],
            color=img_info["color"],
        )
        img_path.write_bytes(png_bytes)

        manifest.append({
            "filename": img_info["name"],
            "description": img_info["description"],
            "width": img_info["width"],
            "height": img_info["height"],
            "path": str(img_path),
        })
        logger.info(f"  📸 Created {img_info['name']} ({img_info['width']}x{img_info['height']})")

    # Save manifest
    manifest_path = config.PROCESSED_DIR / "image_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    logger.success(f"Prepared {len(SAMPLE_IMAGES)} sample images → {config.RAW_DIR}")
    return config.RAW_DIR


def load_image_manifest() -> list[dict]:
    """Load the image manifest."""
    manifest_path = config.PROCESSED_DIR / "image_manifest.json"
    if not manifest_path.exists():
        prepare_sample_images()

    with open(manifest_path, "r") as f:
        return json.load(f)


if __name__ == "__main__":
    prepare_sample_images()
    manifest = load_image_manifest()
    for img in manifest:
        print(f"  📸 {img['filename']} — {img['description']}")
