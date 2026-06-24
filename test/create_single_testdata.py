"""Generate single-mode test data: test/testdata_single/{A,B,C}/*.tiff (32x32, 3-ch, uint16)."""

import os
import numpy as np
import tifffile

OUT = os.path.join(os.path.dirname(__file__), "testdata_single")
N_PER_CLASS = 5
SIZE = 32
CHANNELS = 3
CLASSES = ["A", "B", "C"]
SEED = 42


def generate():
    rng = np.random.default_rng(SEED)
    for cls_name in CLASSES:
        cls_dir = os.path.join(OUT, cls_name)
        os.makedirs(cls_dir, exist_ok=True)
        for i in range(N_PER_CLASS):
            img = rng.integers(0, 4096, (CHANNELS, SIZE, SIZE), dtype=np.uint16)
            path = os.path.join(cls_dir, f"cell_{i}.tiff")
            tifffile.imwrite(path, img)
    print(f"Single testdata: {sum(len(files) for _, _, files in os.walk(OUT))} files -> {OUT}")


if __name__ == "__main__":
    generate()
