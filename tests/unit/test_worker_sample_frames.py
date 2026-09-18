from io import BytesIO
from pathlib import Path

from PIL import Image

from infra_mas.worker.service import _compose_contact_sheet  # pyright: ignore[reportPrivateUsage]


def test_compose_contact_sheet_preserves_all_samples_in_chronological_grid(
    tmp_path: Path,
) -> None:
    frames: list[Path] = []
    for index, color in enumerate(("red", "green", "blue"), start=1):
        path = tmp_path / f"frame-{index:03d}.jpg"
        Image.new("RGB", (20, 10), color=color).save(path)
        frames.append(path)

    encoded = _compose_contact_sheet(frames, columns=2, duration_s=30.0)

    with Image.open(BytesIO(encoded)) as sheet:
        assert sheet.format == "JPEG"
        assert sheet.size == (40, 68)
