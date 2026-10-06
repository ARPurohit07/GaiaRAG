import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from gaia_rag.units import (
    absolute_magnitude,
    distance_renderings,
    ly_to_pc,
    parallax_mas_to_pc,
    pc_to_ly,
)


def test_pc_ly_roundtrip():
    assert abs(ly_to_pc(pc_to_ly(100.0)) - 100.0) < 1e-9


def test_known_distances():
    # Proxima Centauri: 1.301 pc is 4.24 ly.
    assert abs(pc_to_ly(1.301) - 4.243) < 0.01
    # 1 mas parallax is 1 kpc by definition.
    assert abs(parallax_mas_to_pc(1.0) - 1000.0) < 1e-9


def test_absolute_magnitude_at_10pc_is_apparent():
    assert abs(absolute_magnitude(5.0, 10.0) - 5.0) < 1e-12


def test_renderings_cover_units():
    r = distance_renderings(1000.0)
    assert abs(r["kpc"] - 1.0) < 1e-9
    assert abs(r["ly"] - 3261.56) < 0.1


def test_bad_parallax_raises():
    try:
        parallax_mas_to_pc(-0.5)
    except ValueError:
        return
    raise AssertionError("negative parallax must raise")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("PASS", name)
