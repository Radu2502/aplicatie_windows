import numpy as np
import pytest

from ecvestru import click_tinta as A


def test_view_full_frame_downscaled_vertical():
    v = A.make_view(720, 1280, max_w=900, max_h=700)
    assert (v.x0, v.y0, v.w, v.h) == (0, 0, 720, 1280)
    assert v.disp_h == 700 and v.disp_w == round(720 * 700 / 1280)
    # colțul dreapta-jos afișat → colțul cadrului original
    assert A.to_original(v, v.disp_w, v.disp_h) == pytest.approx((720, 1280))


def test_click_maps_to_original_pixels_with_browser_rescale():
    v = A.make_view(848, 480, max_w=900, max_h=700)   # afișat 848x480 → 900x509
    # browserul a micșorat imaginea la 600 px lățime: componenta întoarce width=600
    x, y = A.to_original(v, 300, 150, shown_w=600, shown_h=600 * v.disp_h / v.disp_w)
    assert (x, y) == pytest.approx((424, 150 * 480 / (600 * v.disp_h / v.disp_w)))
    assert x == pytest.approx(424)


def test_zoom_crop_is_clamped_and_round_trips():
    v = A.make_view(848, 480, zoom=4, cx=10, cy=470)   # centrat lângă colț → lipit de margine
    assert (v.x0, v.y0) == (0, 480 - 120)
    assert (v.w, v.h) == (212, 120)
    for p in [(0, 360), (100.5, 400.25), (211, 479)]:
        assert A.to_original(v, *A.to_display(v, *p)) == pytest.approx(p)


def test_render_draws_point_where_click_maps():
    frame = np.zeros((480, 848, 3), np.uint8)
    v = A.make_view(848, 480, zoom=2, cx=600, cy=200)
    img = A.render(frame, v, {"hip": (610.0, 210.0)})
    assert img.shape == (v.disp_h, v.disp_w, 3)
    dx, dy = A.to_display(v, 610, 210)
    assert tuple(img[round(dy), round(dx)]) == A.COLORS["hip"]
