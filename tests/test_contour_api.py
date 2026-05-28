import importlib
import sys

import numpy as np
from scipy import ndimage


def test_package_import_exposes_public_contour_api():
    package = importlib.import_module("hrpqct_geodesic_contour")

    assert hasattr(package, "contour")
    assert hasattr(package, "ContourCancelledError")


def test_contour_accepts_plain_numpy_density_and_returns_original_shape(monkeypatch):
    from hrpqct_geodesic_contour import contour
    import hrpqct_geodesic_contour.core as core

    density = np.zeros((16, 16, 18), dtype=float)
    density[4:12, 4:12, 4:14] = 600.0
    excluded_component = np.zeros_like(density, dtype=bool)

    def fake_active_contour(image, init_level_set, **kwargs):
        return init_level_set

    monkeypatch.setattr(core, "morphological_geodesic_active_contour", fake_active_contour)

    contour_mask, auxiliary_masks = contour(
        density,
        voxel_size_mm=0.0607,
        masks=[excluded_component],
        filter_parameters=((1, 1.0, 1, 1),),
    )

    assert contour_mask.shape == density.shape
    assert contour_mask.dtype == bool
    assert len(auxiliary_masks) == 1
    assert auxiliary_masks[0].shape == density.shape


def test_contour_reports_progress_during_active_contour_iterations(monkeypatch):
    from hrpqct_geodesic_contour import contour
    import hrpqct_geodesic_contour.core as core

    density = np.zeros((16, 16, 18), dtype=float)
    density[4:12, 4:12, 4:14] = 600.0
    progress_events = []

    def fake_active_contour(image, init_level_set, **kwargs):
        kwargs["iter_callback"](init_level_set)
        return init_level_set

    monkeypatch.setattr(core, "morphological_geodesic_active_contour", fake_active_contour)

    contour(
        density,
        masks=[np.zeros_like(density, dtype=bool)],
        filter_parameters=((1, 1.0, 1, 1),),
        progress_callback=lambda event: progress_events.append(event),
    )

    assert any(event["stage"] == "active_contour_iteration" for event in progress_events)
    assert progress_events[-1]["stage"] == "finished"


def test_contour_can_be_cancelled_from_callback(monkeypatch):
    from hrpqct_geodesic_contour import ContourCancelledError, contour
    import hrpqct_geodesic_contour.core as core

    density = np.zeros((16, 16, 18), dtype=float)
    density[4:12, 4:12, 4:14] = 600.0

    def fake_active_contour(image, init_level_set, **kwargs):
        kwargs["iter_callback"](init_level_set)
        return init_level_set

    monkeypatch.setattr(core, "morphological_geodesic_active_contour", fake_active_contour)

    with np.testing.assert_raises(ContourCancelledError):
        contour(
            density,
            masks=[np.zeros_like(density, dtype=bool)],
            filter_parameters=((1, 1.0, 1, 1),),
            cancel_callback=lambda: True,
        )


def test_contour_fills_internal_holes_in_final_mask(monkeypatch):
    from hrpqct_geodesic_contour import contour
    import hrpqct_geodesic_contour.core as core

    density = np.zeros((16, 16, 18), dtype=float)
    density[4:12, 4:12, 4:14] = 600.0

    def fake_active_contour(image, init_level_set, **kwargs):
        contour_result = np.array(init_level_set, dtype=bool, copy=True)
        contour_result[3:5, 3:5, 18:20] = False
        return contour_result

    monkeypatch.setattr(core, "morphological_geodesic_active_contour", fake_active_contour)

    contour_mask, _auxiliary_masks = contour(
        density,
        masks=[np.zeros_like(density, dtype=bool)],
        filter_parameters=((1, 1.0, 1, 1),),
    )

    assert contour_mask[7:9, 7:9, 7:9].all()


def test_fill_internal_holes_fills_enclosed_cavities_but_preserves_exterior():
    from hrpqct_geodesic_contour.core import fill_internal_holes

    mask = np.zeros((9, 9, 9), dtype=bool)
    mask[2:7, 2:7, 2:7] = True
    mask[4, 4, 4] = False

    filled = fill_internal_holes(mask)

    assert filled[4, 4, 4]
    assert not filled[0, 0, 0]
    np.testing.assert_array_equal(filled, ndimage.binary_fill_holes(mask))


def test_fill_internal_holes_treats_top_and_bottom_faces_as_closed():
    from hrpqct_geodesic_contour.core import fill_internal_holes

    mask = np.zeros((9, 9, 7), dtype=bool)
    mask[2:7, 2:7, :] = True
    mask[4, 4, :] = False

    filled = fill_internal_holes(mask)

    assert filled[4, 4, :].all()
    assert not filled[0, 0, 0]


def test_contour_rejects_non_3d_density():
    from hrpqct_geodesic_contour import contour

    density = np.zeros((16, 16), dtype=float)

    with np.testing.assert_raises(ValueError):
        contour(density)
