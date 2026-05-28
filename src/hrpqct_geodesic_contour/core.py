"""Standalone geodesic contour extraction for HR-pQCT radius scans."""

from __future__ import annotations

import itertools
import logging
from pathlib import Path
from typing import Iterable

import numpy as np
from edt import edt as distance_transform_edt
from scipy.ndimage import (
    binary_dilation,
    binary_fill_holes,
    binary_opening,
    convolve,
    generate_binary_structure,
    label,
)
from skimage.exposure import equalize_adapthist
from skimage.filters import gaussian
from skimage.measure import label as label_sk
from skimage.segmentation import (
    inverse_gaussian_gradient,
    morphological_geodesic_active_contour,
    watershed,
)
from skimage.transform import rescale, resize

logger = logging.getLogger(__name__)

DEFAULT_FILTER_PARAMETERS = ((2, 14.0, 5, 2), (1, 3.0, 5, 2), (1, 1.5, 5, 2))


class ContourCancelledError(RuntimeError):
    """Raised when contour generation is cancelled by the caller."""


def contour(
    density: np.ndarray,
    voxel_size_mm: float | tuple[float, float, float] | None = None,
    masks: Iterable[np.ndarray] | None = None,
    report_path: str | Path | None = None,
    extra_debug: bool = False,
    filter_parameters: Iterable[tuple[int, float, int, int]] | None = None,
    bone_threshold: float = 250.0,
    fill_holes: bool = True,
    progress_callback=None,
    cancel_callback=None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """Return a tight outer contour for a human radius HR-pQCT scan.

    Parameters
    ----------
    density
        Three-dimensional density image, typically in mg HA/cm^3.
    voxel_size_mm
        Optional scalar or 3-tuple voxel size in millimetres. It is only stored
        as report metadata and does not change the contour algorithm.
    masks
        Optional precomputed fragment masks. If omitted, fragments are detected
        automatically by watershed. When provided, the first mask is treated as
        the target fragment and later masks are excluded from the active contour.
    report_path
        Optional HDF5 file path for intermediate arrays. Requires ``h5py``.
    extra_debug
        If true and ``report_path`` is set, save every active-contour iteration.
    filter_parameters
        Sequence of ``(scaling_factor, sigma, num_iter, smoothing)`` tuples.
    bone_threshold
        Density threshold used to seed automatic watershed fragment detection.
    fill_holes
        If true, fill enclosed cavities in the final contour mask.
    progress_callback
        Optional callable receiving progress event dictionaries.
    cancel_callback
        Optional callable returning true when processing should stop.

    Returns
    -------
    tuple
        ``(contour_mask, auxiliary_masks)`` where ``contour_mask`` has the same
        shape as ``density`` and ``auxiliary_masks`` contains the prepared
        contour support mask plus non-target fragments.
    """

    _raise_if_cancelled(cancel_callback)
    _report_progress(progress_callback, "started", message="Preparing density image.")
    density_volume = _as_3d_density(density)
    _report_progress(progress_callback, "fragments", message="Preparing fragment masks.")
    component_masks = list(masks) if masks is not None else watershed_ultradistal_arm(
        density_volume > bone_threshold
    )
    component_masks = [_as_bool_mask(mask, density_volume.shape) for mask in component_masks]

    for i, component_mask in enumerate(component_masks):
        _report(report_path, voxel_size_mm, (f"0.{i}. component", component_mask))

    _raise_if_cancelled(cancel_callback)
    _report_progress(progress_callback, "prepare_arrays", message="Preparing active contour arrays.")
    contour_energy, initial_level_set, crop_slices, original_shape, support_mask = _prepare_arrays(
        density_volume, component_masks[1:], report_path, voxel_size_mm
    )

    if filter_parameters is None:
        filter_parameters = DEFAULT_FILTER_PARAMETERS
    filter_parameters = tuple(filter_parameters)

    contour_mask = initial_level_set
    for i, (scaling_factor, sigma, num_iter, smoothing) in enumerate(filter_parameters):
        _raise_if_cancelled(cancel_callback)
        _report_progress(
            progress_callback,
            "active_contour_pass",
            current=i,
            total=len(filter_parameters),
            message=f"Running geodesic contour pass {i + 1} of {len(filter_parameters)}.",
        )
        edge_energy = padded_inverse_double_gaussian_gradient(contour_energy, sigma)
        _report(report_path, voxel_size_mm, (f"6.{i}. edge energy", edge_energy))

        if i == 0:
            edge_energy[np.logical_not(contour_mask)] = 1.0

        padding = 15
        edge_energy = np.pad(edge_energy, ((0, 0), (0, 0), (padding, padding)), mode="edge")
        padded_level_set = np.pad(contour_mask, ((0, 0), (0, 0), (padding, padding)), mode="edge")

        padded_shape = edge_energy.shape
        if scaling_factor != 1:
            edge_energy = rescale(edge_energy, 1 / scaling_factor, preserve_range=True)
            padded_level_set = (
                rescale(padded_level_set.astype(float), 1 / scaling_factor, preserve_range=True)
                > 0.5
            )

        contour_mask = morphological_geodesic_active_contour(
            edge_energy,
            init_level_set=padded_level_set,
            smoothing=smoothing,
            num_iter=num_iter,
            iter_callback=_get_callback(
                report_path,
                voxel_size_mm,
                extra_debug,
                i,
                progress_callback,
                cancel_callback,
                num_iter,
            ),
        )

        if scaling_factor != 1:
            contour_mask = resize(contour_mask, padded_shape, preserve_range=True) > 0.5

        contour_mask = contour_mask[:, :, padding:-padding] > 0.5
        _report(report_path, voxel_size_mm, (f"7.{i}. contour", contour_mask))

    _raise_if_cancelled(cancel_callback)
    if fill_holes:
        _report_progress(progress_callback, "fill_holes", message="Filling internal contour holes.")
        contour_mask = fill_internal_holes(contour_mask)

    _report_progress(progress_callback, "finished", message="Geodesic contour finished.")
    return (
        _transform_to_initial_shape(contour_mask, crop_slices, original_shape),
        [_transform_to_initial_shape(support_mask, crop_slices, original_shape)] + component_masks[1:],
    )


def binary_padded_closing_edt(array: np.ndarray, dist: float) -> np.ndarray:
    """Perform binary closing using distance transforms and padded boundaries."""
    if np.asarray(array).dtype != bool:
        raise ValueError("Only boolean arrays supported.")

    pad = int(np.ceil(dist + 1))
    padded = _pad_constant(_pad_symmetric(array, pad, [2]), pad, [0, 1])

    closed = distance_transform_edt(padded == 0) < dist
    closed = distance_transform_edt(closed == 1) > dist

    return _depad_array(closed, pad)


def binary_padded_full_closing_in_plane(array: np.ndarray, dist: float) -> np.ndarray:
    """Close and fill each axial slice of a 3D binary image."""
    if np.asarray(array).dtype != bool:
        raise ValueError("Only boolean arrays supported.")

    pad = int(np.ceil(dist + 1))
    padded = _pad_constant(_pad_symmetric(array, pad, [2]), pad, [0, 1])

    for sl in range(padded.shape[2]):
        padded[:, :, sl] = (
            distance_transform_edt(distance_transform_edt(~padded[:, :, sl]) <= dist)
            > dist
        )
        padded[:, :, sl] = label_sk(padded[:, :, sl], background=-1) != 1

    return _depad_array(padded, pad)


def binary_padded_opening(array: np.ndarray, iterations: int) -> np.ndarray:
    """Perform binary opening with padded axial boundaries."""
    pad = iterations
    padded = _pad_constant(_pad_symmetric(array, pad, [2]), pad, [0, 1])
    opened = binary_opening(padded, iterations=iterations)
    return _depad_array(opened, pad)


def fill_internal_holes(mask: np.ndarray) -> np.ndarray:
    """Fill contour cavities while treating axial stack endpoints as closed."""
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.ndim != 3:
        raise ValueError("mask must be a 3D array.")

    capped = mask_array.copy()
    capped[:, :, 0] = binary_fill_holes(capped[:, :, 0])
    capped[:, :, -1] = binary_fill_holes(capped[:, :, -1])
    return binary_fill_holes(capped)


def padded_inverse_double_gaussian_gradient(array: np.ndarray, sigma: float) -> np.ndarray:
    """Pad, smooth, and compute an inverse Gaussian gradient image."""
    pad = int(np.ceil(sigma * 4 + 1))
    padded = _pad_edge(_pad_symmetric(array, pad, [2]), pad, [0, 1])
    gradient = inverse_gaussian_gradient(gaussian(padded, sigma), sigma=sigma)
    return _depad_array(gradient, pad)


def greatest_connected_component(
    image: np.ndarray,
    connectivity: int | None = 1,
    threshold: float | None = None,
    background: float | None = None,
) -> np.ndarray:
    """Return the largest connected component in ``image`` as a boolean mask."""
    if threshold is not None and background is not None:
        raise ValueError("Only specify threshold or background, not both.")

    if threshold is not None:
        binary_image = image >= threshold
    elif background is not None:
        binary_image = image != background
    else:
        binary_image = image != 0

    if not np.any(binary_image) or np.all(binary_image):
        logger.warning("Image only contains one value.")

    labels = label_sk(binary_image.astype(int), connectivity=connectivity)
    nonzero = np.nonzero(labels.ravel())
    label_count = np.bincount(labels.ravel()[nonzero])

    if not np.any(label_count):
        return np.zeros(np.shape(image), dtype=bool)

    greatest_label = np.argmax(label_count)
    if np.sum(label_count == np.max(label_count)) > 1:
        logger.warning(
            "Multiple greatest connected components with equal size; returning lowest label."
        )

    return labels == greatest_label


def boundingbox_from_mask(mask: np.ndarray, return_type: str = "slice") -> tuple[slice, ...] | tuple[list[int], ...]:
    """Return slices or index lists describing the bounding box of ``mask``."""
    if not np.any(mask):
        raise ValueError("Given mask is empty. Cannot compute a bounding box.")

    out = []
    try:
        for axes in itertools.combinations(range(mask.ndim), mask.ndim - 1):
            nonzero = np.any(mask, axis=axes)
            extent = np.where(nonzero)[0][[0, -1]]
            extent[1] += 1
            if return_type == "slice":
                out.append(slice(*extent))
            elif return_type == "list":
                out.append(extent.tolist())
            else:
                raise ValueError("return_type must be 'slice' or 'list'.")
    except IndexError as exc:
        raise ValueError("Mask is empty. Cannot compute a bounding box.") from exc

    return tuple(reversed(out))


def count_neighbours(array_thresholded: np.ndarray, neighbourhood_array: np.ndarray | None = None) -> np.ndarray:
    """Count neighbouring true voxels for each voxel in a boolean image."""
    array = np.asarray(array_thresholded)
    if array.dtype != bool:
        raise TypeError("count_neighbours only works with boolean arrays.")
    if neighbourhood_array is None:
        neighbourhood_array = von_neumann_neighbourhood()[0]
    return convolve(array.astype(np.int8), neighbourhood_array)


def find_high_value_surface(
    array: np.ndarray,
    threshold: float,
    neighbourhood= None,
) -> np.ndarray:
    """Find voxels above threshold that touch lower-valued voxels."""
    if neighbourhood is None:
        neighbourhood = von_neumann_neighbourhood
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError("Only 3D arrays supported.")
    return _find_voxels_below(arr >= threshold, neighbourhood)


def find_low_value_surface(
    array: np.ndarray,
    threshold: float,
    neighbourhood= None,
) -> np.ndarray:
    """Find voxels below threshold that touch higher-valued voxels."""
    if neighbourhood is None:
        neighbourhood = von_neumann_neighbourhood
    arr = np.asarray(array)
    if arr.ndim != 3:
        raise ValueError("Only 3D arrays supported.")
    return _find_voxels_below(arr < threshold, neighbourhood)


def watershed_ultradistal_arm(segmentation: np.ndarray) -> list[np.ndarray]:
    """Separate radius, ulna, and nearby wrist-bone fragments by watershed."""
    radius_seed, ulna_seed = radius_ulna_seeds(segmentation)
    segmentation = binary_padded_closing_edt(segmentation, 5)
    labelled, _ = label(segmentation)
    labels, counts = np.unique(labelled, return_counts=True)

    sorted_values = sorted(zip(counts, labels), key=lambda x: x[0])[:-1]
    additional_markers = []
    for fragment_label in (value[1] for value in sorted_values[-7:-2]):
        mask = labelled == fragment_label
        bbox = boundingbox_from_mask(mask)
        if bbox[2].start == 0 and bbox[2].stop < 100:
            additional_markers.append(mask)

    marker_masks = [radius_seed, ulna_seed] + additional_markers
    energy_landscape = np.copy(segmentation).astype(int)
    energy_landscape[energy_landscape < 0] = 0
    energy_landscape *= -1

    markers = np.zeros(segmentation.shape, dtype=int)
    for marker_id, mask in enumerate(marker_masks, start=1):
        markers[mask] = marker_id

    water = watershed(energy_landscape, markers)
    return [water == marker_id for marker_id in range(1, len(marker_masks) + 1)]


def radius_ulna_seeds(segmentation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return seed masks for radius and ulna from the distal end of the scan."""
    segmentation = _as_bool_mask(segmentation, np.shape(segmentation))
    if segmentation.shape[2] < 40:
        raise ValueError("Automatic radius/ulna seeding requires at least 40 slices.")

    labelled, num = label(segmentation[:, :, -40:-35])
    if num < 2:
        raise ValueError("Could not identify both radius and ulna seed components.")

    sizes = sorted(((x, np.sum(labelled == x)) for x in range(1, num + 1)), key=lambda i: i[1])
    radius = (labelled == sizes[-1][0])[:, :, -1]
    ulna = (labelled == sizes[-2][0])[:, :, -1]
    empty = np.full(radius.shape, False)

    empty_height = segmentation.shape[2] // 2
    seed_height = segmentation.shape[2] - empty_height

    return (
        np.stack((empty,) * empty_height + (radius,) * seed_height, axis=2),
        np.stack((empty,) * empty_height + (ulna,) * seed_height, axis=2),
    )


def von_neumann_neighbourhood() -> tuple[np.ndarray, int]:
    neighbourhood = generate_binary_structure(3, 1)
    neighbourhood[1, 1, 1] = False
    return neighbourhood, int(np.sum(neighbourhood))


def moore_neighbourhood() -> tuple[np.ndarray, int]:
    neighbourhood = np.full((3, 3, 3), True)
    neighbourhood[1, 1, 1] = False
    return neighbourhood, int(np.sum(neighbourhood))


def _prepare_arrays(
    density: np.ndarray,
    excluded_components: list[np.ndarray],
    report_path: str | Path | None,
    voxel_size_mm: float | tuple[float, float, float] | None,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...], tuple[int, ...], np.ndarray]:
    bounds = boundingbox_from_mask(density > 0)
    normalized = density / np.max(density)
    normalized[bounds] = _equalize(normalized[bounds])
    support_mask = normalized > 0.5

    for component_mask in excluded_components:
        support_mask[component_mask] = False

    support_mask = greatest_connected_component(support_mask)
    support_mask = binary_padded_full_closing_in_plane(support_mask, 50)
    support_mask = _all_but_outer_background(support_mask)
    support_mask = binary_dilation(support_mask, iterations=5)

    for component_mask in excluded_components:
        support_mask[component_mask] = False

    density_roi, support_roi, crop_slices, original_shape = _crop(density, support_mask)
    excluded_rois = [np.copy(component_mask[crop_slices]) for component_mask in excluded_components]

    _report(report_path, voxel_size_mm, ("1. density roi", density_roi), ("1. support roi", support_roi))

    contour_energy = density_roi / np.max(density_roi)
    contour_energy = _equalize(contour_energy)
    for component_roi in excluded_rois:
        contour_energy[component_roi] *= 0

    support_surface = find_high_value_surface(support_roi, 0.5)
    contour_energy[support_roi == 0] = np.mean(contour_energy[support_surface])

    return contour_energy, support_roi, crop_slices, original_shape, support_roi


def _as_3d_density(density: np.ndarray) -> np.ndarray:
    density_volume = np.asarray(density, dtype=float)
    if density_volume.ndim != 3:
        raise ValueError("density must be a 3D array.")
    if not np.any(density_volume > 0):
        raise ValueError("density must contain positive foreground values.")
    return density_volume


def _as_bool_mask(mask: np.ndarray, expected_shape: tuple[int, ...]) -> np.ndarray:
    mask_array = np.asarray(mask, dtype=bool)
    if mask_array.shape != expected_shape:
        raise ValueError(f"mask shape {mask_array.shape} does not match density shape {expected_shape}.")
    return mask_array


def _padding_for_axes(pad: int, axes: Iterable[int]) -> tuple[tuple[int, int], ...]:
    axes = set(axes)
    return tuple(((pad, pad) if axis in axes else (0, 0)) for axis in range(3))


def _pad_symmetric(array: np.ndarray, pad: int, axes: Iterable[int]) -> np.ndarray:
    return np.pad(array, _padding_for_axes(pad, axes), "symmetric")


def _pad_edge(array: np.ndarray, pad: int, axes: Iterable[int]) -> np.ndarray:
    return np.pad(array, _padding_for_axes(pad, axes), "edge")


def _pad_constant(array: np.ndarray, pad: int, axes: Iterable[int]) -> np.ndarray:
    return np.pad(array, _padding_for_axes(pad, axes), "constant", constant_values=False)


def _depad_array(array: np.ndarray, pad: int) -> np.ndarray:
    return array[pad:-pad, pad:-pad, pad:-pad]


def _find_voxels_below(array_thresholded: np.ndarray, neighbourhood) -> np.ndarray:
    if np.asarray(array_thresholded).dtype != bool:
        raise TypeError("Finding surfaces only works with boolean arrays.")
    neighbourhood_array, neighbourhood_max_count = neighbourhood()
    neighbourhood_count = count_neighbours(array_thresholded, neighbourhood_array)
    return np.logical_and(neighbourhood_count < neighbourhood_max_count, array_thresholded)


def _transform_to_initial_shape(
    cropped_mask: np.ndarray,
    slices: tuple[slice, ...],
    shape: tuple[int, ...],
) -> np.ndarray:
    padding_extents = tuple((slices[i].start, shape[i] - slices[i].stop) for i in range(3))
    return np.pad(cropped_mask, padding_extents, mode="constant", constant_values=False)


def _get_callback(
    report_path: str | Path | None,
    voxel_size_mm: float | tuple[float, float, float] | None,
    extra_debug: bool,
    iteration: int,
    progress_callback=None,
    cancel_callback=None,
    total_iterations: int | None = None,
):
    report_callback = None
    if report_path is not None and extra_debug:
        report_callback = _get_iter_function(report_path, voxel_size_mm, iteration)

    def callback(level_set):
        _raise_if_cancelled(cancel_callback)
        if report_callback is not None:
            report_callback(level_set)
        callback.counter += 1
        _report_progress(
            progress_callback,
            "active_contour_iteration",
            current=callback.counter,
            total=total_iterations,
            pass_index=iteration,
            message=f"Running active contour pass {iteration + 1}.",
        )

    callback.counter = 0
    return callback


def _get_iter_function(
    report_path: str | Path,
    voxel_size_mm: float | tuple[float, float, float] | None,
    iteration: int,
):
    def iter_func(level_set):
        _report(report_path, voxel_size_mm, (f"level set {iteration} step {iter_func.counter}", level_set))
        iter_func.counter += 1

    iter_func.counter = 0
    return iter_func


def _equalize(density: np.ndarray) -> np.ndarray:
    temp_arrays = [np.zeros(density.shape), np.zeros(density.shape)]
    for axis in (0, 1):
        for in_array, out_array in zip(
            np.moveaxis(density, axis, 0), np.moveaxis(temp_arrays[axis], axis, 0)
        ):
            out_array[...] = equalize_adapthist(in_array)

    temp_arrays[0] += temp_arrays[1]
    temp_arrays[0] /= 2
    return temp_arrays[0]


def _all_but_outer_background(array: np.ndarray) -> np.ndarray:
    labelled = label_sk(array, background=-1)
    return labelled != 1


def _crop(
    density: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[slice, ...], tuple[int, ...]]:
    shape = mask.shape
    slices = boundingbox_from_mask(mask)
    return np.copy(density[slices]), np.copy(mask[slices]), slices, shape


def _report(
    report_path: str | Path | None,
    voxel_size_mm: float | tuple[float, float, float] | None,
    *datasets: tuple[str, np.ndarray],
) -> None:
    if report_path is None:
        return

    try:
        import h5py
    except ImportError as exc:
        raise ImportError("Install hrpqct-geodesic-contour[reports] to write HDF5 reports.") from exc

    with h5py.File(report_path, "a") as h5_file:
        for name, dataset in datasets:
            if name in h5_file:
                del h5_file[name]
            item = h5_file.create_dataset(name, data=dataset)
            if voxel_size_mm is not None:
                item.attrs["voxel_size_mm"] = voxel_size_mm


def _report_progress(progress_callback, stage: str, **payload) -> None:
    if progress_callback is None:
        return
    event = {"stage": stage}
    event.update(payload)
    progress_callback(event)


def _raise_if_cancelled(cancel_callback) -> None:
    if cancel_callback is not None and cancel_callback():
        raise ContourCancelledError("Geodesic contour generation was cancelled.")
