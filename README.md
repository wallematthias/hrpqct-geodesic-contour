# HR-pQCT Geodesic Contour

Standalone geodesic active-contour segmentation for fractured human radius
HR-pQCT scans. The package provides a plain NumPy API for use in analysis
scripts, notebooks, and 3D Slicer extensions.

Implemented according to:

Ohs N, Collins CJ, Tourolle DC, Atkins PR, Schroeder BJ, Blauth M, Christen P,
Mueller R. Automated segmentation of fractured distal radii by 3D geodesic
active contouring of in vivo HR-pQCT images. Bone. 2021 Jun;147:115930.
doi:10.1016/j.bone.2021.115930. PMID:33753277.

## Install

```bash
pip install hrpqct-geodesic-contour
```

For optional HDF5 debug reports:

```bash
pip install "hrpqct-geodesic-contour[reports]"
```

## Usage

```python
import numpy as np
from hrpqct_geodesic_contour import contour

density = np.asarray(...)  # 3D density image, typically mg HA/cm^3
mask, auxiliary_masks = contour(density, voxel_size_mm=0.0607)
```
