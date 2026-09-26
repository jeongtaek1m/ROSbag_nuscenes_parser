"""Camera models for the calibration, in torch.

  kb        Kannala-Brandt / OpenCV-fisheye: fx fy cx cy k1 k2 k3 k4
  radtan    OpenCV plumb_bob pinhole:        fx fy cx cy k1 k2 p1 p2 k3
  rational  OpenCV rational pinhole:         fx fy cx cy k1 k2 p1 p2 k3 k4 k5 k6
            (radial as (1+k1r^2+k2r^4+k3r^6)/(1+k4r^2+k5r^4+k6r^6)). Added so a
            target-based calibration that used it can be carried in unchanged --
            a plumb_bob refit of it costs 0.4-3.5 px on these lenses.

Either can carry a *windshield field*: two uniform cubic B-spline grids of
pixel displacements over the image, added after the lens model,

    pixel = lens(X) + S_inf(lens(X)) + S_near(lens(X)) / ||X||

S_inf is a smooth, arbitrary 2D distortion (a curved cover of non-constant
curvature); S_near/||X|| is the part of the refraction that depends on
distance — a thick cover makes the camera slightly non-central, and the
offset grows roughly with inverse depth (Verbiest et al., ECCV 2020).
"""
from __future__ import annotations

import numpy as np
import torch

N_INTR = {"kb": 8, "radtan": 9, "rational": 12}


def bspline_weights(f: torch.Tensor) -> torch.Tensor:
    """Uniform cubic B-spline basis for fractional positions f in [0, 1): (N, 4)."""
    f2, f3 = f * f, f * f * f
    return torch.stack([(1 - f) ** 3 / 6,
                        (3 * f3 - 6 * f2 + 4) / 6,
                        (-3 * f3 + 3 * f2 + 3 * f + 1) / 6,
                        f3 / 6], dim=-1)


class Grid:
    """Control grid (gx+3) x (gy+3) covering the W x H image with gx x gy cells."""

    def __init__(self, W: int, H: int, gx: int, gy: int):
        self.W, self.H, self.gx, self.gy = W, H, gx, gy
        self.nx, self.ny = gx + 3, gy + 3
        self.n = self.nx * self.ny

    def basis(self, u: torch.Tensor, v: torch.Tensor):
        """(indices (N,16), weights (N,16)) of the control points acting at (u, v)."""
        tx = (u / self.W).clamp(0.0, 1.0 - 1e-6) * self.gx
        ty = (v / self.H).clamp(0.0, 1.0 - 1e-6) * self.gy
        ix, iy = torch.floor(tx).long(), torch.floor(ty).long()
        wx, wy = bspline_weights(tx - ix), bspline_weights(ty - iy)
        ar = torch.arange(4, device=u.device)
        idx = (iy[:, None, None] + ar[None, :, None]) * self.nx + (ix[:, None, None] + ar[None, None, :])
        w = wy[:, :, None] * wx[:, None, :]
        return idx.reshape(-1, 16), w.reshape(-1, 16)

    def regularizer(self, n_samples=(48, 27)) -> tuple[np.ndarray, np.ndarray]:
        """Linear operators on one grid's control values (length n):
        D: second differences (smoothness), A: the affine part (constant, x, y) of
        the field it produces over the image — which the lens model and the
        extrinsic already express, so it is held at zero to fix the gauge."""
        n, nx, ny = self.n, self.nx, self.ny
        rows = []
        for j in range(ny):
            for i in range(1, nx - 1):
                r = np.zeros(n); r[j * nx + i - 1] = 1; r[j * nx + i] = -2; r[j * nx + i + 1] = 1
                rows.append(r)
        for j in range(1, ny - 1):
            for i in range(nx):
                r = np.zeros(n); r[(j - 1) * nx + i] = 1; r[j * nx + i] = -2; r[(j + 1) * nx + i] = 1
                rows.append(r)
        D = np.array(rows)
        us = torch.linspace(0, self.W - 1, n_samples[0], dtype=torch.float64)
        vs = torch.linspace(0, self.H - 1, n_samples[1], dtype=torch.float64)
        U, V = torch.meshgrid(us, vs, indexing="xy")
        idx, w = self.basis(U.reshape(-1), V.reshape(-1))
        B = np.zeros((len(idx), n))
        np.add.at(B, (np.repeat(np.arange(len(idx)), 16), idx.reshape(-1).numpy()), w.reshape(-1).numpy())
        P = np.stack([np.ones(len(idx)), U.reshape(-1).numpy() / self.W - 0.5,
                      V.reshape(-1).numpy() / self.H - 0.5], axis=1)
        A = np.linalg.pinv(P) @ B          # (3, n): affine coefficients of the field
        return D, A


def lens_project(Xc: torch.Tensor, intr: torch.Tensor, model: str) -> torch.Tensor:
    """(N, 3) camera-frame points -> (N, 2) pixels, lens model only."""
    x, y, z = Xc[:, 0], Xc[:, 1], Xc[:, 2]
    if model == "kb":
        fx, fy, cx, cy, k1, k2, k3, k4 = intr
        r = torch.sqrt(x * x + y * y).clamp_min(1e-12)
        th = torch.atan2(r, z)
        t2 = th * th
        thd = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
        s = thd / r
        return torch.stack([fx * s * x + cx, fy * s * y + cy], dim=-1)
    if model == "rational":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = intr
    else:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr
        k4 = k5 = k6 = 0.0 * fx
    zz = z.clamp_min(1e-6)
    xn, yn = x / zz, y / zz
    r2 = xn * xn + yn * yn
    rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
    xd = xn * rad + 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
    yd = yn * rad + p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
    return torch.stack([fx * xd + cx, fy * yd + cy], dim=-1)


def lens_unproject(uv: torch.Tensor, intr: torch.Tensor, model: str) -> torch.Tensor:
    """(N, 2) pixels -> (N, 3) unit rays, inverting the lens model."""
    if model == "kb":
        fx, fy, cx, cy, k1, k2, k3, k4 = intr
        mx, my = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
        thd = torch.sqrt(mx * mx + my * my).clamp_min(1e-12)
        th = thd.clone()
        for _ in range(20):
            t2 = th * th
            f = th * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4)))) - thd
            df = 1 + t2 * (3 * k1 + t2 * (5 * k2 + t2 * (7 * k3 + t2 * 9 * k4)))
            th = (th - f / df).clamp(0, 3.1)
        s = torch.sin(th) / thd
        return torch.stack([s * mx, s * my, torch.cos(th)], dim=-1)
    if model == "rational":
        fx, fy, cx, cy, k1, k2, p1, p2, k3, k4, k5, k6 = intr
    else:
        fx, fy, cx, cy, k1, k2, p1, p2, k3 = intr
        k4 = k5 = k6 = 0.0 * fx
    xd, yd = (uv[:, 0] - cx) / fx, (uv[:, 1] - cy) / fy
    xn, yn = xd.clone(), yd.clone()
    for _ in range(30):
        r2 = xn * xn + yn * yn
        rad = (1 + r2 * (k1 + r2 * (k2 + r2 * k3))) / (1 + r2 * (k4 + r2 * (k5 + r2 * k6)))
        dx = 2 * p1 * xn * yn + p2 * (r2 + 2 * xn * xn)
        dy = p1 * (r2 + 2 * yn * yn) + 2 * p2 * xn * yn
        xn, yn = (xd - dx) / rad, (yd - dy) / rad
    ray = torch.stack([xn, yn, torch.ones_like(xn)], dim=-1)
    return ray / ray.norm(dim=-1, keepdim=True)


def project(Xc: torch.Tensor, intr: torch.Tensor, model: str,
            grid: Grid | None = None, G: torch.Tensor | None = None) -> torch.Tensor:
    """Full model. G: (4, n) = [S_inf x, S_inf y, S_near x, S_near y] control values."""
    uv = lens_project(Xc, intr, model)
    if grid is None:
        return uv
    idx, w = grid.basis(uv[:, 0], uv[:, 1])
    rho = 1.0 / Xc.norm(dim=-1)
    g = G[:, idx]                                     # (4, N, 16)
    s = (g * w[None]).sum(-1)                         # (4, N)
    return uv + torch.stack([s[0] + rho * s[2], s[1] + rho * s[3]], dim=-1)


def unproject(uv: torch.Tensor, intr: torch.Tensor, model: str,
              grid: Grid | None = None, G: torch.Tensor | None = None) -> torch.Tensor:
    """Rays for observed pixels (the depth-dependent part is ignored)."""
    if grid is None:
        return lens_unproject(uv, intr, model)
    ideal = uv.clone()
    for _ in range(6):
        idx, w = grid.basis(ideal[:, 0], ideal[:, 1])
        s = (G[:2, idx] * w[None]).sum(-1)
        ideal = uv - s.T
    return lens_unproject(ideal, intr, model)
