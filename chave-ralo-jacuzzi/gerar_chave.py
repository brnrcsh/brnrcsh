#!/usr/bin/env python3
"""Chave para soltar bujão de ralo (rosca externa BSP) com cabo de vassoura.

A chave tem rosca interna BSP (gás), fenda para abraçadeira e furo para
parafuso de trava. O cabo entra no soquete e é travado por dois parafusos,
para girar o bujão de fora da água.

Gera, para cada medida:
  chave-g*.stl        ferramenta completa
  anel-teste-g*.stl   anel curto só para testar se a rosca entra no bujão

Imprima o anel primeiro. Se ele rosqueia, imprima a chave da mesma medida.
"""

from __future__ import annotations

import os

import numpy as np
import manifold3d as m3d

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

# Furo do cabo. 24 mm aceita cabo de madeira de 22–23 mm, com folga para
# tinta e ovalização. Os parafusos travam o giro.
HANDLE_HOLE_D = 24.0
HANDLE_OD = 40.0
SOCKET_DEPTH = 52.0
SHOULDER = 8.0
COLLET = 18.0
CLEARANCE_D = 0.60  # folga no diâmetro da rosca, para sair da impressora
WALL = 7.0
SLIT_W = 2.4
N_THETA = 96
Z_STEP = 0.18

# (arquivo, nome, diâmetro maior, diâmetro menor, passo, entalhes)
# Diâmetros básicos ISO 228 / rosca gás BSP.
SIZES = [
    ("g1-2", '1/2"', 20.955, 18.631, 25.4 / 14, 1),
    ("g3-4", '3/4"', 26.441, 24.117, 25.4 / 14, 2),
    ("g1", '1"', 33.249, 30.291, 25.4 / 11, 3),
    ("g1-1-4", '1.1/4"', 41.910, 38.952, 25.4 / 11, 4),
    ("g1-1-2", '1.1/2"', 47.803, 44.845, 25.4 / 11, 5),
]


def r_of(phase: float, r_maj: float, r_min: float, pitch: float) -> float:
    """Perfil Whitworth 55° da rosca externa, fase em [0, passo)."""
    x = phase % pitch
    flat = pitch / 6.0
    flank = pitch / 3.0
    if x < flat / 2:
        return r_maj
    x -= flat / 2
    if x < flank:
        return r_maj + (r_min - r_maj) * (x / flank)
    x -= flank
    if x < flat:
        return r_min
    x -= flat
    if x < flank:
        return r_min + (r_maj - r_min) * (x / flank)
    return r_maj


def helix_bolt(r_maj: float, r_min: float, pitch: float, z0: float, z1: float) -> m3d.Manifold:
    """Sólido da rosca macho (núcleo + filete). Rosca direita."""
    zs = np.arange(z0, z1, Z_STEP)
    if len(zs) == 0 or zs[-1] < z1 - 1e-6:
        zs = np.append(zs, z1)
    thetas = np.linspace(0.0, 2.0 * np.pi, N_THETA, endpoint=False)
    n_z = len(zs)
    n_th = len(thetas)
    verts = np.empty((n_z * n_th + 2, 3), dtype=np.float32)
    k = 0
    lead = pitch / (2.0 * np.pi)
    for z in zs:
        for th in thetas:
            phase = (z - lead * th) % pitch
            r = r_of(phase, r_maj, r_min, pitch)
            verts[k, 0] = r * np.cos(th)
            verts[k, 1] = r * np.sin(th)
            verts[k, 2] = z
            k += 1
    bottom_i = k
    top_i = k + 1
    verts[bottom_i] = (0.0, 0.0, zs[0])
    verts[top_i] = (0.0, 0.0, zs[-1])

    faces = []
    for i in range(n_z - 1):
        for j in range(n_th):
            v00 = i * n_th + j
            v01 = i * n_th + (j + 1) % n_th
            v10 = (i + 1) * n_th + j
            v11 = (i + 1) * n_th + (j + 1) % n_th
            faces.append((v00, v01, v11))
            faces.append((v00, v11, v10))
    for j in range(n_th):
        faces.append((bottom_i, (j + 1) % n_th, j))
        base = (n_z - 1) * n_th
        faces.append((top_i, base + j, base + (j + 1) % n_th))

    mesh = m3d.Mesh(
        vert_properties=np.ascontiguousarray(verts[:, :3]),
        tri_verts=np.ascontiguousarray(np.array(faces, dtype=np.int32)),
    )
    solid = m3d.Manifold(mesh)
    if solid.is_empty() or solid.volume() <= 0 or str(solid.status()) not in ("Error.NoError", "NoError"):
        raise RuntimeError(f"rosca macho inválida: status={solid.status()} vol={solid.volume()}")
    return solid


def cyl(h: float, r: float, r2: float | None = None, segs: int = 64, center: bool = False) -> m3d.Manifold:
    if r2 is None:
        r2 = r
    return m3d.Manifold.cylinder(float(h), float(r), float(r2), int(segs), center)


def hex_prism(across_flats: float, h: float) -> m3d.Manifold:
    # No cilindro de 6 lados o raio é até o vértice. Entre faces = R * sqrt(3).
    radius = across_flats / np.sqrt(3.0)
    return m3d.Manifold.cylinder(float(h), float(radius), float(radius), 6, False)


def write_stl(path: str, solid: m3d.Manifold) -> None:
    mesh = solid.to_mesh()
    verts = np.asarray(mesh.vert_properties, dtype=np.float32)[:, :3]
    faces = np.asarray(mesh.tri_verts, dtype=np.int32)
    tris = verts[faces]
    normals = np.cross(tris[:, 1] - tris[:, 0], tris[:, 2] - tris[:, 0])
    length = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = normals / np.maximum(length, 1e-12)
    dtype = np.dtype([("n", "<f4", (3,)), ("v", "<f4", (3, 3)), ("a", "<u2")])
    rec = np.zeros(len(faces), dtype=dtype)
    rec["n"] = normals.astype(np.float32)
    rec["v"] = tris
    with open(path, "wb") as fh:
        fh.write(b"chave bujao ralo jacuzzi BSP".ljust(80, b" "))
        fh.write(np.uint32(len(faces)).tobytes())
        rec.tofile(fh)


def radii_at(solid: m3d.Manifold, z: float, r_max_hole: float) -> np.ndarray:
    """Raios do filete, ignorando as fendas (0/90/180/270°) e o lado de fora."""
    cs = solid.slice(float(z))
    pts = np.concatenate([np.asarray(p, dtype=float) for p in cs.to_polygons()])
    r = np.hypot(pts[:, 0], pts[:, 1])
    ang = np.degrees(np.arctan2(pts[:, 1], pts[:, 0])) % 180.0
    near_slit = np.minimum(ang % 90.0, 90.0 - (ang % 90.0)) < 8.0
    hole = (r < r_max_hole) & ~near_slit
    return r[hole]


def build_wrench(major: float, minor: float, pitch: float, notches: int) -> tuple[m3d.Manifold, dict]:
    r_maj = (major + CLEARANCE_D) / 2.0
    r_min = (minor + CLEARANCE_D) / 2.0
    collet_od = (major + CLEARANCE_D) + 2.0 * WALL
    collet_r = collet_od / 2.0
    hex_af = collet_od + 8.0

    z_seat = SOCKET_DEPTH + SHOULDER
    z_mouth = z_seat + COLLET

    handle = cyl(z_seat, HANDLE_OD / 2.0, segs=80)
    hexpart = hex_prism(hex_af, (z_seat + 3.0) - (SOCKET_DEPTH - 8.0)).translate(
        [0, 0, SOCKET_DEPTH - 8.0]
    )
    collet = cyl(COLLET, collet_r, segs=80).translate([0, 0, z_seat])
    body = handle + hexpart + collet

    body -= cyl(SOCKET_DEPTH, HANDLE_HOLE_D / 2.0, segs=80)

    thread = helix_bolt(r_maj, r_min, pitch, z_seat + 0.35, z_mouth + 1.2)
    seat = cyl(0.7, r_maj + 0.05, segs=80).translate([0, 0, z_seat])
    lead = cyl(3.0, r_maj + 2.0, segs=64).translate([0, 0, z_mouth - 2.6])
    body -= thread + seat + lead

    slit_z0 = z_seat + 4.0
    slit_h = (z_mouth + 0.6) - slit_z0
    slit = m3d.Manifold.cube([collet_od + 6.0, SLIT_W, slit_h], True).translate(
        [0, 0, slit_z0 + slit_h / 2.0]
    )
    body -= slit
    body -= slit.rotate([0, 0, 90])

    groove_w = 10.0
    groove_depth = 1.5
    groove_z = z_mouth - 7.0
    tube = cyl(groove_w, collet_r + 3.0, segs=80).translate([0, 0, groove_z - groove_w / 2.0])
    tube = tube - cyl(groove_w + 1.0, collet_r - groove_depth, segs=80).translate(
        [0, 0, groove_z - (groove_w + 1.0) / 2.0]
    )
    body -= tube

    # Parafuso de trava: furo só numa garra, da parte de fora até o bujão.
    # Não atravessa o centro, senão o parafuso não encosta na rosca.
    screw_z = z_seat + 7.0
    screw_len = (collet_r - r_min) + 5.0
    screw_mid_r = (collet_r + r_min) / 2.0
    screw = cyl(screw_len, 2.55, segs=24, center=True).rotate([0, 90, 0])
    screw = screw.translate([screw_mid_r, 0, screw_z]).rotate([0, 0, 45])
    body -= screw

    flat_r = hex_af / 2.0
    for i in range(notches):
        notch = cyl(3.0, 1.7, segs=16, center=True).rotate([0, 90, 0]).translate(
            [flat_r - 0.3, -6.0 + i * 4.0, SOCKET_DEPTH + 2.0]
        )
        body -= notch

    # Cruz destacável dentro do soquete, para o ombro não precisar de ponte longa.
    rib_h = SOCKET_DEPTH + 1.6
    rib = m3d.Manifold.cube([20.0, 1.05, rib_h], False).translate([-10.0, -0.525, 0])
    body += rib
    body += rib.rotate([0, 0, 90])

    for pin_z in (15.0, 34.0):
        pin = cyl(HANDLE_OD + 8.0, 3.1, segs=24, center=True).rotate([0, 90, 0])
        body -= pin.translate([0, 0, pin_z])

    info = {
        "collet_od": collet_od,
        "groove_d": collet_od - 2 * groove_depth,
        "hex_af": hex_af,
        "z_seat": z_seat,
        "z_mouth": z_mouth,
        "r_maj": r_maj,
        "r_min": r_min,
        "pitch": pitch,
        "height": z_mouth,
    }
    return body, info


def build_test_ring(major: float, minor: float, pitch: float, notches: int) -> m3d.Manifold:
    r_maj = (major + CLEARANCE_D) / 2.0
    r_min = (minor + CLEARANCE_D) / 2.0
    h = 12.0
    af = (major + CLEARANCE_D) + 2.0 * WALL
    ring = hex_prism(af, h)
    thread = helix_bolt(r_maj, r_min, pitch, -1.0, h + 1.0)
    lead = cyl(2.4, r_maj + 1.8, segs=48).translate([0, 0, h - 2.2])
    ring -= thread + lead
    for i in range(notches):
        notch = cyl(2.4, 1.6, segs=12).rotate([0, 90, 0]).translate(
            [af / np.sqrt(3.0) * 0.92, -af * 0.18 + i * 3.6, h / 2]
        )
        ring -= notch
    return ring


def check_wrench(solid: m3d.Manifold, info: dict, label: str) -> None:
    status = str(solid.status())
    if "NoError" not in status:
        raise RuntimeError(f"{label}: status {status}")
    if solid.volume() < 10000:
        raise RuntimeError(f"{label}: volume baixo {solid.volume():.0f}")

    bb = solid.bounding_box()
    height = bb[5] - bb[2]
    if abs(height - info["height"]) > 0.8:
        raise RuntimeError(f"{label}: altura {height:.2f} != {info['height']:.2f}")

    z_thread = info["z_seat"] + COLLET * 0.72
    inner = radii_at(solid, z_thread, info["r_maj"] + 1.2)
    if len(inner) < 20:
        raise RuntimeError(f"{label}: corte da rosca sem furo ({len(inner)} pts)")
    if inner.min() > info["r_min"] + 0.55 or inner.max() < info["r_maj"] - 0.55:
        raise RuntimeError(
            f"{label}: raios da rosca {inner.min():.2f}–{inner.max():.2f}, "
            f"esperado {info['r_min']:.2f}–{info['r_maj']:.2f}"
        )

    # O ombro fecha o furo da rosca: no meio do ombro não há furo grande.
    z_shoulder = info["z_seat"] - SHOULDER / 2.0
    hole = radii_at(solid, z_shoulder, HANDLE_HOLE_D / 2.0 + 1.0)
    if len(hole) and hole.min() < HANDLE_HOLE_D / 2.0:
        # a cruz ocupa o centro; não pode existir o furo da rosca (raio ~r_min)
        if np.median(hole) > info["r_min"]:
            raise RuntimeError(f"{label}: ombro furado")

    cs = solid.slice(z_thread)
    parts = cs.decompose()
    n = len(parts) if isinstance(parts, (list, tuple)) else None
    # 4 garras separadas no corte da pinça
    if n is not None and n < 4:
        raise RuntimeError(f"{label}: fendas não separaram as garras ({n})")

    print(
        f"  ok {label}: vol={solid.volume()/1000:.1f} cm³  "
        f"rosca r {inner.min():.2f}–{inner.max():.2f}  "
        f"garras={n}  altura={height:.1f} mm"
    )


def main() -> None:
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Folga da rosca: +{:.2f} mm no diâmetro".format(CLEARANCE_D))
    print(f"Furo do cabo: {HANDLE_HOLE_D:.1f} mm   OD do cabo: {HANDLE_OD:.0f} mm")
    print()
    for key, name, major, minor, pitch, notches in SIZES:
        print(f"BSP {name}")
        wrench, info = build_wrench(major, minor, pitch, notches)
        check_wrench(wrench, info, name)
        wrench_path = os.path.join(OUT_DIR, f"chave-{key}.stl")
        write_stl(wrench_path, wrench)

        ring = build_test_ring(major, minor, pitch, notches)
        if "NoError" not in str(ring.status()) or ring.volume() <= 0:
            raise RuntimeError(f"anel {name} inválido {ring.status()} {ring.volume()}")
        ring_path = os.path.join(OUT_DIR, f"anel-teste-{key}.stl")
        write_stl(ring_path, ring)
        print(
            f"  OD da pinça {info['collet_od']:.1f} mm | "
            f"canal da abraçadeira Ø {info['groove_d']:.1f} mm | "
            f"sextavado {info['hex_af']:.0f} mm"
        )
        print(f"  {os.path.basename(wrench_path)}  {os.path.basename(ring_path)}")
        print()


if __name__ == "__main__":
    main()
