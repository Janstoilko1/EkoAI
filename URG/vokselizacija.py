
"""
Projektna naloga URG - STL, vodotesnost, vokselizacija, volumen

Uporaba:
    python main_voxelization_solution.py model.stl --voxel-size 1.0 --show-stl --show-voxels

Oznake vokslov med algoritmom:
    0 = zrak/neznano
    1 = povrsje / notranja lupina
    2 = zunanja okolica
    3 = zunanja lupina
    4 = notranjost materiala
    5 = vrzel

Končni materialni model:
    0 = zrak
    1 = material
"""

from __future__ import annotations

import argparse
import struct
from collections import Counter, deque
from pathlib import Path

import numpy as np


EPS = 1e-9


# ============================================================
# 1. STL READER
# ============================================================

def is_binary_stl(path: str | Path) -> bool:
    path = Path(path)

    with open(path, "rb") as f:
        header = f.read(80)
        count_bytes = f.read(4)

    if len(count_bytes) < 4:
        return False

    triangle_count = struct.unpack("<I", count_bytes)[0]
    expected_size = 84 + triangle_count * 50
    actual_size = path.stat().st_size

    return expected_size == actual_size


def read_binary_stl_triangles(path: str | Path) -> list[np.ndarray]:
    triangles = []

    with open(path, "rb") as f:
        f.read(80)
        triangle_count = struct.unpack("<I", f.read(4))[0]

        for _ in range(triangle_count):
            data = f.read(50)
            if len(data) < 50:
                break

            # 12 float32 vrednosti:
            # normal_x, normal_y, normal_z,
            # v1x, v1y, v1z, v2x, ...
            values = struct.unpack("<12fH", data)
            v1 = values[3:6]
            v2 = values[6:9]
            v3 = values[9:12]

            triangles.append(np.array([v1, v2, v3], dtype=float))

    return triangles


def read_ascii_stl_triangles(path: str | Path) -> list[np.ndarray]:
    vertices = []
    triangles = []

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if line.startswith("vertex"):
                parts = line.split()
                p = [float(parts[1]), float(parts[2]), float(parts[3])]
                vertices.append(p)

                if len(vertices) == 3:
                    triangles.append(np.array(vertices, dtype=float))
                    vertices = []

    return triangles


def build_indexed_mesh(triangle_points: list[np.ndarray], decimals: int = 6):
    """
    STL ponavlja koordinate oglišč pri vsakem trikotniku.
    Tukaj enake točke združimo in trikotnike pretvorimo v indekse.
    """
    vertex_to_index = {}
    vertices = []
    triangles = []

    for tri in triangle_points:
        face = []

        for p in tri:
            key = tuple(round(float(x), decimals) for x in p)

            if key not in vertex_to_index:
                vertex_to_index[key] = len(vertices)
                vertices.append(key)

            face.append(vertex_to_index[key])

        triangles.append(face)

    return np.array(vertices, dtype=float), np.array(triangles, dtype=np.int32)


def read_stl(path: str | Path, decimals: int = 6):
    path = Path(path)

    if is_binary_stl(path):
        triangle_points = read_binary_stl_triangles(path)
    else:
        triangle_points = read_ascii_stl_triangles(path)

    vertices, triangles = build_indexed_mesh(triangle_points, decimals=decimals)
    return vertices, triangles


# ============================================================
# 2. VODOTESNOST
# ============================================================

def make_edge(i1, i2):
    """
    Rob 3-7 mora biti enak kot rob 7-3,
    zato indeksa vedno uredimo.
    """
    return tuple(sorted((int(i1), int(i2))))


def check_watertight(triangles: np.ndarray):
    edge_counter = Counter()

    for tri in triangles:
        a, b, c = tri

        edge_counter[make_edge(a, b)] += 1
        edge_counter[make_edge(b, c)] += 1
        edge_counter[make_edge(c, a)] += 1

    bad_edges = [
        (edge, count)
        for edge, count in edge_counter.items()
        if count != 2
    ]

    return len(bad_edges) == 0, edge_counter, bad_edges


# ============================================================
# 3. OSNOVNE FUNKCIJE ZA VOKSLE
# ============================================================

def create_voxel_grid(vertices: np.ndarray, voxel_size: float, padding_voxels: float = 1.25):
    """
    Naredi AABB prostor okoli modela.

    padding_voxels je malo večji od 1, da se zmanjša možnost,
    da površje modela leži točno na meji med dvema voksloma.
    Naloga zahteva vsaj 1 voxel odmika, zato je 1.25 še vedno OK.
    """
    min_xyz = vertices.min(axis=0)
    max_xyz = vertices.max(axis=0)

    grid_min = min_xyz - padding_voxels * voxel_size
    grid_max = max_xyz + padding_voxels * voxel_size

    shape = np.ceil((grid_max - grid_min) / voxel_size).astype(int)
    shape = np.maximum(shape, 1)

    labels = np.zeros(tuple(shape), dtype=np.int16)

    return labels, grid_min


def coord_to_index(p: np.ndarray, grid_min: np.ndarray, voxel_size: float):
    return np.floor((p - grid_min) / voxel_size).astype(int)


def voxel_bounds(i: int, j: int, k: int, grid_min: np.ndarray, voxel_size: float):
    vmin = grid_min + np.array([i, j, k], dtype=float) * voxel_size
    vmax = vmin + voxel_size
    return vmin, vmax


def voxel_corners_and_edges(vmin: np.ndarray, vmax: np.ndarray):
    x0, y0, z0 = vmin
    x1, y1, z1 = vmax

    corners = np.array([
        [x0, y0, z0],  # 0
        [x1, y0, z0],  # 1
        [x1, y1, z0],  # 2
        [x0, y1, z0],  # 3
        [x0, y0, z1],  # 4
        [x1, y0, z1],  # 5
        [x1, y1, z1],  # 6
        [x0, y1, z1],  # 7
    ], dtype=float)

    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7),
    ]

    return corners, edges


# ============================================================
# 4. TEST PRESEKA TRIKOTNIK - VOKSEL
#    Implementacija sledi ideji iz navodil:
#    ravnina trikotnika ∩ voksel -> 2D problem -> presek likov.
# ============================================================

def unique_points(points, eps=1e-8):
    unique = []

    for p in points:
        p = np.array(p, dtype=float)

        already_exists = False
        for q in unique:
            if np.linalg.norm(p - q) <= eps:
                already_exists = True
                break

        if not already_exists:
            unique.append(p)

    return np.array(unique, dtype=float)


def points_are_voxel_face(points: np.ndarray, vmin: np.ndarray, vmax: np.ndarray, eps=1e-8) -> bool:
    """
    Vrne True, če presečne točke predstavljajo celotno stranico voksla.
    To je primer dotika, ki ga po navodilih ne štejemo kot pravi presek.
    """
    if len(points) != 4:
        return False

    for axis in range(3):
        if np.all(np.abs(points[:, axis] - vmin[axis]) <= eps):
            return True
        if np.all(np.abs(points[:, axis] - vmax[axis]) <= eps):
            return True

    return False


def plane_voxel_intersection_polygon(triangle: np.ndarray, vmin: np.ndarray, vmax: np.ndarray):
    """
    Poišče presek med ravnino trikotnika in vokslom.

    Rezultat:
        points = oglišča presečnega mnogokotnika v 3D
        normal = normala ravnine

    Če ni pravega preseka, vrne (None, None).
    """
    A, B, C = triangle

    normal = np.cross(B - A, C - A)
    normal_length = np.linalg.norm(normal)

    if normal_length <= EPS:
        return None, None

    D = -np.dot(normal, A)

    corners, edges = voxel_corners_and_edges(vmin, vmax)

    intersection_points = []

    for i, j in edges:
        p0 = corners[i]
        p1 = corners[j]

        d0 = np.dot(normal, p0) + D
        d1 = np.dot(normal, p1) + D

        # Celoten rob leži v ravnini.
        if abs(d0) <= EPS and abs(d1) <= EPS:
            intersection_points.append(p0)
            intersection_points.append(p1)

        # Prvo krajišče leži v ravnini.
        elif abs(d0) <= EPS:
            intersection_points.append(p0)

        # Drugo krajišče leži v ravnini.
        elif abs(d1) <= EPS:
            intersection_points.append(p1)

        # Rob seka ravnino.
        elif d0 * d1 < 0:
            t = d0 / (d0 - d1)
            if -EPS <= t <= 1.0 + EPS:
                p = p0 + t * (p1 - p0)
                intersection_points.append(p)

    points = unique_points(intersection_points)

    # Prazen presek, točka ali daljica -> samo dotik.
    if len(points) < 3:
        return None, None

    # Če ravnina sovpada s stranico voksla -> samo dotik.
    if points_are_voxel_face(points, vmin, vmax):
        return None, None

    return points, normal


def project_points_to_plane_2d(points_3d: np.ndarray, origin: np.ndarray, normal: np.ndarray):
    """
    3D točke v ravnini preslika v 2D koordinate (u, v).

    To naredimo tako, da sestavimo lokalni koordinatni sistem:
        u_axis in v_axis ležita v ravnini,
        n_axis je normala ravnine.
    """
    n_axis = normal / np.linalg.norm(normal)

    # Če normala ni skoraj vzporedna z z-osjo, vzamemo z-os kot pomožni vektor.
    if abs(np.dot(n_axis, np.array([0.0, 0.0, 1.0]))) < 0.9:
        helper = np.array([0.0, 0.0, 1.0])
    else:
        helper = np.array([1.0, 0.0, 0.0])

    u_axis = np.cross(helper, n_axis)
    u_axis = u_axis / np.linalg.norm(u_axis)

    v_axis = np.cross(n_axis, u_axis)
    v_axis = v_axis / np.linalg.norm(v_axis)

    shifted = points_3d - origin

    u = shifted @ u_axis
    v = shifted @ v_axis

    return np.column_stack((u, v))


def cross2(o, a, b):
    return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])


def convex_hull_2d(points: np.ndarray):
    """
    Monotone chain algoritem.
    Vrne oglišča konveksne lupine v pravilnem vrstnem redu.
    """
    pts = sorted(set((round(float(p[0]), 10), round(float(p[1]), 10)) for p in points))

    if len(pts) <= 1:
        return np.array(pts, dtype=float)

    pts = [np.array(p, dtype=float) for p in pts]

    lower = []
    for p in pts:
        while len(lower) >= 2 and cross2(lower[-2], lower[-1], p) <= EPS:
            lower.pop()
        lower.append(p)

    upper = []
    for p in reversed(pts):
        while len(upper) >= 2 and cross2(upper[-2], upper[-1], p) <= EPS:
            upper.pop()
        upper.append(p)

    hull = lower[:-1] + upper[:-1]

    return np.array(hull, dtype=float)


def polygon_area_2d(poly: np.ndarray) -> float:
    if len(poly) < 3:
        return 0.0

    x = poly[:, 0]
    y = poly[:, 1]

    return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))


def convex_polygons_intersect_with_area(poly_a: np.ndarray, poly_b: np.ndarray, eps=1e-9) -> bool:
    """
    2D SAT za dva konveksna mnogokotnika.

    Vrne True samo, če imata lika pozitivno presečno ploščino.
    Če se samo dotikata v točki ali po robu, vrne False.
    """
    if polygon_area_2d(poly_a) <= eps or polygon_area_2d(poly_b) <= eps:
        return False

    for poly in (poly_a, poly_b):
        n = len(poly)

        for i in range(n):
            p1 = poly[i]
            p2 = poly[(i + 1) % n]

            edge = p2 - p1
            edge_len = np.linalg.norm(edge)

            if edge_len <= eps:
                continue

            # Pravokotnica na rob je ločitvena os.
            axis = np.array([-edge[1], edge[0]], dtype=float)
            axis = axis / np.linalg.norm(axis)

            proj_a = poly_a @ axis
            proj_b = poly_b @ axis

            min_a, max_a = proj_a.min(), proj_a.max()
            min_b, max_b = proj_b.min(), proj_b.max()

            overlap = min(max_a, max_b) - max(min_a, min_b)

            # Če je overlap <= 0, se lika ločita ali se samo dotikata.
            if overlap <= eps:
                return False

    return True


def triangle_intersects_voxel(triangle: np.ndarray, vmin: np.ndarray, vmax: np.ndarray) -> bool:
    """
    Glavni test za vokselizacijo površja.

    1. izračuna presek ravnine trikotnika z vokslom,
    2. presek in trikotnik preslika v 2D,
    3. preveri pozitivno ploščinsko sekanje med 2D trikotnikom in 2D presečnim mnogokotnikom.
    """
    intersection_points, normal = plane_voxel_intersection_polygon(triangle, vmin, vmax)

    if intersection_points is None:
        return False

    origin = triangle[0]

    tri_2d = project_points_to_plane_2d(triangle, origin, normal)
    poly_2d = project_points_to_plane_2d(intersection_points, origin, normal)

    tri_hull = convex_hull_2d(tri_2d)
    poly_hull = convex_hull_2d(poly_2d)

    if len(tri_hull) < 3 or len(poly_hull) < 3:
        return False

    return convex_polygons_intersect_with_area(tri_hull, poly_hull)


# ============================================================
# 5. VOKSELIZACIJA POVRŠJA
# ============================================================

def voxelize_surface(
    vertices: np.ndarray,
    triangles: np.ndarray,
    labels: np.ndarray,
    grid_min: np.ndarray,
    voxel_size: float,
    progress: bool = True,
):
    """
    Označi površinske voksle z oznako 1.
    """
    nx, ny, nz = labels.shape

    for tri_index, tri_indices in enumerate(triangles):
        A = vertices[int(tri_indices[0])]
        B = vertices[int(tri_indices[1])]
        C = vertices[int(tri_indices[2])]

        triangle = np.array([A, B, C], dtype=float)

        tri_min = triangle.min(axis=0)
        tri_max = triangle.max(axis=0)

        # AABB trikotnika pretvorimo v interval indeksov vokslov.
        idx_min = coord_to_index(tri_min, grid_min, voxel_size) - 1
        idx_max = coord_to_index(tri_max, grid_min, voxel_size) + 1

        idx_min = np.maximum(idx_min, [0, 0, 0])
        idx_max = np.minimum(idx_max, [nx - 1, ny - 1, nz - 1])

        for i in range(idx_min[0], idx_max[0] + 1):
            for j in range(idx_min[1], idx_max[1] + 1):
                for k in range(idx_min[2], idx_max[2] + 1):

                    if labels[i, j, k] == 1:
                        continue

                    vmin, vmax = voxel_bounds(i, j, k, grid_min, voxel_size)

                    if triangle_intersects_voxel(triangle, vmin, vmax):
                        labels[i, j, k] = 1

        if progress and (tri_index + 1) % 250 == 0:
            print(f"  obdelanih trikotnikov: {tri_index + 1}/{len(triangles)}")

    return labels


# ============================================================
# 6. FLOOD FILL IN VOKSELIZACIJA NOTRANJOSTI
# ============================================================

def neighbors6(i: int, j: int, k: int, shape):
    nx, ny, nz = shape

    candidates = [
        (i + 1, j, k),
        (i - 1, j, k),
        (i, j + 1, k),
        (i, j - 1, k),
        (i, j, k + 1),
        (i, j, k - 1),
    ]

    for a, b, c in candidates:
        if 0 <= a < nx and 0 <= b < ny and 0 <= c < nz:
            yield a, b, c


def flood_fill_value(labels: np.ndarray, start, old_value: int, new_value: int):
    if labels[start] != old_value:
        return 0

    q = deque([start])
    labels[start] = new_value
    count = 1

    while q:
        i, j, k = q.popleft()

        for ni, nj, nk in neighbors6(i, j, k, labels.shape):
            if labels[ni, nj, nk] == old_value:
                labels[ni, nj, nk] = new_value
                q.append((ni, nj, nk))
                count += 1

    return count


def find_outer_shell_start(labels: np.ndarray):
    """
    Poišče površinski voxel 1, ki meji na zunanjo okolico 2.
    To je začetni voxel zunanje lupine.
    """
    coords = np.argwhere(labels == 1)

    for i, j, k in coords:
        for ni, nj, nk in neighbors6(int(i), int(j), int(k), labels.shape):
            if labels[ni, nj, nk] == 2:
                return int(i), int(j), int(k)

    return None


def flood_air_component(labels: np.ndarray, start):
    """
    Poplavi eno regijo zračnih vokslov 0.
    Med poplavljanjem preverja, ali ta regija meji na zunanjo lupino 3.
    """
    q = deque([start])
    labels[start] = -1

    cells = [start]
    touches_outer_shell = False

    while q:
        i, j, k = q.popleft()

        for ni, nj, nk in neighbors6(i, j, k, labels.shape):
            value = labels[ni, nj, nk]

            if value == 0:
                labels[ni, nj, nk] = -1
                q.append((ni, nj, nk))
                cells.append((ni, nj, nk))

            elif value == 3:
                touches_outer_shell = True

    return cells, touches_outer_shell


def voxelize_interior(labels: np.ndarray):
    """
    Implementacija logike iz navodil:

    0 -> zrak/neznano
    1 -> površje
    2 -> zunanja okolica
    3 -> zunanja lupina
    4 -> notranjost materiala
    5 -> vrzel

    Na koncu vrne:
        material = 0/1 model
        labels = vmesne oznake za diagnostiko
    """
    labels = labels.copy()

    # 1) Zunanja okolica: flood fill iz vogala [0,0,0].
    outside_count = flood_fill_value(labels, (0, 0, 0), old_value=0, new_value=2)
    print(f"Zunanja okolica, oznaka 2: {outside_count} vokslov")

    # 2) Zunanja lupina: površinski voxel, ki meji na zunanjo okolico.
    outer_shell_start = find_outer_shell_start(labels)

    if outer_shell_start is None:
        print("OPOZORILO: zunanje lupine ni bilo mogoče najti.")
    else:
        outer_shell_count = flood_fill_value(labels, outer_shell_start, old_value=1, new_value=3)
        print(f"Zunanja lupina, oznaka 3: {outer_shell_count} vokslov")

    # 3) Vse preostale zračne regije 0.
    #    Če regija meji na zunanjo lupino 3 -> notranjost materiala 4.
    #    Sicer -> vrzel 5.
    zero_coords = np.argwhere(labels == 0)

    for coord in zero_coords:
        i, j, k = map(int, coord)

        if labels[i, j, k] != 0:
            continue

        cells, touches_outer_shell = flood_air_component(labels, (i, j, k))
        new_label = 4 if touches_outer_shell else 5

        for cell in cells:
            labels[cell] = new_label

    # 4) Končno preštevilčenje:
    #    1 notranja lupina -> 1 material
    #    2 zunanja okolica -> 0 zrak
    #    3 zunanja lupina -> 1 material
    #    4 notranjost -> 1 material
    #    5 vrzel -> 0 zrak
    material = np.zeros_like(labels, dtype=np.uint8)
    material[(labels == 1) | (labels == 3) | (labels == 4)] = 1

    return material, labels


# ============================================================
# 7. VOLUMEN
# ============================================================

def compute_volume(material_voxels: np.ndarray, voxel_size: float) -> float:
    material_count = int(np.sum(material_voxels == 1))
    voxel_volume = voxel_size ** 3

    return material_count * voxel_volume


# ============================================================
# 8. VIZUALIZACIJA
# ============================================================

def set_axes_equal(ax):
    """
    Nastavi enako merilo po x, y, z.
    """
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    y_range = abs(y_limits[1] - y_limits[0])
    z_range = abs(z_limits[1] - z_limits[0])

    max_range = max(x_range, y_range, z_range)

    x_middle = sum(x_limits) / 2
    y_middle = sum(y_limits) / 2
    z_middle = sum(z_limits) / 2

    ax.set_xlim3d([x_middle - max_range / 2, x_middle + max_range / 2])
    ax.set_ylim3d([y_middle - max_range / 2, y_middle + max_range / 2])
    ax.set_zlim3d([z_middle - max_range / 2, z_middle + max_range / 2])


def visualize_triangles(vertices: np.ndarray, triangles: np.ndarray):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    polys = []
    for tri in triangles:
        polys.append(vertices[tri])

    collection = Poly3DCollection(
        polys,
        alpha=0.35,
        edgecolor="black",
        linewidths=0.25
    )

    ax.add_collection3d(collection)

    ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
    ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
    ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    set_axes_equal(ax)

    plt.title("Trikotniški STL model")
    plt.show()


def visualize_voxels(material_voxels: np.ndarray, max_voxels_to_show: int = 200_000):
    import matplotlib.pyplot as plt

    filled = material_voxels.astype(bool)
    count = int(filled.sum())

    if count == 0:
        print("Ni materialnih vokslov za prikaz.")
        return

    if count > max_voxels_to_show:
        print(f"Preveč vokslov za prikaz ({count}). Povečaj --voxel-size ali dvigni limit.")
        return

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")

    ax.voxels(filled, edgecolor="k", linewidth=0.1)

    ax.set_xlabel("i")
    ax.set_ylabel("j")
    ax.set_zlabel("k")
    set_axes_equal(ax)

    plt.title("Vokselski model")
    plt.show()


# ============================================================
# 9. GLAVNI PROGRAM
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description="URG projektna naloga: STL uvoz, vodotesnost, vokselizacija, volumen."
    )

    parser.add_argument("stl", help="Pot do STL datoteke.")
    parser.add_argument("--voxel-size", type=float, default=1.0, help="Velikost stranice voksla.")
    parser.add_argument("--round-decimals", type=int, default=6, help="Zaokroževanje koordinat pri združevanju oglišč.")
    parser.add_argument("--show-stl", action="store_true", help="Prikaži trikotniški model.")
    parser.add_argument("--show-voxels", action="store_true", help="Prikaži vokselski model.")
    parser.add_argument("--save-npy", default="material_voxels.npy", help="Ime izhodne .npy datoteke z materialnimi voksli.")
    parser.add_argument("--no-progress", action="store_true", help="Skrij napredek pri vokselizaciji.")

    args = parser.parse_args()

    stl_path = Path(args.stl)

    if not stl_path.exists():
        raise FileNotFoundError(f"STL datoteka ne obstaja: {stl_path}")

    if args.voxel_size <= 0:
        raise ValueError("--voxel-size mora biti večji od 0.")

    print("Berem STL ...")
    vertices, triangles = read_stl(stl_path, decimals=args.round_decimals)

    print(f"Število unikatnih točk: {len(vertices)}")
    print(f"Število trikotnikov: {len(triangles)}")

    print("\nPreverjam vodotesnost ...")
    is_watertight, edge_counter, bad_edges = check_watertight(triangles)

    if is_watertight:
        print("Model je vodotesen: vsak rob se pojavi točno 2x.")
    else:
        print("Model NI vodotesen.")
        print(f"Število napačnih robov: {len(bad_edges)}")
        print("Prvih nekaj napačnih robov:")
        for edge, count in bad_edges[:20]:
            print(f"  rob {edge} -> {count}x")

    if args.show_stl:
        visualize_triangles(vertices, triangles)

    print("\nUstvarjam voxel grid ...")
    labels, grid_min = create_voxel_grid(vertices, args.voxel_size)

    print(f"Dimenzije voxel grida: {labels.shape}")
    print(f"Grid min: {grid_min}")
    print(f"Voxel size: {args.voxel_size}")

    print("\nVokseliziram površje ...")
    labels = voxelize_surface(
        vertices,
        triangles,
        labels,
        grid_min,
        args.voxel_size,
        progress=not args.no_progress
    )

    surface_count = int(np.sum(labels == 1))
    print(f"Površinski voksli, oznaka 1: {surface_count}")

    print("\nVokseliziram notranjost ...")
    material_voxels, diagnostic_labels = voxelize_interior(labels)

    volume = compute_volume(material_voxels, args.voxel_size)
    material_count = int(np.sum(material_voxels == 1))

    print("\nRezultat:")
    print(f"Materialni voksli: {material_count}")
    print(f"Volumen enega voksla: {args.voxel_size ** 3}")
    print(f"Ocenjeni volumen modela: {volume}")

    np.save(args.save_npy, material_voxels)
    print(f"\nMaterialni voksli shranjeni v: {args.save_npy}")

    diagnostic_path = Path(args.save_npy).with_name("diagnostic_labels.npy")
    np.save(diagnostic_path, diagnostic_labels)
    print(f"Diagnostične oznake shranjene v: {diagnostic_path}")

    if args.show_voxels:
        visualize_voxels(material_voxels)


if __name__ == "__main__":
    main()
