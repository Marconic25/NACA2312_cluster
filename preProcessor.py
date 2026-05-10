import math
import os
import shutil

FLAP_ANGLE_DEG = +5 #positive rotates up
HINGE = (0.775, -0.045)
SNAP_DIR   = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mesh_baseline")
SOURCE_DIR = os.path.join(SNAP_DIR, "constant", "triSurface_0deg")
TARGET_DIR = os.path.join(SNAP_DIR, "constant", "triSurface")


def read_obj(filepath):
    """Read an OBJ file, return list of (line_type, data).
       line_type is 'v' for vertex lines, 'other' for everything else.
       For 'v': data is [x, y, z] as floats.
       For 'other': data is the raw line string."""
    entries = []
    with open(filepath, "r") as f:
        for line in f:
            stripped = line.strip()
            if stripped.startswith("v "):
                parts = stripped.split()
                entries.append(("v", [float(parts[1]), float(parts[2]), float(parts[3])]))
            else:
                entries.append(("other", line.rstrip("\n")))
    return entries


def rotate_vertex(x, y, z, hx, hy, angle_rad):
    """Rotate point (x,y,z) around (hx,hy) by angle_rad about Z axis."""
    cos_a = math.cos(angle_rad)
    sin_a = math.sin(angle_rad)
    dx = x - hx
    dy = y - hy
    x_new = dx * cos_a - dy * sin_a + hx
    y_new = dx * sin_a + dy * cos_a + hy
    return x_new, y_new, z


def write_obj(filepath, entries):
    """Write entries back to an OBJ file."""
    with open(filepath, "w") as f:
        for line_type, data in entries:
            if line_type == "v":
                f.write(f"v {data[0]:.6f} {data[1]:.6f} {data[2]:.6f}\n")
            else:
                f.write(data + "\n")


def main():
    angle_rad = math.radians(FLAP_ANGLE_DEG)
    hx, hy = HINGE

    # Create target triSurface directory
    os.makedirs(TARGET_DIR, exist_ok=True)

    # Copy wing_main.obj unchanged
    shutil.copy(os.path.join(SOURCE_DIR, "wing_main.obj"), TARGET_DIR)

    # Read flap, rotate, write
    flap_entries = read_obj(os.path.join(SOURCE_DIR, "flap.obj"))

    for i, (line_type, data) in enumerate(flap_entries):
        if line_type == "v":
            flap_entries[i] = ("v", list(rotate_vertex(data[0], data[1], data[2], hx, hy, angle_rad)))

    write_obj(os.path.join(TARGET_DIR, "flap.obj"), flap_entries)

    print(f"Done. Flap rotated by {FLAP_ANGLE_DEG} deg around hinge {HINGE}")
    print(f"  Source: {SOURCE_DIR}")
    print(f"  Target: {TARGET_DIR}")


if __name__ == "__main__":
    main()
