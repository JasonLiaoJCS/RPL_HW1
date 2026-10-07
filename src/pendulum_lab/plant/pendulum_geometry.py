"""Pendulum-link geometric constants, parsed from the URDF.

The packaged `data/model.urdf` provides nominal CAD properties.
This module parses the URDF on import and exposes three constants:
    PENDULUM_MASS_KG, PENDULUM_COM_M, PENDULUM_I_COM_SWING_KG_M2.

These nominal geometric properties do not establish the properties of a
student's assembled rig; validate the model against your own measurements.

Intentionally stdlib-only so the sysid workflow can import it without
pulling in MuJoCo / gymnasium.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path


URDF_PATH = Path(__file__).resolve().parent / "data" / "model.urdf"


def _load_pendulum_geometry(urdf_path: Path) -> tuple[float, float, float]:
    """Parse pendulum-link mass, swing-axis COM distance, and I_com from URDF.

    Returns (mass_kg, com_m, I_com_swing_kg_m2) where:
    - mass_kg: total pendulum mass.
    - com_m: perpendicular distance from the rotation axis (link x-axis,
      matching the `arm_to_pendulum` joint axis) to the COM. Equals
      sqrt(y² + z²) of the inertial origin — the x-component is *along*
      the rotation axis and has no effect on swing dynamics (see URDF
      note in `data/model.urdf`).
    - I_com_swing_kg_m2: pendulum's moment of inertia about its own COM
      along the swing axis (ixx of the inertia tensor at COM).
    """
    inertial = ET.parse(urdf_path).getroot().find(
        "./link[@name='pendulum']/inertial"
    )
    if inertial is None:
        raise RuntimeError(
            f"Could not find <link name='pendulum'>/<inertial> in {urdf_path}"
        )
    mass = float(inertial.find("mass").get("value"))
    xyz = [float(v) for v in inertial.find("origin").get("xyz").split()]
    com_m = (xyz[1] ** 2 + xyz[2] ** 2) ** 0.5
    ixx = float(inertial.find("inertia").get("ixx"))
    return mass, com_m, ixx


PENDULUM_MASS_KG, PENDULUM_COM_M, PENDULUM_I_COM_SWING_KG_M2 = (
    _load_pendulum_geometry(URDF_PATH)
)
