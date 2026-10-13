"""
| File: rotations.py
| Author: Marcelo Jacinto (marcelo.jacinto@tecnico.ulisboa.pt)
| License: BSD-3-Clause. Copyright (c) 2023, Marcelo Jacinto. All rights reserved.
| Description: Implements utilitary rotations between ENU and NED inertial frame conventions and FLU and FRD body frame conventions.
"""
import numpy as np
from scipy.spatial.transform import Rotation

# Quaternion for rotation between ENU and NED INERTIAL frames
# NED to ENU: +PI/2 rotation about Z (Down) followed by a +PI rotation around X (old North/new East)
# ENU to NED: +PI/2 rotation about Z (Up) followed by a +PI rotation about X (old East/new North)
# This rotation is symmetric, so q_ENU_to_NED == q_NED_to_ENU.
# Note: this quaternion follows the convention [qx, qy, qz, qw]
q_ENU_to_NED = np.array([0.70711, 0.70711, 0.0, 0.0])

# A scipy rotation from the ENU inertial frame to the NED inertial frame of reference
rot_ENU_to_NED = Rotation.from_quat(q_ENU_to_NED)

# Quaternion for rotation between body FLU and body FRD frames
# +PI rotation around X (Forward) axis rotates from Forward, Right, Down (aircraft)
# to Forward, Left, Up (base_link) frames and vice-versa.
# This rotation is symmetric, so q_FLU_to_FRD == q_FRD_to_FLU.
# Note: this quaternion follows the convention [qx, qy, qz, qw]
q_FLU_to_FRD = np.array([1.0, 0.0, 0.0, 0.0])

# A scipe rotation from the FLU body frame to the FRD body frame
rot_FLU_to_FRD = Rotation.from_quat(q_FLU_to_FRD)

# ---------------------------------------------------------------------------------------------------------------------
# Lightweight numpy equivalents for the per-physics-step hot paths. Creating scipy Rotation objects costs tens of
# microseconds each and the state/sensor/backend code created ~13 per physics step. These helpers give the same results
# (to floating-point round-off) using plain 3x3 matrices and quaternion products.
# ---------------------------------------------------------------------------------------------------------------------

# Matrices of the constant rotations above (scipy normalises the quaternions, so these match the scipy objects exactly)
R_ENU_to_NED = rot_ENU_to_NED.as_matrix()
R_FLU_to_FRD = rot_FLU_to_FRD.as_matrix()
_q_ENU_to_NED_unit = rot_ENU_to_NED.as_quat()
_q_FLU_to_FRD_unit = rot_FLU_to_FRD.as_quat()


def quat_to_matrix(q):
    """Rotation matrix of an [x, y, z, w] quaternion, normalised first like scipy's Rotation.from_quat."""
    x, y, z, w = np.asarray(q, dtype=np.float64) / np.sqrt(np.dot(q, q))
    return np.array([
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ])


def quat_multiply(p, q):
    """Hamilton product p * q of two [x, y, z, w] quaternions (the composition scipy's Rotation p * q performs)."""
    px, py, pz, pw = p
    qx, qy, qz, qw = q
    return np.array([
        pw * qx + px * qw + py * qz - pz * qy,
        pw * qy - px * qz + py * qw + pz * qx,
        pw * qz + px * qy - py * qx + pz * qw,
        pw * qw - px * qx - py * qy - pz * qz,
    ])


def attitude_flu_enu_to_frd_ned(q_flu_enu):
    """[x, y, z, w] attitude of the FRD body frame relative to NED, from the FLU/ENU (Isaac Sim) attitude.
    Equivalent to (rot_ENU_to_NED * Rotation.from_quat(q) * rot_FLU_to_FRD).as_quat()."""
    q = np.asarray(q_flu_enu, dtype=np.float64)
    q = q / np.sqrt(np.dot(q, q))
    return quat_multiply(quat_multiply(_q_ENU_to_NED_unit, q), _q_FLU_to_FRD_unit)
