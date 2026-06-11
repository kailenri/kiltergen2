"""Tests for articulated body IK + Body dataclass."""
import math

from config import FOREARM_LEN, SHIN_LEN, THIGH_LEN, UPPER_ARM_LEN, X_SPACING
from kinematics import Body, solve_2bone_ik


def test_ik_reachable_target_exact():
    """Target inside the workspace returns the exact end position."""
    anchor = (0.0, 0.0)
    target = (3.0 * X_SPACING, 0.0)  # within UPPER+FOREARM
    joint, end = solve_2bone_ik(anchor, target, UPPER_ARM_LEN, FOREARM_LEN, bend_dir=1)
    assert math.isclose(end[0], target[0], abs_tol=1e-6)
    assert math.isclose(end[1], target[1], abs_tol=1e-6)
    # Joint is exactly UPPER_ARM_LEN from anchor.
    d = math.hypot(joint[0] - anchor[0], joint[1] - anchor[1])
    assert math.isclose(d, UPPER_ARM_LEN, abs_tol=1e-6)
    # Joint is exactly FOREARM_LEN from end.
    d2 = math.hypot(end[0] - joint[0], end[1] - joint[1])
    assert math.isclose(d2, FOREARM_LEN, abs_tol=1e-6)


def test_ik_unreachable_clamps_to_line():
    """Target beyond max reach: chain straightens; end is on the anchor->target line."""
    anchor = (0.0, 0.0)
    far = (1000.0, 0.0)
    joint, end = solve_2bone_ik(anchor, far, UPPER_ARM_LEN, FOREARM_LEN, bend_dir=1)
    max_reach = UPPER_ARM_LEN + FOREARM_LEN
    assert math.isclose(end[0], max_reach, abs_tol=1e-6)
    assert math.isclose(end[1], 0.0, abs_tol=1e-6)
    assert math.isclose(joint[0], UPPER_ARM_LEN, abs_tol=1e-6)
    assert math.isclose(joint[1], 0.0, abs_tol=1e-6)


def test_ik_bend_direction_flips_joint():
    """bend_dir=+1 and bend_dir=-1 yield mirrored joint positions."""
    anchor = (0.0, 0.0)
    target = (2.0 * X_SPACING, 0.0)  # straight ahead so flip is along y axis
    j_pos, _ = solve_2bone_ik(anchor, target, UPPER_ARM_LEN, FOREARM_LEN, bend_dir=+1)
    j_neg, _ = solve_2bone_ik(anchor, target, UPPER_ARM_LEN, FOREARM_LEN, bend_dir=-1)
    # X coords equal; Y coords opposite sign.
    assert math.isclose(j_pos[0], j_neg[0], abs_tol=1e-6)
    assert math.isclose(j_pos[1], -j_neg[1], abs_tol=1e-6)
    # Both joints non-trivially off the line.
    assert abs(j_pos[1]) > 0.1


def test_body_from_full_planted_state():
    """All four limbs planted: every joint should be set and finite."""
    limbs = {
        'RH': (12.0, 50.0),
        'LH': (-12.0, 50.0),
        'RF': (10.0, 0.0),
        'LF': (-10.0, 0.0),
    }
    body = Body.from_limb_positions(limbs)
    for joint in (body.hip, body.shoulder, body.head,
                  body.r_shoulder, body.l_shoulder, body.r_hip, body.l_hip,
                  body.r_elbow, body.l_elbow, body.r_knee, body.l_knee):
        assert joint is not None
        assert math.isfinite(joint[0]) and math.isfinite(joint[1])
    # Hip is centroid of feet.
    assert math.isclose(body.hip[0], 0.0, abs_tol=1e-6)
    assert math.isclose(body.hip[1], 0.0, abs_tol=1e-6)
    # Head is above shoulder.
    assert body.head[1] > body.shoulder[1]


def test_body_with_one_foot_cut():
    """Cut foot is excluded from hip calc; cut leg joint is None."""
    limbs = {
        'RH': (12.0, 50.0),
        'LH': (-12.0, 50.0),
        'RF': (10.0, 0.0),
        'LF': (-10.0, 0.0),
    }
    body = Body.from_limb_positions(limbs, cut_feet={'LF'})
    assert body.lf is None
    assert body.l_knee is None
    # Hip = right foot only.
    assert math.isclose(body.hip[0], 10.0, abs_tol=1e-6)


def test_body_with_flag():
    """A flagging foot contributes to hip via its phantom position and gets a knee."""
    limbs = {
        'RH': (12.0, 50.0),
        'LH': (-12.0, 50.0),
        'RF': (10.0, 0.0),
        'LF': None,
    }
    body = Body.from_limb_positions(limbs, flags={'LF': (-20.0, 5.0)})
    assert body.lf is None
    assert body.lf_flag == (-20.0, 5.0)
    assert body.l_knee is not None
    # Hip centroid uses the flag position.
    expected_hip_x = (10.0 + -20.0) / 2.0
    assert math.isclose(body.hip[0], expected_hip_x, abs_tol=1e-6)


def test_body_arm_bone_lengths_respected():
    """Elbow-to-shoulder = UPPER_ARM_LEN, elbow-to-hand = FOREARM_LEN (when reachable)."""
    limbs = {
        'RH': (8.0, 30.0),   # reachable from a typical r_shoulder
        'LH': (-8.0, 30.0),
        'RF': (10.0, 0.0),
        'LF': (-10.0, 0.0),
    }
    body = Body.from_limb_positions(limbs)
    d_shoulder = math.hypot(body.r_elbow[0] - body.r_shoulder[0],
                            body.r_elbow[1] - body.r_shoulder[1])
    d_hand = math.hypot(body.rh[0] - body.r_elbow[0],
                        body.rh[1] - body.r_elbow[1])
    assert math.isclose(d_shoulder, UPPER_ARM_LEN, abs_tol=1e-4)
    assert math.isclose(d_hand, FOREARM_LEN, abs_tol=1e-4)
