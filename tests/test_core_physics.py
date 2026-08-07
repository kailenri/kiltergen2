"""Tests for the angle-aware statics.

These pin down the behaviour that the old in-plane-only model could not
express at all: that the same body position is a different problem at 20
degrees than at 60.
"""

import math

import pytest

from config import X_SPACING
from core.params import COM_WALL_OFFSET, MAX_CORE_TENSION
from core.physics import (
    angle_regime,
    foot_cut_threshold,
    gravity_components,
    in_plane_pull_fraction,
    solve_contact_loads,
)


def standard_pose(angle_deg):
    """A generic hanging pose: hands overhead, feet low, COM between."""
    return solve_contact_loads(
        hand_xys=[(-X_SPACING, 4.0 * X_SPACING), (X_SPACING, 4.0 * X_SPACING)],
        foot_xys=[(-X_SPACING, 0.0), (X_SPACING, 0.0)],
        com_xy=(0.0, 1.8 * X_SPACING),
        angle_deg=angle_deg,
    )


# -- gravity split ----------------------------------------------------------


def test_gravity_split_at_vertical():
    """A vertical wall pulls straight down it and not off it at all."""
    par, norm = gravity_components(0.0)
    assert math.isclose(par, 1.0, abs_tol=1e-9)
    assert math.isclose(norm, 0.0, abs_tol=1e-9)


def test_gravity_split_at_roof():
    """A roof pulls entirely off the wall and not down it."""
    par, norm = gravity_components(90.0)
    assert math.isclose(par, 0.0, abs_tol=1e-9)
    assert math.isclose(norm, 1.0, abs_tol=1e-9)


def test_gravity_split_is_unit():
    for angle in (0, 10, 25, 40, 55, 70):
        par, norm = gravity_components(angle)
        assert math.isclose(math.hypot(par, norm), 1.0, abs_tol=1e-9)


# -- the core result: feet stop bearing weight as the board steepens --------


def test_feet_bear_weight_on_vertical():
    """At vertical the wall presses back on the feet and the core rests."""
    loads = standard_pose(0.0)
    assert loads.foot_normal > 0.0
    assert loads.tension_demand == 0.0
    assert loads.foot_engagement == 1.0
    assert loads.feet_secure


def test_feet_stop_bearing_weight_past_the_crossover():
    """Past the geometric crossover the feet would have to pull, not press."""
    shallow = standard_pose(10.0)
    steep = standard_pose(55.0)
    assert shallow.foot_normal > 0.0
    assert steep.foot_normal == 0.0
    assert steep.tension_demand > 0.0


def test_tension_demand_increases_monotonically_with_angle():
    """Steeper always costs more core, never less."""
    demands = [standard_pose(a).tension_demand for a in range(0, 90, 5)]
    assert demands == sorted(demands)
    assert demands[0] == 0.0
    assert demands[-1] > 0.0


def test_foot_engagement_decreases_monotonically_with_angle():
    engagements = [standard_pose(a).foot_engagement for a in range(0, 90, 5)]
    assert engagements == sorted(engagements, reverse=True)
    assert engagements[0] == pytest.approx(1.0)


def test_hand_load_increases_with_angle_and_reaches_bodyweight():
    """On a roof the hands carry the whole climber."""
    assert standard_pose(0.0).hand_load < standard_pose(45.0).hand_load
    assert standard_pose(45.0).hand_load < standard_pose(85.0).hand_load
    assert standard_pose(90.0).hand_load == pytest.approx(1.0, abs=0.02)


def test_hand_normal_pull_matches_gravity_normal_when_feet_are_off():
    """With the feet unloaded, the inward pull is exactly the off-wall pull."""
    loads = standard_pose(60.0)
    _, g_norm = gravity_components(60.0)
    assert loads.foot_normal == 0.0
    assert loads.hand_normal == pytest.approx(g_norm)


def test_extended_pose_with_high_feet_blows_off_on_steep_ground():
    """Body hanging below high feet is the pose that cannot be held.

    Tension demand scales with (COM below hands) / (hand-to-foot span), so it
    runs away exactly when the feet are high and the body is stretched out
    beneath them -- which is why a steep board spits you off a high step.
    """
    extended = solve_contact_loads(
        hand_xys=[(0.0, 5.0 * X_SPACING)],
        foot_xys=[(0.0, 3.0 * X_SPACING)],
        com_xy=(0.0, 1.0 * X_SPACING),
        angle_deg=70.0,
    )
    assert extended.tension_demand > MAX_CORE_TENSION
    assert not extended.feet_secure


def test_compressed_pose_holds_even_on_a_roof():
    """Staying tucked keeps the feet on where an extended body would cut.

    The counterpart to the test above, and the reason it is not a bug that a
    roof pose can be secure: demand is bounded by the pose's own geometry.
    """
    compressed = standard_pose(88.0)
    assert compressed.tension_demand < MAX_CORE_TENSION
    assert compressed.feet_secure
    # Feet are barely contributing even so -- secure is not the same as easy.
    assert compressed.foot_engagement < 0.2


# -- geometry drives the crossover, not a tuned constant --------------------


def test_crossover_angle_follows_geometry():
    """The feet unload at tan(theta) = com_offset / com_below_hands."""
    hands_y = 4.0 * X_SPACING
    com_y = 1.8 * X_SPACING
    com_below = hands_y - com_y
    expected = math.degrees(math.atan2(COM_WALL_OFFSET, com_below))

    just_under = standard_pose(expected - 2.0)
    just_over = standard_pose(expected + 2.0)
    assert just_under.foot_normal > 0.0
    assert just_over.foot_normal == 0.0


def test_compressed_pose_keeps_feet_on_longer_than_extended_one():
    """Hips close to the hands is why compression works on steep ground."""
    angle = 45.0
    extended = solve_contact_loads(
        hand_xys=[(0.0, 6.0 * X_SPACING)],
        foot_xys=[(0.0, 0.0)],
        com_xy=(0.0, 1.5 * X_SPACING),
        angle_deg=angle,
    )
    compressed = solve_contact_loads(
        hand_xys=[(0.0, 3.0 * X_SPACING)],
        foot_xys=[(0.0, 0.0)],
        com_xy=(0.0, 2.4 * X_SPACING),
        angle_deg=angle,
    )
    assert compressed.tension_demand < extended.tension_demand


# -- degenerate inputs ------------------------------------------------------


def test_no_feet_puts_everything_through_the_hands():
    loads = solve_contact_loads(
        hand_xys=[(0.0, 4.0 * X_SPACING)],
        foot_xys=[],
        com_xy=(0.0, 1.0 * X_SPACING),
        angle_deg=40.0,
    )
    par, norm = gravity_components(40.0)
    assert loads.hand_shear == pytest.approx(par)
    assert loads.hand_normal == pytest.approx(norm)
    assert loads.foot_shear == 0.0
    assert not loads.feet_secure


def test_scrunched_pose_does_not_divide_by_zero():
    """Hands level with feet is clamped, not a blow-up."""
    loads = solve_contact_loads(
        hand_xys=[(0.0, 0.0)],
        foot_xys=[(0.0, 0.0)],
        com_xy=(0.0, 0.0),
        angle_deg=40.0,
    )
    assert math.isfinite(loads.tension_demand)
    assert math.isfinite(loads.hand_load)


# -- balance ----------------------------------------------------------------


def test_com_inside_support_is_balanced():
    loads = solve_contact_loads(
        hand_xys=[(-X_SPACING, 4.0 * X_SPACING), (X_SPACING, 4.0 * X_SPACING)],
        foot_xys=[(-X_SPACING, 0.0), (X_SPACING, 0.0)],
        com_xy=(0.0, 2.0 * X_SPACING),
        angle_deg=30.0,
    )
    assert loads.balanced


def test_com_outside_support_is_unbalanced():
    loads = solve_contact_loads(
        hand_xys=[(-X_SPACING, 4.0 * X_SPACING), (X_SPACING, 4.0 * X_SPACING)],
        foot_xys=[(-X_SPACING, 0.0), (X_SPACING, 0.0)],
        com_xy=(10.0 * X_SPACING, 2.0 * X_SPACING),
        angle_deg=30.0,
    )
    assert not loads.balanced


# -- derived angle behaviour ------------------------------------------------


def test_foot_cut_threshold_shrinks_with_angle():
    """A reach that keeps feet on at vertical strips them on a steep board."""
    assert foot_cut_threshold(0.0) > foot_cut_threshold(40.0)
    assert foot_cut_threshold(40.0) > foot_cut_threshold(70.0)


def test_in_plane_pull_fades_with_angle():
    """Edge orientation matters less the closer the wall gets to a roof."""
    assert in_plane_pull_fraction(0.0) == pytest.approx(1.0)
    assert in_plane_pull_fraction(90.0) == pytest.approx(0.0, abs=1e-9)
    assert in_plane_pull_fraction(30.0) > in_plane_pull_fraction(60.0)


def test_angle_regimes_are_ordered():
    """Regime labels progress in one direction as the board steepens."""
    regimes = [angle_regime(a) for a in (0, 15, 30, 45, 60, 75)]
    order = {"slab": 0, "tension": 1, "steep": 2, "roof": 3}
    ranks = [order[r] for r in regimes]
    assert ranks == sorted(ranks)
    assert regimes[0] == "slab"
