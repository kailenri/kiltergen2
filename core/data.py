"""Per-angle climb extraction.

`holdstojson.py` collapses `climb_stats` with ``ROUND(AVG(difficulty_average))``,
producing one grade per climb across every angle it has been graded at. That
throws away the exact signal needed to model movement at a given angle: of the
165k climbs with stats, 53k are graded at more than one angle, and spreads of
five or six V-grades between the shallowest and steepest setting are routine.

This module keeps one record per ``(climb, angle)`` pair instead. Each record
is a supervised example -- these holds, at this angle, play at this grade --
which is what the cost model gets calibrated against.
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass, field
from typing import Dict, Iterator, List, Optional

from config import DB_PATH, MAX_HOLDS_PER_CLIMB, MIN_HOLDS_PER_CLIMB

#: Kilter role ids, as used in the `frames` / `holds_in` encoding.
ROLE_START = 12
ROLE_HAND = 13
ROLE_FINISH = 14
ROLE_FOOT = 15

HAND_ROLES = (ROLE_START, ROLE_HAND, ROLE_FINISH)


@dataclass
class AngleClimb:
    """One climb as it plays at one specific board angle."""

    uuid: str
    name: str
    angle: int
    holds: List[Dict] = field(default_factory=list)
    #: Community consensus grade at this angle (Kilter 1-39 scale).
    difficulty: Optional[float] = None
    #: Grade shown in the app, which blends consensus with the setter's guess.
    display_difficulty: Optional[float] = None
    #: How many people have logged it here. Low counts are noisy grades and
    #: should be down-weighted during calibration, not dropped outright.
    ascents: int = 0
    quality: Optional[float] = None

    @property
    def hand_holds(self) -> List[Dict]:
        return [h for h in self.holds if h.get("role_id") in HAND_ROLES]

    @property
    def foot_holds(self) -> List[Dict]:
        return [h for h in self.holds if h.get("role_id") == ROLE_FOOT]

    @property
    def start_holds(self) -> List[Dict]:
        return [h for h in self.holds if h.get("role_id") == ROLE_START]

    @property
    def finish_holds(self) -> List[Dict]:
        return [h for h in self.holds if h.get("role_id") == ROLE_FINISH]

    def key(self) -> str:
        return f"{self.uuid}@{self.angle}"


_QUERY = """
SELECT c.uuid,
       c.name,
       c.holds_in,
       cs.angle,
       cs.difficulty_average,
       cs.display_difficulty,
       cs.ascensionist_count,
       cs.quality_average
FROM climbs c
JOIN climb_stats cs ON c.uuid = cs.climb_uuid
WHERE c.holds_in IS NOT NULL
  AND cs.ascensionist_count >= ?
"""


def iter_angle_climbs(
    db_path: str = DB_PATH,
    min_ascents: int = 1,
    angles: Optional[List[int]] = None,
    limit: int = 0,
) -> Iterator[AngleClimb]:
    """Yield one `AngleClimb` per (climb, angle) pair with community stats.

    `min_ascents` filters out grades nobody has confirmed. `angles` restricts
    to specific board settings, which is the fast path when calibrating or
    comparing a single angle.
    """
    sql = _QUERY
    args: List = [min_ascents]
    if angles:
        placeholders = ",".join("?" for _ in angles)
        sql += f" AND cs.angle IN ({placeholders})"
        args.extend(angles)
    sql += " ORDER BY c.uuid, cs.angle"
    if limit > 0:
        sql += f" LIMIT {int(limit)}"

    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        for row in conn.execute(sql, args):
            holds = _parse_holds(row["holds_in"])
            if not _usable(holds):
                continue
            yield AngleClimb(
                uuid=row["uuid"],
                name=row["name"] or "",
                angle=int(row["angle"]),
                holds=holds,
                difficulty=row["difficulty_average"],
                display_difficulty=row["display_difficulty"],
                ascents=int(row["ascensionist_count"] or 0),
                quality=row["quality_average"],
            )
    finally:
        conn.close()


def angle_spread(db_path: str = DB_PATH, min_angles: int = 2) -> Dict[str, Dict]:
    """Per-climb grade spread across angles.

    The size of this spread is the headroom an angle-aware model has over one
    that averages: a climb whose grade moves six points between 20 and 50
    degrees is one an angle-blind scorer must get wrong at one end or
    the other.
    """
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """
            SELECT climb_uuid,
                   COUNT(*) AS n_angles,
                   MIN(angle) AS min_angle,
                   MAX(angle) AS max_angle,
                   MIN(difficulty_average) AS min_grade,
                   MAX(difficulty_average) AS max_grade
            FROM climb_stats
            GROUP BY climb_uuid
            HAVING n_angles >= ?
            """,
            (min_angles,),
        ).fetchall()
    finally:
        conn.close()

    return {
        r["climb_uuid"]: {
            "n_angles": r["n_angles"],
            "angle_range": (r["min_angle"], r["max_angle"]),
            "grade_range": (r["min_grade"], r["max_grade"]),
            "spread": (r["max_grade"] or 0) - (r["min_grade"] or 0),
        }
        for r in rows
    }


def _parse_holds(raw: Optional[str]) -> List[Dict]:
    if not raw:
        return []
    try:
        holds = json.loads(raw)
    except (TypeError, ValueError):
        return []
    if not isinstance(holds, list):
        return []
    return [h for h in holds if isinstance(h, dict) and "x" in h and "y" in h]


def _usable(holds: List[Dict]) -> bool:
    """Reject climbs that cannot produce a sensible sequence."""
    if not (MIN_HOLDS_PER_CLIMB <= len(holds) <= MAX_HOLDS_PER_CLIMB):
        return False
    roles = {h.get("role_id") for h in holds}
    # Needs somewhere to start, somewhere to finish, and something to pull on.
    return ROLE_START in roles and ROLE_FINISH in roles
