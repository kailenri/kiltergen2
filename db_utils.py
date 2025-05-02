import sqlite3
import json
import re
from typing import List, Dict
from config import DB_PATH

def process_climbs() -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        for column in ['holds_in', 'boulder_name']:
            try:
                conn.execute(f"ALTER TABLE climbs ADD COLUMN {column} TEXT")
            except sqlite3.OperationalError:
                pass

        climbs = conn.execute("""
            SELECT uuid, frames 
            FROM climbs 
            WHERE is_listed = 1 AND layout_id = 1
            AND frames IS NOT NULL
        """).fetchall()

        processed_climbs = []
        for climb in climbs:
            holds = parse_frames_string(climb['frames'], conn)
            conn.execute("""
                UPDATE climbs 
                SET holds_in = ? 
                WHERE uuid = ?
            """, (json.dumps(holds, indent=2), climb['uuid']))
            processed_climbs.append({
                'uuid': climb['uuid'],
                'holds_in': json.dumps(holds, indent=2)
            })

        conn.execute("""
            UPDATE climbs
            SET boulder_name = (
                SELECT dg.boulder_name
                FROM climb_stats cs
                JOIN difficulty_grades dg ON ROUND(cs.difficulty_average) = dg.difficulty
                WHERE cs.climb_uuid = climbs.uuid
                AND ROUND(cs.difficulty_average) IN (
                    SELECT difficulty FROM difficulty_grades
                )
            )
            WHERE is_listed = 1 AND layout_id = 1
        """)

        conn.commit()
        return processed_climbs
    finally:
        conn.close()

def parse_frames_string(frames: str, conn: sqlite3.Connection) -> List[Dict]:
    """Parse frames string into holds data with better missing hold handling"""
    pattern = re.compile(r'p(\d+)r(\d+)')
    matches = pattern.findall(frames)
    
    holds = []
    missing_placements = set()
    placement_ids_to_check = {int(placement_id_str) for placement_id_str, _ in matches}
    
    # Pre-check which placement IDs exist in the database and get their hole_ids
    placement_to_hole_map = {}
    cursor = conn.execute("""
        SELECT p.id, p.hole_id, h.x, h.y, h.name 
        FROM placements p
        JOIN holes h ON p.hole_id = h.id
        WHERE p.id IN ({})
    """.format(','.join('?' for _ in placement_ids_to_check)),
        list(placement_ids_to_check))
    
    for row in cursor.fetchall():
        placement_to_hole_map[row['id']] = {
            'hole_id': row['hole_id'],
            'x': row['x'],
            'y': row['y'],
            'name': row['name']
        }
    
    missing_placements = placement_ids_to_check - set(placement_to_hole_map.keys())
    if missing_placements:
        print(f"Critical Warning: Missing {len(missing_placements)} placements in database: {sorted(missing_placements)}")
        print("These placements are referenced in frames but don't exist in the placements table.")
        print("This will affect sequence generation for these climbs.")
    
    # Now process only the existing placements
    for placement_id_str, role_id_str in matches:
        placement_id = int(placement_id_str)
        if placement_id not in placement_to_hole_map:
            continue
            
        try:
            role_id = int(role_id_str)
            hole_data = placement_to_hole_map[placement_id]
            
            # Get role data
            role = conn.execute("SELECT position FROM placement_roles WHERE id = ?", (role_id,)).fetchone()
            
            holds.append({
                'placement_id': placement_id,
                'hole_id': hole_data['hole_id'],
                'x': hole_data['x'],
                'y': hole_data['y'],
                'name': hole_data['name'],
                'position': role['position'] if role else None,
                'role_id': role_id
            })
            
        except ValueError:
            print(f"Invalid ID format: p{placement_id_str}r{role_id_str}")
            continue
    
    return holds