import json
import argparse
import os
import sqlite3
from collections import defaultdict
import random

def setup_parser():
    """Set up command line argument parser"""
    parser = argparse.ArgumentParser(
        description="Extract holds_in data from climbs database"
    )
    
    parser.add_argument(
        "--input", 
        required=True,
        help="Path to SQLite database file"
    )
    parser.add_argument(
        "--output", 
        default="climb_data.json",
        help="Path to output JSON file"
    )
    parser.add_argument(
        "--table", 
        default="climbs",
        help="Table name containing climbs data"
    )
    parser.add_argument(
        "--limit", 
        type=int,
        default=0,
        help="Limit the number of climbs to extract (0 for all)"
    )
    
    return parser

def extract_holds_in(db_path, table_name, limit=0):
    """Extract holds_in data from the database"""
    print(f"Connecting to database: {db_path}")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    
    # Check if the table exists
    cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table_name}'")
    if not cursor.fetchone():
        print(f"Error: Table '{table_name}' not found in database")
        conn.close()
        return []
    
    # Check if holds_in column exists
    cursor.execute(f"PRAGMA table_info({table_name})")
    columns = [row['name'] for row in cursor.fetchall()]
    if 'holds_in' not in columns:
        print(f"Error: 'holds_in' column not found in table '{table_name}'")
        print(f"Available columns: {', '.join(columns)}")
        conn.close()
        return []
    
    # Get columns for query
    columns_to_fetch = ['uuid']
    if 'name' in columns:
        columns_to_fetch.append('name')
    columns_to_fetch.append('holds_in')
    
    # Prepare the query
    query = f"SELECT {', '.join(columns_to_fetch)} FROM {table_name} WHERE holds_in IS NOT NULL"
    
    if limit > 0:
        query += f" LIMIT {limit}"
    
    print(f"Executing query: {query}")
    cursor.execute(query)
    rows = cursor.fetchall()
    print(f"Found {len(rows)} climbs with non-empty holds_in data")
    
    # Convert to list of dictionaries
    results = []
    for row in rows:
        try:
            # Parse holds_in as JSON
            holds_in = json.loads(row['holds_in'])
            
            result = {
                "id": row['uuid'],
                "name": row['name'] if 'name' in row.keys() else f"Climb {len(results) + 1}",
                "best_sequence": {
                    "holds": holds_in,
                    "sequence": generate_sequence_from_holds(holds_in)
                }
            }
            results.append(result)
        except json.JSONDecodeError:
            print(f"Warning: Could not parse holds_in data for climb {row['uuid']} - skipping")
        except Exception as e:
            print(f"Error processing climb {row['uuid']}: {e}")
    
    conn.close()
    print(f"Successfully processed {len(results)} climbs")
    return results

def generate_sequence_from_holds(holds):
    """Generate a plausible climbing sequence from holds data"""
    if not holds:
        return []
    
    # Map holds by role
    holds_by_role = defaultdict(list)
    for hold in holds:
        if 'role_id' in hold and 'id' in hold:
            holds_by_role[hold['role_id']].append(hold)
    
    sequence = []
    
    # Add start holds (typically role_id = 12)
    start_holds = holds_by_role.get(12, [])
    if not start_holds:
        # If no start holds defined, use the lowest holds
        sorted_by_y = sorted(holds, key=lambda h: h.get('y', 0))
        start_holds = sorted_by_y[:min(2, len(sorted_by_y))]
    
    for i, hold in enumerate(start_holds[:2]):
        sequence.append({
            'hold': hold.get('id', hold.get('hole_id', str(i))),
            'limb': 'RH' if i == 0 else 'LH'
        })
    
    # Add hand holds (typically role_id = 13)
    hand_holds = holds_by_role.get(13, [])
    if not hand_holds and holds:
        # If no hand holds defined, use middle holds
        all_holds = sorted(holds, key=lambda h: h.get('y', 0))
        if len(all_holds) > 4:
            hand_holds = all_holds[2:-2]  # Skip first 2 and last 2
        else:
            hand_holds = all_holds
    
    # Randomize a bit
    if hand_holds:
        random.shuffle(hand_holds)
        hand_count = min(len(hand_holds), random.randint(3, 8))
        
        for i, hold in enumerate(hand_holds[:hand_count]):
            limb = 'RH' if (i + len(sequence)) % 2 == 0 else 'LH'
            sequence.append({
                'hold': hold.get('id', hold.get('hole_id', str(i))),
                'limb': limb
            })
    
    # Add finish holds (typically role_id = 14)
    finish_holds = holds_by_role.get(14, [])
    if not finish_holds and holds:
        # If no finish holds defined, use the highest holds
        sorted_by_y = sorted(holds, key=lambda h: h.get('y', 0), reverse=True)
        finish_holds = sorted_by_y[:min(1, len(sorted_by_y))]
    
    for i, hold in enumerate(finish_holds[:1]):
        limb = 'RH' if len(sequence) % 2 == 0 else 'LH'
        sequence.append({
            'hold': hold.get('id', hold.get('hole_id', str(i))),
            'limb': limb
        })
    
    return sequence

def save_json(results, output_path):
    """Save results to JSON file"""
    data = {"results": results}
    
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Data saved to {output_path}")
    return True

def main():
    """Main entry point"""
    parser = setup_parser()
    args = parser.parse_args()
    
    try:
        results = extract_holds_in(args.input, args.table, args.limit)
        
        if results:
            save_json(results, args.output)
            print(f"Successfully saved {len(results)} climbs to {args.output}")
        else:
            print("No valid holds_in data found")
    
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()