import json
import sqlite3
import time
from typing import List, Dict
from tqdm.auto import tqdm
from SequenceGen.sequence_generator import ClimbSequenceGenerator
from config import *
import warnings

def get_climbs_batch(offset: int, limit: int) -> List[Dict]:
    """Fetch climbs in batches with proper row handling"""
    conn = sqlite3.connect(DB_PATH)
    try:
        # Explicitly name the columns we're selecting
        cursor = conn.execute("""
            SELECT uuid, holds_in, name 
            FROM climbs 
            WHERE is_listed = 1 
              AND layout_id = 1
              AND json_array_length(holds_in) BETWEEN ? AND ?
            ORDER BY uuid
            LIMIT ? OFFSET ?
        """, (MIN_HOLDS_PER_CLIMB, MAX_HOLDS_PER_CLIMB, limit, offset))
        
        processed = []
        for row in cursor.fetchall():
            try:
                # Access columns by position since we're getting tuples
                climb_uuid = row[0]
                holds_json = row[1]
                boulder_name = row[2]
                
                holds = json.loads(holds_json)
                if validate_holds(holds):
                    processed.append({
                        'id': climb_uuid,
                        'name': boulder_name,
                        'holds': holds
                    })
            except (json.JSONDecodeError, IndexError) as e:
                print(f"Skipping invalid climb data: {str(e)}")
                continue
        return processed
    finally:
        conn.close()

def validate_holds(holds: List[Dict]) -> bool:
    """Validate holds structure and required fields"""
    if not isinstance(holds, list):
        return False
        
    required = ['x', 'y', 'hole_id', 'role_id']
    for h in holds:
        if not isinstance(h, dict):
            return False
        if not all(k in h for k in required):
            return False
        # More flexible role validation - accept any plausible role ID
        if not isinstance(h['role_id'], (int, float)):
            return False
    return True

def process_climb(climb: Dict) -> Dict:
    """Process a single climb with robust error handling"""
    try:
        if not climb.get('holds'):
            return {'id': climb.get('id'), 'name': climb.get('name'), 'error': 'No holds data'}
            
        # Create the generator and explicitly use configured beam width
        generator = ClimbSequenceGenerator(climb['holds'])
        # Pass the BEAM_WIDTH from config
        result = generator.generate_sequences(beam_width=BEAM_WIDTH)
        
        if result['status'] != 'success':
            return {
                'id': climb['id'], 
                'name': climb.get('name'),  # Include name in error case
                'error': result.get('message', 'Unknown error')
            }
            
        return {
            'id': climb['id'],
            'name': climb.get('name'),  # Include name in success case
            'best_sequence': result['best_sequence'],
            'all_sequences': result.get('all_sequences', [])[:3],  # Save top 3 alternatives
            'stats': result['stats']
        }
    except Exception as e:
        return {'id': climb.get('id'), 'name': climb.get('name'), 'error': str(e)}


def main():
    warnings.filterwarnings('ignore', category=RuntimeWarning)
    
    print(f"Processing max {MAX_CLIMBS_TO_PROCESS} climbs "
          f"with {MIN_HOLDS_PER_CLIMB}-{MAX_HOLDS_PER_CLIMB} holds each")
    print(f"Using beam width of {BEAM_WIDTH}")
    
    results = []
    batch_size = 20  
    total_processed = 0
    
    start_time = time.time()
    
    with tqdm(total=MAX_CLIMBS_TO_PROCESS, desc="Overall progress") as pbar:
        while total_processed < MAX_CLIMBS_TO_PROCESS:
            climbs = get_climbs_batch(total_processed, batch_size)
            if not climbs:
                print("No more climbs to process")
                break
                
            # Process current batch
            for i, climb in enumerate(climbs):
                batch_start = time.time()
                print(f"\nProcessing climb {total_processed + i + 1}/{MAX_CLIMBS_TO_PROCESS}: {climb['id']}")
                result = process_climb(climb)
                
                if 'error' in result:
                    print(f"Error: {result['error']}")
                else:
                    seq_len = result['best_sequence']['evaluation']['sequence_length']
                    score = result['best_sequence']['evaluation']['score']
                    print(f"Success! Sequence length: {seq_len}, Score: {score:.2f}")
                    
                results.append(result)
                pbar.update(1)
                
                batch_time = time.time() - batch_start
                print(f"Climb processed in {batch_time:.2f} seconds")
                
                if len(results) >= MAX_CLIMBS_TO_PROCESS:
                    break
                    
            total_processed += len(climbs)
    
    # Calculate statistics about the results
    successful = sum(1 for r in results if 'best_sequence' in r)
    failed = sum(1 for r in results if 'error' in r)
    
    # Save results with metadata
    output_file = f"climb_results_{int(time.time())}.json"
    with open(output_file, 'w') as f:
        json.dump({
            'metadata': {
                'total_attempted': total_processed,
                'successful': successful,
                'failed': failed,
                'parameters': {
                    'max_climbs': MAX_CLIMBS_TO_PROCESS,
                    'max_holds': MAX_HOLDS_PER_CLIMB,
                    'min_holds': MIN_HOLDS_PER_CLIMB,
                    'beam_width': BEAM_WIDTH,
                    'max_hand_reach': MAX_HAND_REACH,
                    'max_foot_reach': MAX_FOOT_REACH
                }
            },
            'results': results
        }, f, indent=2)
    
    print(f"\nProcessing complete. Saved results to {output_file}")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Success rate: {successful/max(1, total_processed)*100:.1f}%")

if __name__ == "__main__":
    start_time = time.time()
    main()
    elapsed = time.time() - start_time
    print(f"Total runtime: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")