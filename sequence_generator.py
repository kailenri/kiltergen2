import numpy as np
import heapq
import math
from collections import defaultdict
from typing import List, Dict, Any, Tuple
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from time import perf_counter
from tqdm.auto import tqdm
from config import *

class ClimbSequenceGenerator:
    def __init__(self, holds_data: List[Dict], num_workers: int = 4):
        self.num_workers = num_workers
        self._setup_timing()
        self._start_timer('total_init')

        self.holds = holds_data
        self.hold_ids = [h['hole_id'] for h in holds_data]
        self.hold_dict = {h['hole_id']: h for h in holds_data}

        self.start_holds = [h for h in holds_data if h['role_id'] == 12]
        self.finish_holds = [h for h in holds_data if h['role_id'] == 14]
        self.hand_holds = [h for h in holds_data if h['role_id'] in {12, 13, 14}]
        # Include all holds that feet can use
        self.foot_holds = [h for h in holds_data if h['role_id'] in {12, 13, 14, 15}]

        if not self.start_holds or not self.finish_holds:
            raise ValueError("Missing essential start or finish holds")

        self.hold_coords = np.array([(h['x'], h['y']) for h in holds_data])
        self.hold_distances = self._calculate_distances()
        self.hand_reach_matrix = self._compute_hand_reachability_matrix()
        self.foot_reach_matrix = self._compute_foot_reachability_matrix()
        print(f"Reachable hand transitions: {np.sum(self.hand_reach_matrix)}")
        print(f"Reachable foot transitions: {np.sum(self.foot_reach_matrix)}")
        self.hold_grid = self._create_hold_grid()

        self._stop_timer('total_init')
        self._print_timing('Initialization')

    def _calculate_distances(self) -> np.ndarray:
        """Calculate pairwise distances between all holds"""
        dx = self.hold_coords[:,0,None] - self.hold_coords[:,0]
        dy = self.hold_coords[:,1,None] - self.hold_coords[:,1]
        distances = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(distances, 0)
        return distances

    def _compute_hand_reachability_matrix(self) -> np.ndarray:
        """Compute which holds are reachable by hands from each hold"""
        n = len(self.holds)
        reachable = np.zeros((n, n), dtype=bool)

        # Hand holds can reach other hand holds within MAX_HAND_REACH
        is_hand = np.array([h['role_id'] in {12, 13, 14} for h in self.holds])
        reachable = (self.hold_distances <= MAX_HAND_REACH) & is_hand[:,None] & is_hand

        # Special cases for start and finish holds
        start_indices = [i for i, h in enumerate(self.holds) if h['role_id'] == 12]
        finish_indices = [i for i, h in enumerate(self.holds) if h['role_id'] == 14]

        # Can't move from non-start holds to start holds
        for j in start_indices:
            reachable[:,j] = False
            for i in start_indices:
                reachable[i,j] = True

        # Can't move from finish holds
        for i in finish_indices:
            reachable[i,:] = False
            for j in finish_indices:
                reachable[i,j] = True

        return reachable

    def _compute_foot_reachability_matrix(self) -> np.ndarray:
        """Compute which holds are reachable by feet from each hold"""
        n = len(self.holds)
        reachable = np.zeros((n, n), dtype=bool)

        # Feet can reach all holds within MAX_FOOT_REACH
        reachable = self.hold_distances <= MAX_FOOT_REACH

        # Special restrictions for feet
        for i in range(n):
            for j in range(n):
                # Can't reach holds that are too high up (unless it's a start hold)
                if self.holds[j]['role_id'] != 12:  # Not a start hold
                    if self.holds[j]['y'] < self.holds[i]['y'] - X_SPACING * 1.5:
                        reachable[i, j] = False

        return reachable

    def _process_hold_chunk(self, holds: List[Dict]) -> Dict[float, List[Dict]]:
        chunk_grid = defaultdict(list)
        for hold in holds:
            chunk_grid[hold['y']].append(hold)
        return chunk_grid

    def _sort_column(self, y: float, holds: List[Dict]) -> List[Dict]:
        return sorted(holds, key=lambda h: h['x'])

    def _create_hold_grid(self) -> Dict[float, List[Dict]]:
        """Create a grid of holds organized by y-coordinate"""
        grid = defaultdict(list)
        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            chunk_size = max(1, len(self.holds) // (self.num_workers * 4))
            futures = [
                executor.submit(self._process_hold_chunk, self.holds[i:i + chunk_size])
                for i in range(0, len(self.holds), chunk_size)
            ]
            for future in tqdm(futures, desc="Distributing holds", leave=False):
                for y, holds in future.result().items():
                    grid[y].extend(holds)

        with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
            futures = {
                executor.submit(self._sort_column, y, holds): y
                for y, holds in grid.items()
            }
            for future in tqdm(futures, desc="Sorting columns", leave=False):
                y = futures[future]
                grid[y] = future.result()

        return grid

    @lru_cache(maxsize=100000)
    def is_valid_hand_transition(self, current_id: int, next_id: int, limb: str, current_limbs: Tuple) -> bool:
        """Check if a hand transition is valid based on reach and other constraints"""
        next_hold = self.hold_dict[next_id]
        
        # Only consider valid hand holds
        if next_hold['role_id'] not in {12, 13, 14}:
            return False
            
        # Check if hold is already occupied by another limb
        for other_limb, hold_id in current_limbs:
            if other_limb != limb and hold_id == next_id:
                return False
        
        # Check reach distance using precomputed matrix
        if current_id != -1:  # If not initial placement
            i = self.hold_ids.index(current_id)
            j = self.hold_ids.index(next_id)
            if not self.hand_reach_matrix[i, j]:
                return False
                
        return True

    def is_valid_foot_transition(self, current_id: int, next_id: int, limb: str, current_limbs: Tuple) -> bool:
        """Check if a foot transition is valid based on reach and other constraints"""
        next_hold = self.hold_dict[next_id]
        
        # Check if hold is already occupied by another limb
        for other_limb, hold_id in current_limbs:
            if other_limb != limb and hold_id == next_id:
                return False
        
        # Check reach distance using precomputed matrix
        if current_id != -1:  # If not initial placement
            i = self.hold_ids.index(current_id)
            j = self.hold_ids.index(next_id)
            if not self.foot_reach_matrix[i, j]:
                return False
        
        return True

    def evaluate_sequence(self, sequence: List[Dict]) -> Dict:
        """Evaluate a climbing sequence based on multiple metrics"""
        if not sequence:
            return {'score': 0, 'details': {}}

        metrics = {
            'hold_quality': 0,
            'movement_efficiency': 0,
            'limb_alternation': 0,
            'body_position': 0,
            'cross_prevention': 0,
            'completion': 0
        }
        
        limb_use = defaultdict(int)
        prev_limb = None
        limb_positions = {
            'RH': None, 'LH': None,
            'RF': None, 'LF': None
        }

        for i, move in enumerate(sequence):
            hold = self.hold_dict[move['hold']]
            limb = move['limb']
            limb_positions[limb] = (hold['x'], hold['y'])
            
            # Hold quality scoring
            if hold['role_id'] == 14:  # Finish hold
                metrics['hold_quality'] += 10
            elif hold['role_id'] == 12:  # Start hold
                metrics['hold_quality'] += 5
            else:  # Regular hold
                metrics['hold_quality'] += 1

            # Movement efficiency - reward shorter moves
            if i > 0:
                prev_hold = self.hold_dict[sequence[i-1]['hold']]
                dist = math.hypot(hold['x']-prev_hold['x'], hold['y']-prev_hold['y'])
                # Give higher scores for more efficient movements
                metrics['movement_efficiency'] += 1/(dist + 0.1)

            # Limb alternation (hands only)
            if limb in ['RH', 'LH'] and prev_limb in ['RH', 'LH'] and limb != prev_limb:
                metrics['limb_alternation'] += 1
            
            # Cross prevention - penalize crossed arms or legs
            if all(pos is not None for pos in [limb_positions['RH'], limb_positions['LH']]):
                if limb_positions['RH'][0] < limb_positions['LH'][0]:  # RH is left of LH (crossed)
                    metrics['cross_prevention'] -= 5
            
            if all(pos is not None for pos in [limb_positions['RF'], limb_positions['LF']]):
                if limb_positions['RF'][0] < limb_positions['LF'][0]:  # RF is left of LF (crossed)
                    metrics['cross_prevention'] -= 3
            
            # Body position - reward stable triangular positions
            if all(pos is not None for pos in [limb_positions['RH'], limb_positions['LH'], 
                                             limb_positions['RF'], limb_positions['LF']]):
                # Calculate center of gravity
                cog_x = sum(pos[0] for pos in limb_positions.values())/4
                cog_y = sum(pos[1] for pos in limb_positions.values())/4
                
                # Check if CoG is within the support polygon formed by feet
                min_foot_x = min(limb_positions['RF'][0], limb_positions['LF'][0])
                max_foot_x = max(limb_positions['RF'][0], limb_positions['LF'][0])
                
                if min_foot_x <= cog_x <= max_foot_x:
                    metrics['body_position'] += 2  # Good stability
                else:
                    metrics['body_position'] += 0.5  # Less stable
                    
                # Reward hands being higher than feet
                if min(limb_positions['RH'][1], limb_positions['LH'][1]) > max(limb_positions['RF'][1], limb_positions['LF'][1]):
                    metrics['body_position'] += 1
                    
            prev_limb = limb
            limb_use[limb] += 1

        # Completion bonus
        last_move = sequence[-1]
        metrics['completion'] = 10 if self.hold_dict[last_move['hold']]['role_id'] == 14 else 0

        # Normalize metrics
        seq_len = len(sequence)
        if seq_len > 1:
            metrics['movement_efficiency'] /= (seq_len - 1)
        
        hand_moves = sum(1 for m in sequence if m['limb'] in ['RH', 'LH'])
        if hand_moves > 1:
            metrics['limb_alternation'] /= (hand_moves - 1)
            
        # Normalize cross prevention to a 0-5 range
        metrics['cross_prevention'] = max(0, metrics['cross_prevention'] + 5)

        # Weight and combine metrics
        weights = {
            'hold_quality': 0.15,
            'movement_efficiency': 0.20,
            'limb_alternation': 0.20,
            'body_position': 0.15,
            'cross_prevention': 0.20,
            'completion': 0.10
        }

        total_score = sum(metrics[k]*weights[k] for k in metrics)

        return {
            'score': total_score,
            'details': metrics,
            'limb_balance': dict(limb_use),
            'sequence_length': seq_len
        }

    def generate_sequences(self, beam_width: int = None) -> Dict[str, Any]:
        """Generate climbing sequences using beam search"""
        # Use configured beam width if none provided
        beam_width = beam_width or BEAM_WIDTH
        
        self._start_timer('total_search')
        beam = self._initialize_beam()
        completed_sequences = []
        iterations = 0
        MAX_ITERATIONS = 100  

        with tqdm(desc="Generating sequences", unit="iter") as pbar:
            while beam and iterations < MAX_ITERATIONS:
                # Split beam for parallel processing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    chunk_size = max(1, len(beam) // (self.num_workers * 2))
                    futures = [
                        executor.submit(self._expand_states, beam[i:i+chunk_size], beam_width)
                        for i in range(0, len(beam), chunk_size)
                    ]
                    next_beam = list(chain(*[f.result() for f in futures]))

                # Keep only the best states for the next iteration
                beam = heapq.nlargest(
                    beam_width * 5,
                    next_beam,
                    key=lambda x: (x['score'], -len(x['sequence']))
                )[:beam_width]

                # Check for completed sequences
                for state in beam:
                    if self._is_complete(state):
                        evaluation = self.evaluate_sequence(state['sequence'])
                        completed_sequences.append({
                            'sequence': state['sequence'],
                            'evaluation': evaluation
                        })
                iterations += 1
                pbar.update(1)
                
                # If we've found enough complete sequences, we can stop early
                if len(completed_sequences) >= beam_width * 2:
                    break

        self._stop_timer('total_search')

        if not completed_sequences:
            return {'status': 'error', 'message': 'No valid sequences found'}

        # Sort completed sequences by evaluation score
        completed_sequences.sort(key=lambda x: x['evaluation']['score'], reverse=True)
        
        best_seq = completed_sequences[0]
        return {
            'status': 'success',
            'best_sequence': best_seq,
            'all_sequences': completed_sequences[:beam_width],
            'stats': {
                'total_sequences': len(completed_sequences),
                'processing_time': self.sections.get('total_search', 0),
                'iterations': iterations,
                'holds_processed': len(self.holds)
            }
        }

    def _expand_states(self, states: List[Dict], beam_width: int) -> List[Dict]:
        """Expand each state by trying all possible limb movements"""
        new_states = []
        for state in tqdm(states, desc="Expanding states", leave=False):
            # Skip completed states
            if self._is_complete(state):
                new_states.append(state)  # Keep completed states in the beam
                continue

            # Convert limbs dict to tuple for caching
            current_limbs_tuple = tuple(sorted(
                (limb, hold['hole_id'] if hold else -1) 
                for limb, hold in state['limbs'].items()
            ))

            # Hand movements
            for limb in ['RH', 'LH']:
                current = state['limbs'][limb]
                current_id = current['hole_id'] if current else -1

                for next_hold in self.hand_holds:
                    next_id = next_hold['hole_id']
                    if current_id != next_id and self.is_valid_hand_transition(
                            current_id, next_id, limb, current_limbs_tuple):
                        # Create new state with this hand movement
                        new_limbs = {k: v for k, v in state['limbs'].items()}
                        new_limbs[limb] = next_hold
                        
                        # Calculate state score based on hold and other factors
                        move_score = 1
                        if next_hold['role_id'] == 14:  # Finish hold
                            move_score = 10
                        elif next_hold['role_id'] == 12:  # Start hold
                            move_score = 5
                            
                        new_states.append({
                            'sequence': state['sequence'] + [{
                                'limb': limb,
                                'hold': next_id,
                                'position': next_hold.get('position')
                            }],
                            'limbs': new_limbs,
                            'score': state['score'] + move_score
                        })
                        
            # Foot movements
            for limb in ['RF', 'LF']:
                current = state['limbs'].get(limb, None)
                current_id = current['hole_id'] if current else -1
                
                for next_hold in self.foot_holds:
                    next_id = next_hold['hole_id']
                    if current_id != next_id and self.is_valid_foot_transition(
                            current_id, next_id, limb, current_limbs_tuple):
                        # Create new state with this foot movement
                        new_limbs = {k: v for k, v in state['limbs'].items()}
                        new_limbs[limb] = next_hold
                        
                        # Foot moves get lower scores
                        move_score = 0.5
                        
                        new_states.append({
                            'sequence': state['sequence'] + [{
                                'limb': limb,
                                'hold': next_id,
                                'position': next_hold.get('position')
                            }],
                            'limbs': new_limbs,
                            'score': state['score'] + move_score
                        })
        
        return new_states

    def _initialize_beam(self):
        """Initialize the beam with valid starting positions"""
        # Get potential start positions
        beam = []
        
        # Start with just hands on start holds (simpler approach)
        for rh in self.start_holds:
            for lh in self.start_holds:
                if rh != lh:  # Different holds for each hand
                    # Initial state with just hands positioned
                    beam.append({
                        'sequence': [],
                        'limbs': {
                            'RH': rh, 
                            'LH': lh,
                            'RF': None,
                            'LF': None
                        },
                        'score': 0
                    })
        
        # If beam is empty, try allowing the same hold for both hands
        if not beam and self.start_holds:
            for h in self.start_holds:
                beam.append({
                    'sequence': [],
                    'limbs': {
                        'RH': h, 
                        'LH': h,
                        'RF': None,
                        'LF': None
                    },
                    'score': 0
                })
        
        # Make sure we have some starting positions
        if not beam:
            raise ValueError("Could not create valid starting positions")
            
        return beam

    def _score_initial_position(self, state):
        """Score initial positions based on naturalness"""
        limbs = state['limbs']
        
        # Check if we have the required limbs
        if not all(limb in limbs for limb in ['RH', 'LH']):
            return 0
            
        hand_width = 0
        if limbs['RH'] and limbs['LH']:
            hand_width = abs(limbs['RH']['x'] - limbs['LH']['x'])
            
        foot_width = 0
        if limbs.get('RF') and limbs.get('LF'):
            foot_width = abs(limbs['RF']['x'] - limbs['LF']['x'])
            
        # Check if hands are above feet
        hands_above_feet = False
        if all(limbs.get(limb) for limb in ['RH', 'LH', 'RF', 'LF']):
            min_hand_y = min(limbs['RH']['y'], limbs['LH']['y'])
            max_foot_y = max(limbs['RF']['y'], limbs['LF']['y'])
            hands_above_feet = min_hand_y > max_foot_y
            
        score = hand_width + foot_width + (10 if hands_above_feet else 0)
        return score

    def _is_complete(self, state):
        """Check if a climb is complete (at least one hand on a finish hold)"""
        limbs = state['limbs']
        return (limbs['RH'] and limbs['RH']['role_id'] == 14) or \
               (limbs['LH'] and limbs['LH']['role_id'] == 14)

    def _setup_timing(self):
        self.timers = {}
        self.sections = {}

    def _start_timer(self, name):
        self.timers[name] = perf_counter()

    def _stop_timer(self, name):
        if name in self.timers:
            elapsed = perf_counter() - self.timers[name]
            self.sections[name] = elapsed

    def _print_timing(self, phase_name):
        print(f"\n{phase_name} Timings:")
        for name, duration in self.sections.items():
            print(f"- {name:<20}: {duration:.4f}s")
        total = sum(self.sections.values())
        print(f"Total {phase_name.lower()} time: {total:.4f}s")
        self.sections.clear()