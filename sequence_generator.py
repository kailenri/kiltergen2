import numpy as np
import heapq
import math
import json
import os
from collections import defaultdict
from typing import List, Dict, Any, Tuple, Optional
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from itertools import chain
from time import perf_counter
from tqdm.auto import tqdm
from config import *
from kinematics import (
    estimate_hip as _kin_estimate_hip,
    estimate_shoulder as _kin_estimate_shoulder,
    ellipse_dnorm as _kin_ellipse_dnorm,
    convex_hull as _kin_convex_hull,
    point_in_poly as _kin_point_in_poly,
    is_dynamic_hand_move,
    choose_cut_foot,
    is_crossing_hand_move,
    compute_flag_position,
    choose_flag_foot,
)


# ---------------------------------------------------------------------------
# Hold orientation + quality (from scripts/build_hold_orientations.py output)
# ---------------------------------------------------------------------------

_ORIENTATIONS_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    'data', 'hold_orientations.json',
)
_ORIENTATIONS_CACHE: Optional[Dict[int, Dict]] = None


def load_hold_orientations() -> Dict[int, Dict]:
    """Lazy-load per-hold orientation metadata. Empty dict if file missing."""
    global _ORIENTATIONS_CACHE
    if _ORIENTATIONS_CACHE is not None:
        return _ORIENTATIONS_CACHE
    _ORIENTATIONS_CACHE = {}
    if not os.path.exists(_ORIENTATIONS_PATH):
        return _ORIENTATIONS_CACHE
    try:
        with open(_ORIENTATIONS_PATH) as f:
            payload = json.load(f)
        for hid_str, entry in payload.get('holds', {}).items():
            try:
                _ORIENTATIONS_CACHE[int(hid_str)] = entry
            except (TypeError, ValueError):
                continue
    except Exception:
        pass
    return _ORIENTATIONS_CACHE


def get_hold_quality(hole_id: int) -> float:
    """Quality in [0.5, 1.0]; neutral 0.7 fallback for unmapped holds."""
    entry = load_hold_orientations().get(hole_id)
    if entry is None:
        return 0.7
    return float(entry.get('quality', 0.7))


def get_hold_direction(hole_id: int, min_confidence: float = 0.4
                       ) -> Optional[float]:
    """Returns direction_deg (edge-tangent) or None if confidence too low."""
    entry = load_hold_orientations().get(hole_id)
    if entry is None:
        return None
    if float(entry.get('confidence', 0.0)) < min_confidence:
        return None
    return float(entry.get('direction_deg', 0.0))

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
        # Feet may use any hold
        self.foot_holds = list(holds_data)

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
        dx = self.hold_coords[:,0,None] - self.hold_coords[:,0]
        dy = self.hold_coords[:,1,None] - self.hold_coords[:,1]
        distances = np.sqrt(dx**2 + dy**2)
        np.fill_diagonal(distances, 0)
        return distances

    # --- Physically informed reachability helpers ---
    def _parse_limbs_tuple(self, limbs_tuple: Tuple) -> Dict[str, Tuple[float, float] or None]:
        """Convert cached limbs tuple (('LH', id), ...) into a mapping limb -> (x,y) or None"""
        positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
        for limb, hid in limbs_tuple:
            if hid is None or hid == -1:
                positions[limb] = None
                continue
            hold = self.hold_dict.get(hid)
            if hold:
                positions[limb] = (hold['x'], hold['y'])
            else:
                positions[limb] = None
        return positions

    def _estimate_hip(self, foot_positions: List[Tuple[float, float]]):
        return _kin_estimate_hip(foot_positions)

    def _estimate_shoulder(self, hip_xy: Tuple[float, float]):
        return _kin_estimate_shoulder(hip_xy)

    def _ellipse_dnorm(self, hold_xy: Tuple[float, float], shoulder_xy: Tuple[float, float], rx: float, ry: float) -> float:
        return _kin_ellipse_dnorm(hold_xy, shoulder_xy, rx, ry)

    def _convex_hull(self, points: List[Tuple[float, float]]):
        return _kin_convex_hull(points)

    def _point_in_poly(self, pt: Tuple[float, float], poly: List[Tuple[float, float]]) -> bool:
        return _kin_point_in_poly(pt, poly)

    def _compute_hand_reachability_matrix(self) -> np.ndarray:
        n = len(self.holds)
        reachable = np.zeros((n, n), dtype=bool)

        #make sure holds are within max reach
        is_hand = np.array([h['role_id'] in {12, 13, 14} for h in self.holds])
        reachable = (self.hold_distances <= MAX_HAND_REACH) & is_hand[:,None] & is_hand

        #marks start and finish 
        start_indices = [i for i, h in enumerate(self.holds) if h['role_id'] == 12]
        finish_indices = [i for i, h in enumerate(self.holds) if h['role_id'] == 14]

        #making sure you start at the start holds
        for j in start_indices:
            reachable[:,j] = False
            for i in start_indices:
                reachable[i,j] = True

        #end on finish 
        for i in finish_indices:
            reachable[i,:] = False
            for j in finish_indices:
                reachable[i,j] = True

        return reachable

    def _compute_foot_reachability_matrix(self) -> np.ndarray:
        """Compute which holds are reachable by feet from each hold"""
        n = len(self.holds)
        reachable = np.zeros((n, n), dtype=bool)

        #compute all reachable holds within the max distance
        reachable = self.hold_distances <= MAX_FOOT_REACH
        for i in range(n):
            for j in range(n):
                #makes higher holds unreachaable 
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
        # creates a grid of each hold based on the Y coords
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
        # ensures that the hand transition is valid based on constraints
        next_hold = self.hold_dict[next_id]

        # consider only valid holds
        if next_hold['role_id'] not in {12, 13, 14}:
            return False

        # checks if hold is occupied
        for other_limb, hold_id in current_limbs:
            if other_limb != limb and hold_id == next_id:
                return False

        # dynamic reachability: shoulder-centric anisotropic ellipse + COM/stability
        positions = self._parse_limbs_tuple(current_limbs)
        foot_positions = [p for k, p in positions.items() if k in ('RF', 'LF') and p is not None]

        # if we don't have feet planted, fall back to coarse matrix check
        hip = self._estimate_hip(foot_positions)
        if hip is None:
            if current_id != -1:
                try:
                    i = self.hold_ids.index(current_id)
                    j = self.hold_ids.index(next_id)
                    if not self.hand_reach_matrix[i, j]:
                        return False
                except ValueError:
                    return False
            return True

        shoulder = self._estimate_shoulder(hip)
        rx = MAX_HAND_REACH * 0.9
        ry = MAX_HAND_REACH * 0.6 + (X_SPACING * 1.2)

        hold_xy = (next_hold['x'], next_hold['y'])
        dnorm = self._ellipse_dnorm(hold_xy, shoulder, rx, ry)
        dynamic_allow = 1.15
        if dnorm > dynamic_allow:
            return False

        # stability: check COM projection vs support polygon
        com_offset = X_SPACING * 1.2
        com = (hip[0], hip[1] + com_offset)

        support_pts = list(foot_positions)
        for k in ('RH', 'LH'):
            if positions.get(k) is not None and k != limb:
                support_pts.append(positions[k])
        support_pts.append(hold_xy)

        if len(support_pts) >= 3:
            hull = self._convex_hull(support_pts)
            inside = self._point_in_poly(com, hull)
            if not inside and dnorm > 1.05:
                return False

        return True

    def is_valid_foot_transition(self, current_id: int, next_id: int, limb: str, current_limbs: Tuple) -> bool:
        """Check if a foot transition is valid based on reach and other constraints"""
        next_hold = self.hold_dict[next_id]
        
        #checks if hold is occupied 
        for other_limb, hold_id in current_limbs:
            if other_limb != limb and hold_id == next_id:
                return False
        
        #checks reachability
        if current_id != -1:  
            i = self.hold_ids.index(current_id)
            j = self.hold_ids.index(next_id)
            if not self.foot_reach_matrix[i, j]:
                return False
        
        return True

    def evaluate_sequence(self, sequence: List[Dict]) -> Dict:
        """Evaluate a climbing sequence using additional stability metrics.

        Adds `triangular_support` and `hand_support` metrics, strengthens
        cross penalties, and normalizes scores by hand moves so beam
        search prefers stable hand-centric sequences.
        """
        if not sequence:
            return {'score': 0, 'details': {}}

        metrics = {
            'hold_quality': 0.0,
            'hold_alignment': 0.0,
            'movement_efficiency': 0.0,
            'limb_alternation': 0.0,
            'body_position': 0.0,
            'cross_prevention': 0.0,
            'triangular_support': 0.0,
            'hand_support': 0.0,
            'completion': 0.0,
            'coverage': 0.0,
            'foot_tension_recovery': 0.0,
            'flag_quality': 0.0,
        }

        limb_use = defaultdict(int)
        prev_limb = None
        limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
        # Track simulated cut feet to score replant behavior. Mirrors the
        # generator's beam-state cut_feet semantics so post-hoc evaluation
        # rewards re-planting and penalizes proceeding with hands while cut.
        cut_feet = set()
        steps_since_cut = {}  # foot -> moves elapsed since cut
        # Parallel: flag state (off-hold but valid stance).
        flags = {'RF': None, 'LF': None}

        for i, move in enumerate(sequence):
            hold = self.hold_dict.get(move['hold'])
            if not hold:
                continue
            limb = move['limb']

            # Foot-cut bookkeeping (BEFORE updating limb_positions).
            if FOOT_CUT_ENABLED:
                if limb in ('RH', 'LH'):
                    prev_hand_xy = limb_positions[limb]
                    new_hand_xy = (hold['x'], hold['y'])
                    foot_xys = [limb_positions[f] for f in ('RF', 'LF')
                                if limb_positions[f] is not None]
                    other_hand = 'LH' if limb == 'RH' else 'RH'
                    other_hand_xy = limb_positions[other_hand]
                    if cut_feet:
                        # Climber proceeded with a hand while a foot was cut.
                        metrics['foot_tension_recovery'] -= 3.0
                    cut_happened = False
                    if is_dynamic_hand_move(prev_hand_xy, new_hand_xy, foot_xys,
                                            limb, other_hand_xy):
                        foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                        cut = choose_cut_foot(limb, foot_dict)
                        if cut is not None:
                            cut_feet.add(cut)
                            steps_since_cut[cut] = 0
                            limb_positions[cut] = None
                            flags[cut] = None
                            cut_happened = True
                    # Phantom-flag trigger (only when not cutting).
                    if FLAG_ENABLED and not cut_happened:
                        if is_crossing_hand_move(new_hand_xy, other_hand_xy, limb):
                            foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                            flag_foot = choose_flag_foot(limb, foot_dict)
                            if flag_foot is not None:
                                planted = [v for v in foot_dict.values() if v is not None]
                                hip = _kin_estimate_hip(planted)
                                if hip is not None:
                                    flag_xy = compute_flag_position(hip, new_hand_xy, flag_foot)
                                    flags[flag_foot] = flag_xy
                                    limb_positions[flag_foot] = None
                                    # Reward: flag on opposite side of moving hand.
                                    side_hand = 1 if new_hand_xy[0] >= hip[0] else -1
                                    side_flag = 1 if flag_xy[0] >= hip[0] else -1
                                    if side_hand != side_flag:
                                        metrics['flag_quality'] += 1.0
                                    else:
                                        metrics['flag_quality'] += 0.3
                elif limb in ('RF', 'LF') and limb in cut_feet:
                    elapsed = steps_since_cut.get(limb, 0)
                    if elapsed <= FOOT_REPLANT_MAX_STEPS:
                        metrics['foot_tension_recovery'] += 2.0
                    cut_feet.discard(limb)
                    steps_since_cut.pop(limb, None)
                    flags[limb] = None
                elif limb in ('RF', 'LF') and flags.get(limb) is not None:
                    # Re-plant from a flag.
                    flags[limb] = None
                    metrics['flag_quality'] += 0.5
                # Age remaining cuts.
                for f in list(steps_since_cut.keys()):
                    steps_since_cut[f] += 1

            limb_positions[limb] = (hold['x'], hold['y'])

            # hold quality (CV-derived from board image; neutral 0.7 fallback).
            q = get_hold_quality(hold['hole_id'])
            if hold['role_id'] == 14:
                metrics['hold_quality'] += 10
            elif hold['role_id'] == 12:
                metrics['hold_quality'] += 5
            elif limb in ('RH', 'LH'):
                metrics['hold_quality'] += q
            else:
                metrics['hold_quality'] += 0.5 * q

            # Hand-pull alignment: compare pull-vector (hand - shoulder) to
            # the hold's edge-tangent direction. Cosine in [-1, 1]; we keep
            # positive contributions and clip negatives to penalize pulling
            # against a hold's natural direction.
            if limb in ('RH', 'LH'):
                hold_dir = get_hold_direction(hold['hole_id'])
                feet_xys = [limb_positions[f] for f in ('RF', 'LF')
                            if limb_positions[f] is not None]
                if hold_dir is not None and feet_xys:
                    hip = _kin_estimate_hip(feet_xys)
                    shoulder = _kin_estimate_shoulder(hip) if hip else None
                    if shoulder is not None:
                        pdx = hold['x'] - shoulder[0]
                        pdy = hold['y'] - shoulder[1]
                        pmag = math.hypot(pdx, pdy)
                        if pmag > 1e-6:
                            pull_deg = math.degrees(math.atan2(pdy, pdx))
                            # Edge tangent is axis-symmetric (±180 equivalent);
                            # take absolute cosine of the doubled-angle diff,
                            # then map back via cos(diff)^2 → align ∈ [-1, 1].
                            diff = math.radians(pull_deg - hold_dir)
                            align = math.cos(diff)
                            # Push axis-symmetry: a sidepull aligned 180° away
                            # is just as compatible as 0°.
                            align = max(align, math.cos(diff + math.pi))
                            metrics['hold_alignment'] += align

            # movement efficiency (shorter moves preferred)
            if i > 0:
                prev_hold = self.hold_dict.get(sequence[i - 1]['hold'])
                if prev_hold:
                    dist = math.hypot(hold['x'] - prev_hold['x'], hold['y'] - prev_hold['y'])
                    metrics['movement_efficiency'] += 1.0 / (dist + 0.1)

            # limb alternation for hands
            if limb in ('RH', 'LH') and prev_limb in ('RH', 'LH') and limb != prev_limb:
                metrics['limb_alternation'] += 1

            # stronger crossing penalties
            if limb_positions['RH'] is not None and limb_positions['LH'] is not None:
                if limb_positions['RH'][0] < limb_positions['LH'][0]:
                    metrics['cross_prevention'] -= 8

            if limb_positions['RF'] is not None and limb_positions['LF'] is not None:
                if limb_positions['RF'][0] < limb_positions['LF'][0]:
                    metrics['cross_prevention'] -= 4

            # triangular support: reward having 3+ limbs on holds
            filled = [p for p in limb_positions.values() if p is not None]
            if len(filled) >= 3:
                metrics['triangular_support'] += len(filled) - 2

            # hand_support: when a hand reaches, check complementary limbs
            if limb == 'LH':
                # LH reach: ideally RH and LF are planted
                if limb_positions['RH'] is not None and limb_positions['LF'] is not None:
                    metrics['hand_support'] += 1
                else:
                    metrics['hand_support'] -= 1
            if limb == 'RH':
                # RH reach: ideally LH and RF are planted
                if limb_positions['LH'] is not None and limb_positions['RF'] is not None:
                    metrics['hand_support'] += 1
                else:
                    metrics['hand_support'] -= 1

            # body position: hands above feet and COG inside foot span
            if limb_positions['RF'] is not None and limb_positions['LF'] is not None and limb_positions['RH'] is not None and limb_positions['LH'] is not None:
                cog_x = sum(p[0] for p in limb_positions.values()) / 4.0
                min_foot_x = min(limb_positions['RF'][0], limb_positions['LF'][0])
                max_foot_x = max(limb_positions['RF'][0], limb_positions['LF'][0])
                if min_foot_x <= cog_x <= max_foot_x:
                    metrics['body_position'] += 2
                # hands higher than feet
                if min(limb_positions['RH'][1], limb_positions['LH'][1]) > max(limb_positions['RF'][1], limb_positions['LF'][1]):
                    metrics['body_position'] += 1

            prev_limb = limb
            limb_use[limb] += 1

        # completion bonus: last move on a finish hold
        last_move = sequence[-1]
        metrics['completion'] = 10.0 if self.hold_dict.get(last_move['hold'], {}).get('role_id') == 14 else 0.0

        # coverage: fraction of required hand-role holds visited by a hand.
        required_hand_ids = {h['hole_id'] for h in self.hand_holds}
        if required_hand_ids:
            visited_hand_ids = {m['hold'] for m in sequence
                                if m['limb'] in ('RH', 'LH')}
            metrics['coverage'] = 10.0 * (len(required_hand_ids & visited_hand_ids)
                                          / len(required_hand_ids))

        # normalize metrics
        seq_len = len(sequence)
        if seq_len > 1:
            metrics['movement_efficiency'] /= (seq_len - 1)

        hand_moves = sum(1 for m in sequence if m['limb'] in ('RH', 'LH'))
        if hand_moves > 1:
            metrics['limb_alternation'] /= max(1, (hand_moves - 1))
        # Normalize alignment so length doesn't dominate; keep ~[-1, 1] range.
        if hand_moves > 0:
            metrics['hold_alignment'] /= hand_moves

        # keep cross_prevention non-negative (higher is better)
        metrics['cross_prevention'] = max(0.0, metrics['cross_prevention'] + 8.0)

        # assemble weights
        weights = {
            'hold_quality': 0.07,
            'hold_alignment': 0.08,
            'movement_efficiency': 0.13,
            'limb_alternation': 0.12,
            'body_position': 0.10,
            'cross_prevention': 0.07,
            'triangular_support': 0.10,
            'hand_support': 0.10,
            'completion': 0.10,
            'coverage': 0.10,
            'foot_tension_recovery': 0.05,
            'flag_quality': 0.05,
        }

        total_score = 0.0
        for k, v in metrics.items():
            w = weights.get(k, 0.0)
            total_score += v * w

        return {
            'score': total_score,
            'details': metrics,
            'limb_balance': dict(limb_use),
            'sequence_length': seq_len,
        }

    def generate_sequences(self, beam_width: int = None) -> Dict[str, Any]:
        #generate sequences using beam
        beam_width = beam_width or BEAM_WIDTH
        
        self._start_timer('total_search')
        beam = self._initialize_beam()
        completed_sequences = []
        iterations = 0
        MAX_ITERATIONS = 100  

        with tqdm(desc="Generating sequences", unit="iter") as pbar:
            while beam and iterations < MAX_ITERATIONS:
                #split for parallel processing
                with ThreadPoolExecutor(max_workers=self.num_workers) as executor:
                    chunk_size = max(1, len(beam) // (self.num_workers * 2))
                    futures = [
                        executor.submit(self._expand_states, beam[i:i+chunk_size], beam_width)
                        for i in range(0, len(beam), chunk_size)
                    ]
                    next_beam = list(chain(*[f.result() for f in futures]))

                #keep the best states for the next iteration
                beam = heapq.nlargest(
                    beam_width * 5,
                    next_beam,
                    key=lambda x: (x['score'], -len(x['sequence']))
                )[:beam_width]

                #check for completed sequences
                for state in beam:
                    if self._is_complete(state):
                        evaluation = self.evaluate_sequence(state['sequence'])
                        completed_sequences.append({
                            'sequence': state['sequence'],
                            'evaluation': evaluation
                        })
                iterations += 1
                pbar.update(1)
                
                #stop if there are completed sequences
                if len(completed_sequences) >= beam_width * 2:
                    break

        self._stop_timer('total_search')

        if not completed_sequences:
            return {'status': 'error', 'message': 'No valid sequences found'}

        #sort by evaluation score
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
            #skip completed states
            if self._is_complete(state):
                new_states.append(state)
                continue

            #convert limbs dict to tuple
            current_limbs_tuple = tuple(sorted(
                (limb, hold['hole_id'] if hold else -1) 
                for limb, hold in state['limbs'].items()
            ))

            cut_feet = state.get('cut_feet', frozenset())
            flags = state.get('flags', {'RF': None, 'LF': None})
            # While any foot is cut by a prior dynamic hand move, the climber
            # must re-plant it before any further hand reach. Mask out hand
            # expansions entirely until cut_feet is empty. Flags do NOT block
            # hand moves (a flag is a valid stance).
            hands_allowed = (not cut_feet) or (not FOOT_CUT_ENABLED)

            # Pre-compute hands already visited so we can reward first-time coverage.
            visited_hand_ids = {m['hold'] for m in state['sequence']
                                if m['limb'] in ('RH', 'LH')}
            finish_ids = {h['hole_id'] for h in self.finish_holds}
            required_non_finish = {h['hole_id'] for h in self.hand_holds} - finish_ids

            if hands_allowed:
                for limb in ['RH', 'LH']:
                    current = state['limbs'][limb]
                    current_id = current['hole_id'] if current else -1
                    if current_id != -1:
                        visited_hand_ids.add(current_id)

                    # Kilter convention: finish is touched last. Don't leave a finish hold,
                    # and don't grab the finish until all other required hands are covered.
                    if current_id in finish_ids:
                        continue
                    coverage_done = required_non_finish.issubset(visited_hand_ids)

                    for next_hold in self.hand_holds:
                        next_id = next_hold['hole_id']
                        if next_id in finish_ids and not coverage_done:
                            continue
                        if current_id != next_id and self.is_valid_hand_transition(
                                current_id, next_id, limb, current_limbs_tuple):
                            #new state with this hand movement
                            new_limbs = {k: v for k, v in state['limbs'].items()}
                            new_limbs[limb] = next_hold

                            #calculate state score based on hold and quality
                            move_score = 1
                            if next_hold['role_id'] == 14:
                                move_score = 10
                            elif next_hold['role_id'] == 12:
                                move_score = 5
                            # Coverage bonus: first time this hand-role hold is touched.
                            if next_id not in visited_hand_ids:
                                move_score += 8

                            # Quality bias: prefer high-quality (jug/crimp) holds.
                            q = get_hold_quality(next_id)
                            move_score *= (0.7 + 0.6 * q)  # q=0.5 -> 1.0x, q=1.0 -> 1.3x

                            # Detect dynamic move and cut a foot if so.
                            new_cut = cut_feet
                            new_flags = dict(flags)
                            cut_happened = False
                            if FOOT_CUT_ENABLED:
                                prev_hand_xy = (current['x'], current['y']) if current else None
                                new_hand_xy = (next_hold['x'], next_hold['y'])
                                foot_xys = [(state['limbs'][f]['x'], state['limbs'][f]['y'])
                                            for f in ('RF', 'LF')
                                            if state['limbs'].get(f) is not None]
                                other_hand = 'LH' if limb == 'RH' else 'RH'
                                other = state['limbs'].get(other_hand)
                                other_hand_xy = (other['x'], other['y']) if other else None
                                if is_dynamic_hand_move(prev_hand_xy, new_hand_xy, foot_xys,
                                                        limb, other_hand_xy):
                                    foot_dict = {f: ((state['limbs'][f]['x'], state['limbs'][f]['y'])
                                                     if state['limbs'].get(f) else None)
                                                 for f in ('RF', 'LF')}
                                    cut = choose_cut_foot(limb, foot_dict)
                                    if cut is not None:
                                        new_limbs[cut] = None
                                        new_cut = frozenset({cut})
                                        new_flags[cut] = None  # cut clears any pre-existing flag
                                        cut_happened = True

                            # Phantom-flag trigger: if the move crosses the body line
                            # (and we did NOT just cut), put the opposite foot into a
                            # flag state. This is a soft stance — no penalty for
                            # continuing with other hand moves later.
                            if FLAG_ENABLED and not cut_happened:
                                other_hand = 'LH' if limb == 'RH' else 'RH'
                                other = state['limbs'].get(other_hand)
                                other_hand_xy = (other['x'], other['y']) if other else None
                                new_hand_xy = (next_hold['x'], next_hold['y'])
                                if is_crossing_hand_move(new_hand_xy, other_hand_xy, limb):
                                    foot_dict = {f: ((state['limbs'][f]['x'], state['limbs'][f]['y'])
                                                     if state['limbs'].get(f) else None)
                                                 for f in ('RF', 'LF')}
                                    flag_foot = choose_flag_foot(limb, foot_dict)
                                    if flag_foot is not None:
                                        # Estimate hip from currently planted feet.
                                        foot_xys = [v for v in foot_dict.values() if v is not None]
                                        hip = _kin_estimate_hip(foot_xys)
                                        if hip is not None:
                                            new_flags[flag_foot] = compute_flag_position(
                                                hip, new_hand_xy, flag_foot,
                                            )
                                            # Foot leaves the hold; cut_feet stays empty.
                                            new_limbs[flag_foot] = None
                                            # Small reward for a balanced flag.
                                            move_score += 0.5

                            new_states.append({
                                'sequence': state['sequence'] + [{
                                    'limb': limb,
                                    'hold': next_id,
                                    'position': next_hold.get('position')
                                }],
                                'limbs': new_limbs,
                                'cut_feet': new_cut,
                                'flags': new_flags,
                                'score': state['score'] + move_score
                            })

            #feeeet
            for limb in ['RF', 'LF']:
                # If feet are cut, only the cut foot may move (replant gate).
                if cut_feet and FOOT_CUT_ENABLED and limb not in cut_feet:
                    continue

                current = state['limbs'].get(limb, None)
                current_id = current['hole_id'] if current else -1

                for next_hold in self.foot_holds:
                    next_id = next_hold['hole_id']
                    if current_id != next_id and self.is_valid_foot_transition(
                            current_id, next_id, limb, current_limbs_tuple):
                        #new state
                        new_limbs = {k: v for k, v in state['limbs'].items()}
                        new_limbs[limb] = next_hold

                        # Re-plant clears this foot from cut_feet AND any flag.
                        new_cut = cut_feet - {limb} if cut_feet else cut_feet
                        new_flags = dict(flags)
                        was_flagging = new_flags.get(limb) is not None
                        new_flags[limb] = None

                        #lower scores for foot movements; small bonus when this
                        #move is a replant restoring tension.
                        move_score = 0.5
                        if cut_feet and limb in cut_feet:
                            move_score += 2.0
                        if was_flagging:
                            move_score += 1.0  # reward landing from a flag

                        new_states.append({
                            'sequence': state['sequence'] + [{
                                'limb': limb,
                                'hold': next_id,
                                'position': next_hold.get('position')
                            }],
                            'limbs': new_limbs,
                            'cut_feet': new_cut,
                            'flags': new_flags,
                            'score': state['score'] + move_score
                        })

        return new_states

    def _initialize_beam(self):
        #initilize with start positions
        beam = []

        def _start_seq(rh_hold, lh_hold):
            # Emit the matched-start as the first two recorded moves so the
            # rendered sequence begins on the start holds.
            return [
                {'limb': 'RH', 'hold': rh_hold['hole_id'],
                 'position': rh_hold.get('position')},
                {'limb': 'LH', 'hold': lh_hold['hole_id'],
                 'position': lh_hold.get('position')},
            ]

        #just hands on start holds
        for rh in self.start_holds:
            for lh in self.start_holds:
                if rh != lh:  # Different holds for each hand
                    # Initial state with just hands positioned
                    beam.append({
                        'sequence': _start_seq(rh, lh),
                        'limbs': {
                            'RH': rh, 
                            'LH': lh,
                            'RF': None,
                            'LF': None
                        },
                        'cut_feet': frozenset(),
                        'flags': {'RF': None, 'LF': None},
                        'score': 0
                    })
        
        #allowing the same hold for both hands
        if not beam and self.start_holds:
            for h in self.start_holds:
                beam.append({
                    'sequence': _start_seq(h, h),
                    'limbs': {
                        'RH': h, 
                        'LH': h,
                        'RF': None,
                        'LF': None
                    },
                    'cut_feet': frozenset(),
                    'flags': {'RF': None, 'LF': None},
                    'score': 0
                })
        
        #check for starting positions
        if not beam:
            raise ValueError("Could not create valid starting positions")
            
        return beam

    def _score_initial_position(self, state):
        """Score initial positions based on naturalness"""
        limbs = state['limbs']
        if not all(limb in limbs for limb in ['RH', 'LH']):
            return 0
            
        hand_width = 0
        if limbs['RH'] and limbs['LH']:
            hand_width = abs(limbs['RH']['x'] - limbs['LH']['x'])
            
        foot_width = 0
        if limbs.get('RF') and limbs.get('LF'):
            foot_width = abs(limbs['RF']['x'] - limbs['LF']['x'])
            
        #checks if hands are above feet
        hands_above_feet = False
        if all(limbs.get(limb) for limb in ['RH', 'LH', 'RF', 'LF']):
            min_hand_y = min(limbs['RH']['y'], limbs['LH']['y'])
            max_foot_y = max(limbs['RF']['y'], limbs['LF']['y'])
            hands_above_feet = min_hand_y > max_foot_y
            
        score = hand_width + foot_width + (10 if hands_above_feet else 0)
        return score

    def _is_complete(self, state):
        limbs = state['limbs']
        # Finish is determined only by hands occupying finish holds
        finish_ids = {h['hole_id'] for h in self.finish_holds}
        hand_ids = {limbs['RH']['hole_id'] if limbs['RH'] else None,
                    limbs['LH']['hole_id'] if limbs['LH'] else None}
        hand_ids.discard(None)

        if not finish_ids:
            return False

        # Finish condition by hand occupancy.
        if len(finish_ids) == 1:
            finish_ok = any(h in finish_ids for h in hand_ids)
        else:
            finish_ok = finish_ids.issubset(hand_ids)
        if not finish_ok:
            return False

        # Kilter convention: every painted hand-role hold (start/hand/finish)
        # must be touched by a hand at some point in the route.
        required_hand_ids = {h['hole_id'] for h in self.hand_holds}
        visited_hand_ids = {m['hold'] for m in state['sequence']
                            if m['limb'] in ('RH', 'LH')}
        visited_hand_ids |= hand_ids
        return required_hand_ids.issubset(visited_hand_ids)

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