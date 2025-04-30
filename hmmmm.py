import json
import numpy as np
import sqlite3
import random
import os
from typing import List, Dict, Any, Tuple, Optional
from collections import defaultdict, Counter
import tensorflow as tf
from tensorflow import keras
from keras import layers
import matplotlib.pyplot as plt
from tqdm.auto import tqdm
from sequence_generator import ClimbSequenceGenerator
from config import *


class ClimbGenerativeModel:
    """
    A generative model for creating new climbing routes based on successful
    sequences from existing routes.
    """
    
    def __init__(self, db_path: str, results_file: str):
        """
        Initialize the generative model with paths to the database and results file.
        
        Args:
            db_path: Path to the SQLite database
            results_file: Path to the JSON file with climbing sequences
        """
        self.db_path = db_path
        self.results_file = results_file
        
        # Data structures to be populated during training
        self.holds_distribution = {}  # Distribution of holds by role type
        self.hold_spacing_model = {}  # Typical spacing between holds
        self.sequence_patterns = {}   # Common sequence patterns
        self.transition_probs = {}    # Transition probabilities between holds
        self.difficulty_model = None  # Model to predict difficulty
        
        # Load data from the results file
        self.load_results_data()
        
        # Connect to the database to get hold information
        self.conn = sqlite3.connect(db_path)
        self.conn.row_factory = sqlite3.Row
        
        # Load the board layout (all possible hold positions)
        self.board_layout = self._load_board_layout()
        
        print(f"Initialized model with {len(self.climbs)} climbs and {len(self.board_layout)} board positions")
    
    def load_results_data(self):
        """Load and parse the climb results data"""
        with open(self.results_file, 'r') as f:
            data = json.load(f)
        
        # Filter to only successful results
        self.climbs = [
            r for r in data['results'] 
            if 'best_sequence' in r and 'error' not in r
        ]
        
        print(f"Loaded {len(self.climbs)} successful climbs from results file")
    
    def _load_board_layout(self) -> List[Dict]:
        """Load the board layout with all possible hold positions"""
        try:
            cursor = self.conn.execute("""
                SELECT h.id as hole_id, h.x, h.y, h.name
                FROM holes h
                JOIN placements p ON h.id = p.hole_id
                WHERE p.layout_id = 1
                GROUP BY h.id
            """)
            
            layout = []
            for row in cursor.fetchall():
                layout.append({
                    'hole_id': row['hole_id'],
                    'x': row['x'],
                    'y': row['y'],
                    'name': row['name']
                })
            return layout
        except Exception as e:
            print(f"Error loading board layout: {e}")
            return []
    
    def analyze_climbs(self):
        """Analyze the climbs dataset to extract patterns and statistics"""
        print("Analyzing climbs to extract patterns...")
        
        # Initialize counters
        role_counts = defaultdict(int)
        hold_counts = defaultdict(int)
        transitions = defaultdict(lambda: defaultdict(int))
        limb_transitions = defaultdict(lambda: defaultdict(int))
        hold_spacings = []
        
        # Analyze each climb
        for climb in tqdm(self.climbs, desc="Analyzing climbs"):
            holds = {h['hole_id']: h for h in climb.get('holds', [])}
            
            # Skip if no holds data
            if not holds:
                continue
                
            # Count roles
            for h in holds.values():
                role_id = h.get('role_id')
                role_counts[role_id] += 1
                hold_counts[h['hole_id']] += 1
            
            # Analyze sequence
            sequence = climb.get('best_sequence', {}).get('sequence', [])
            prev_hold = None
            prev_limb = None
            
            for move in sequence:
                hold_id = move.get('hold')
                limb = move.get('limb')
                
                # Skip invalid data
                if not hold_id or not limb or hold_id not in holds:
                    continue
                
                # Analyze transitions
                if prev_hold is not None:
                    transitions[prev_hold][hold_id] += 1
                    
                    # Calculate spatial distance
                    if prev_hold in holds and hold_id in holds:
                        x1, y1 = holds[prev_hold]['x'], holds[prev_hold]['y']
                        x2, y2 = holds[hold_id]['x'], holds[hold_id]['y']
                        distance = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                        hold_spacings.append((distance, limb[:1]))  # 'H' for hands, 'F' for feet
                
                # Analyze limb transitions
                if prev_limb is not None:
                    limb_transitions[prev_limb][limb] += 1
                
                prev_hold = hold_id
                prev_limb = limb
        
        # Store the distributions
        self.holds_distribution = {
            'roles': dict(role_counts),
            'popularity': dict(hold_counts)
        }
        
        # Convert transitions to probabilities
        self.transition_probs = {
            'holds': self._normalize_transitions(transitions),
            'limbs': self._normalize_transitions(limb_transitions)
        }
        
        # Analyze hold spacings
        hand_spacings = [d for d, limb_type in hold_spacings if limb_type == 'R' or limb_type == 'L']
        foot_spacings = [d for d, limb_type in hold_spacings if limb_type == 'F']
        
        self.hold_spacing_model = {
            'hands': {
                'mean': np.mean(hand_spacings) if hand_spacings else 0,
                'std': np.std(hand_spacings) if hand_spacings else 0
            },
            'feet': {
                'mean': np.mean(foot_spacings) if foot_spacings else 0,
                'std': np.std(foot_spacings) if foot_spacings else 0
            }
        }
        
        # Extract sequence patterns
        self._extract_sequence_patterns()
        
        # Train difficulty prediction model
        self._train_difficulty_model()
        
        print("Analysis complete!")
        print(f"Hold roles: {self.holds_distribution['roles']}")
        print(f"Average hand spacing: {self.hold_spacing_model['hands']['mean']:.2f} ± {self.hold_spacing_model['hands']['std']:.2f}")
        print(f"Average foot spacing: {self.hold_spacing_model['feet']['mean']:.2f} ± {self.hold_spacing_model['feet']['std']:.2f}")
    
    def _normalize_transitions(self, transitions_dict):
        """Convert counts to probabilities"""
        normalized = {}
        for source, targets in transitions_dict.items():
            total = sum(targets.values())
            if total > 0:
                normalized[source] = {t: count/total for t, count in targets.items()}
        return normalized
    
    def _extract_sequence_patterns(self):
        """Extract common patterns in move sequences"""
        limb_patterns = []
        
        # Extract limb patterns from sequences
        for climb in self.climbs:
            sequence = climb.get('best_sequence', {}).get('sequence', [])
            if len(sequence) < 3:
                continue
                
            # Extract limb patterns (e.g., RH -> LH -> RF -> LF)
            limbs = [move.get('limb') for move in sequence if move.get('limb')]
            
            # Consider pattern length from 2 to 4
            for i in range(len(limbs)-1):
                if i+1 < len(limbs):
                    limb_patterns.append((limbs[i], limbs[i+1]))
                if i+2 < len(limbs):
                    limb_patterns.append((limbs[i], limbs[i+1], limbs[i+2]))
                if i+3 < len(limbs):
                    limb_patterns.append((limbs[i], limbs[i+1], limbs[i+2], limbs[i+3]))
        
        # Count patterns
        pattern_counter = Counter(limb_patterns)
        total_patterns = sum(pattern_counter.values())
        
        # Convert to probabilities
        self.sequence_patterns = {
            pattern: count/total_patterns for pattern, count in pattern_counter.most_common(50)
        }
    
    def _train_difficulty_model(self):
        """Train a model to predict difficulty based on climb features"""
        # Extract features
        X = []
        y = []
        
        for climb in self.climbs:
            try:
                # Get evaluation score as a proxy for difficulty
                score = climb.get('best_sequence', {}).get('evaluation', {}).get('score', 0)
                
                if score == 0:
                    continue
                
                # Extract features
                sequence = climb.get('best_sequence', {}).get('sequence', [])
                
                # Skip if sequence is too short
                if len(sequence) < 3:
                    continue
                
                # Features
                num_holds = len(climb.get('holds', []))
                num_moves = len(sequence)
                start_holds = sum(1 for h in climb.get('holds', []) if h.get('role_id') == 12)
                finish_holds = sum(1 for h in climb.get('holds', []) if h.get('role_id') == 14)
                
                # Vertical distance (height gain)
                holds_by_id = {h['hole_id']: h for h in climb.get('holds', [])}
                hold_positions = [(holds_by_id.get(move.get('hold'), {}).get('x', 0),
                                  holds_by_id.get(move.get('hold'), {}).get('y', 0))
                                 for move in sequence if move.get('hold') in holds_by_id]
                
                if hold_positions:
                    min_y = min(y for _, y in hold_positions)
                    max_y = max(y for _, y in hold_positions)
                    height_gain = max_y - min_y
                else:
                    height_gain = 0
                
                # Calculate average hold spacing
                spacings = []
                for i in range(len(hold_positions)-1):
                    x1, y1 = hold_positions[i]
                    x2, y2 = hold_positions[i+1]
                    spacing = np.sqrt((x2-x1)**2 + (y2-y1)**2)
                    spacings.append(spacing)
                
                avg_spacing = np.mean(spacings) if spacings else 0
                
                # Combine features
                features = [
                    num_holds,
                    num_moves,
                    start_holds,
                    finish_holds,
                    height_gain,
                    avg_spacing
                ]
                
                X.append(features)
                y.append(score)
                
            except Exception as e:
                print(f"Error extracting features: {e}")
                continue
        
        # Train a simple linear model if we have enough data
        if len(X) > 10:
            X = np.array(X)
            y = np.array(y)
            
            # Normalize features
            X_mean = np.mean(X, axis=0)
            X_std = np.std(X, axis=0)
            X_std[X_std == 0] = 1  # Avoid division by zero
            X_norm = (X - X_mean) / X_std
            
            # Train a simple neural network
            model = keras.Sequential([
                layers.Dense(10, activation='relu', input_shape=(X_norm.shape[1],)),
                layers.Dense(5, activation='relu'),
                layers.Dense(1)
            ])
            
            model.compile(optimizer='adam', loss='mse')
            model.fit(X_norm, y, epochs=50, batch_size=8, verbose=0)
            
            # Store the model and normalization parameters
            self.difficulty_model = {
                'model': model,
                'X_mean': X_mean,
                'X_std': X_std
            }
            
            print("Trained difficulty prediction model!")
        else:
            print("Not enough data to train difficulty model")
    
    def generate_climb(self, 
                      num_holds: int = 15, 
                      target_difficulty: float = None,
                      min_height: float = None,
                      max_height: float = None) -> Dict:
        """
        Generate a new climbing route with specified parameters.
        
        Args:
            num_holds: Number of holds to include
            target_difficulty: Target difficulty score (if None, will generate a random difficulty)
            min_height: Minimum height for holds (if None, will use full board)
            max_height: Maximum height for holds (if None, will use full board)
            
        Returns:
            Dictionary containing the generated climb data
        """
        print(f"Generating new climb with {num_holds} holds...")
        
        # Filter holds by height if specified
        available_holds = self.board_layout
        if min_height is not None or max_height is not None:
            min_y = min_height if min_height is not None else float('-inf')
            max_y = max_height if max_height is not None else float('inf')
            available_holds = [h for h in available_holds if min_y <= h['y'] <= max_y]
        
        # Ensure we have enough holds
        if len(available_holds) < num_holds:
            raise ValueError(f"Not enough holds in the specified height range. Found {len(available_holds)}, needed {num_holds}")
        
        # Generate hold distribution
        selected_holds = self._generate_hold_distribution(available_holds, num_holds)
        
        # Assign roles to holds
        holds_with_roles = self._assign_hold_roles(selected_holds)
        
        # Validate the generated climb
        if not self._validate_generated_climb(holds_with_roles):
            # Recursive retry if validation fails
            return self.generate_climb(num_holds, target_difficulty, min_height, max_height)
        
        # Create a unique ID for the climb
        climb_id = f"generated_{int(np.random.random() * 1000000)}"
        
        # Give it a creative name
        climb_name = self._generate_climb_name()
        
        return {
            'id': climb_id,
            'name': climb_name,
            'holds': holds_with_roles,
            'generated': True,
            'parameters': {
                'num_holds': num_holds,
                'target_difficulty': target_difficulty,
                'min_height': min_height,
                'max_height': max_height
            }
        }
    
    def _generate_hold_distribution(self, available_holds: List[Dict], num_holds: int) -> List[Dict]:
        """Generate a distribution of holds based on learned patterns"""
        # Cluster holds by vertical zones (bottom, middle, top)
        y_coords = [h['y'] for h in available_holds]
        min_y, max_y = min(y_coords), max(y_coords)
        zone_height = (max_y - min_y) / 3
        
        bottom_zone = [h for h in available_holds if h['y'] <= min_y + zone_height]
        middle_zone = [h for h in available_holds if min_y + zone_height < h['y'] <= min_y + 2*zone_height]
        top_zone = [h for h in available_holds if h['y'] > min_y + 2*zone_height]
        
        # Determine number of holds per zone (more at bottom and middle)
        bottom_count = int(num_holds * 0.4)
        middle_count = int(num_holds * 0.4)
        top_count = num_holds - bottom_count - middle_count
        
        # Select holds with some randomness but respecting spatial distribution
        selected_holds = []
        
        # Helper function to select holds with good spacing
        def select_holds_with_spacing(zone_holds, count, selected, min_spacing=X_SPACING*0.7):
            """Select holds with good spacing between them"""
            remaining = count
            zone_copy = zone_holds.copy()
            
            while remaining > 0 and zone_copy:
                # If this is the first hold or we want some randomness, pick randomly
                if not selected or np.random.random() < 0.3:
                    hold = random.choice(zone_copy)
                else:
                    # Otherwise try to find a hold that's well-spaced from existing holds
                    # Calculate minimum distance to any selected hold
                    hold_distances = []
                    for hold in zone_copy:
                        min_dist = float('inf')
                        for sel_hold in selected:
                            dist = np.sqrt((hold['x'] - sel_hold['x'])**2 + (hold['y'] - sel_hold['y'])**2)
                            min_dist = min(min_dist, dist)
                        hold_distances.append((hold, min_dist))
                    
                    # Sort by distance (largest first) and add some randomness
                    sorted_holds = sorted(hold_distances, key=lambda x: x[1], reverse=True)
                    
                    # Select from top 3 options if available, otherwise take the best
                    top_n = min(3, len(sorted_holds))
                    hold = sorted_holds[np.random.randint(top_n)][0]
                
                selected.append(hold)
                zone_copy.remove(hold)
                remaining -= 1
            
            return selected
        
        # Select from each zone
        selected_holds = select_holds_with_spacing(bottom_zone, bottom_count, selected_holds)
        selected_holds = select_holds_with_spacing(middle_zone, middle_count, selected_holds)
        selected_holds = select_holds_with_spacing(top_zone, top_count, selected_holds)
        
        return selected_holds
    
    def _assign_hold_roles(self, holds: List[Dict]) -> List[Dict]:
        """Assign roles to holds based on position and learned patterns"""
        # Copy holds to avoid modifying the original
        holds_with_roles = [h.copy() for h in holds]
        
        # Sort by height (y-coordinate)
        holds_with_roles.sort(key=lambda h: h['y'])
        
        # Assign start holds (lowest holds)
        num_start_holds = 2  # Typical number of start holds
        for i in range(min(num_start_holds, len(holds_with_roles))):
            holds_with_roles[i]['role_id'] = 12  # Start hold role
        
        # Assign finish holds (highest holds)
        num_finish_holds = min(2, max(1, len(holds_with_roles) // 10))  # 1-2 finish holds
        for i in range(1, num_finish_holds + 1):
            if len(holds_with_roles) - i >= 0:
                holds_with_roles[-i]['role_id'] = 14  # Finish hold role
        
        # Assign intermediate holds
        for i in range(num_start_holds, len(holds_with_roles) - num_finish_holds):
            holds_with_roles[i]['role_id'] = 13  # Intermediate hold role
        
        # Optional: Assign some foot holds
        if len(holds_with_roles) > 8:  # Only for larger problems
            foot_candidates = []
            
            # Find holds that would make good foot holds
            for i, hold in enumerate(holds_with_roles):
                if hold['role_id'] != 12 and hold['role_id'] != 14:  # Not start or finish
                    # Check if there are holds above this one
                    higher_holds = [h for h in holds_with_roles if h['y'] > hold['y'] + X_SPACING]
                    if higher_holds:
                        foot_candidates.append(i)
            
            # Convert some to foothold role
            num_foot_holds = min(len(foot_candidates), max(1, len(holds_with_roles) // 5))
            for i in random.sample(foot_candidates, num_foot_holds):
                holds_with_roles[i]['role_id'] = 15  # Foot hold role
        
        return holds_with_roles
    
    def _validate_generated_climb(self, holds: List[Dict]) -> bool:
        """Validate that the generated climb is viable"""
        # Check if we have start and finish holds
        start_holds = [h for h in holds if h.get('role_id') == 12]
        finish_holds = [h for h in holds if h.get('role_id') == 14]
        
        if not start_holds or not finish_holds:
            print("Validation failed: Missing start or finish holds")
            return False
        
        # Check if the climb is physically possible (can reach from start to finish)
        # We'll use the sequence generator to validate
        try:
            generator = ClimbSequenceGenerator(holds)
            result = generator.generate_sequences(beam_width=20)  # Use smaller beam for validation
            
            if result['status'] != 'success':
                print(f"Validation failed: {result.get('message', 'No valid sequences found')}")
                return False
                
            # Check if sequence reaches a finish hold
            sequence = result.get('best_sequence', {}).get('sequence', [])
            if not sequence:
                print("Validation failed: No sequence generated")
                return False
                
            # Check last move is to a finish hold
            last_move = sequence[-1]
            last_hold = next((h for h in holds if h.get('hole_id') == last_move.get('hold')), None)
            
            if not last_hold or last_hold.get('role_id') != 14:
                print("Validation failed: Sequence doesn't reach finish hold")
                return False
                
            return True
            
        except Exception as e:
            print(f"Validation error: {e}")
            return False
    
    def _generate_climb_name(self) -> str:
        """Generate a creative name for the climb"""
        # Lists of creative climbing-related words
        adjectives = [
            "Crimpy", "Slabby", "Dynamic", "Overhung", "Technical", "Pumpy", "Balancy",
            "Creative", "Delicate", "Powerful", "Flowy", "Tricky", "Smooth", "Exposed",
            "Bold", "Slopey", "Juggy", "Sketchy", "Solid", "Reachy", "Committing"
        ]
        
        nouns = [
            "Arete", "Crack", "Edge", "Dihedral", "Problem", "Project", "Line",
            "Sequence", "Boulder", "Crux", "Traverse", "Dyno", "Gaston", "Mantle",
            "Pinch", "Sloper", "Crimp", "Jug", "Pocket", "Hold", "Flag", "Move"
        ]
        
        themes = [
            "Dream", "Quest", "Journey", "Challenge", "Mystery", "Adventure",
            "Riddle", "Puzzle", "Dance", "Symphony", "Path", "Ascent", "Horizon"
        ]
        
        name_types = [
            f"{random.choice(adjectives)} {random.choice(nouns)}",
            f"The {random.choice(adjectives)} {random.choice(nouns)}",
            f"{random.choice(themes)} of the {random.choice(nouns)}",
            f"{random.choice(adjectives)} {random.choice(themes)}"
        ]
        
        return random.choice(name_types)
    
    def compute_sequence(self, climb: Dict) -> Dict:
        """Compute a sequence for a climb using the sequence generator"""
        try:
            holds = climb.get('holds', [])
            if not holds:
                return {
                    'id': climb.get('id'),
                    'name': climb.get('name'),
                    'error': 'No holds data'
                }
            
            # Use sequence generator with more processing power for final sequence
            generator = ClimbSequenceGenerator(holds)
            result = generator.generate_sequences(beam_width=BEAM_WIDTH)
            
            if result['status'] != 'success':
                return {
                    'id': climb.get('id'),
                    'name': climb.get('name'),
                    'error': result.get('message', 'No valid sequences found')
                }
            
            return {
                'id': climb.get('id'),
                'name': climb.get('name'),
                'holds': holds,
                'best_sequence': result['best_sequence'],
                'all_sequences': result.get('all_sequences', [])[:3],
                'stats': result.get('stats', {}),
                'generated': climb.get('generated', True)
            }
            
        except Exception as e:
            return {
                'id': climb.get('id'),
                'name': climb.get('name'),
                'error': str(e)
            }
    
    def visualize_climb(self, climb: Dict, show_sequence: bool = True) -> None:
        """Visualize a climb and optionally its sequence"""
        holds = climb.get('holds', [])
        if not holds:
            print("No holds data to visualize")
            return
        
        fig, ax = plt.subplots(figsize=(10, 16))
        
        # Extract hold coordinates and roles
        x_coords = [h['x'] for h in holds]
        y_coords = [h['y'] for h in holds]
        roles = [h.get('role_id', 0) for h in holds]
        
        # Define colors for different roles
        role_colors = {
            12: 'green',    # Start holds
            13: 'blue',     # Intermediate holds
            14: 'red',      # Finish holds
            15: 'orange',   # Foot holds
            0: 'gray'       # Unknown role
        }
        
        colors = [role_colors.get(r, 'gray') for r in roles]
        
        # Plot holds
        ax.scatter(x_coords, y_coords, c=colors, s=100, alpha=0.7)
        
        # Add hold labels
        for i, (x, y, role) in enumerate(zip(x_coords, y_coords, roles)):
            role_label = {
                12: 'S',   # Start
                13: '',    # Intermediate (no label)
                14: 'F',   # Finish
                15: 'f'    # Foot
            }.get(role, '')
            
            ax.text(x, y, role_label, fontsize=12, ha='center', va='center')
        
        # Plot sequence if available
        if show_sequence and 'best_sequence' in climb and 'sequence' in climb['best_sequence']:
            sequence = climb['best_sequence']['sequence']
            
            # Create mapping from hole_id to index in holds list
            hold_indices = {h['hole_id']: i for i, h in enumerate(holds)}
            
            # Track positions for each limb
            limb_positions = {}
            
            for i, move in enumerate(sequence):
                limb = move.get('limb')
                hold_id = move.get('hold')
                
                if hold_id not in hold_indices:
                    continue
                
                hold_idx = hold_indices[hold_id]
                x, y = x_coords[hold_idx], y_coords[hold_idx]
                
                # Store current position
                limb_positions[limb] = (x, y)
                
                # Draw arms and legs to show connections
                if len(limb_positions) > 1:
                    # Calculate center of body
                    positions = list(limb_positions.values())
                    center_x = sum(p[0] for p in positions) / len(positions)
                    center_y = sum(p[1] for p in positions) / len(positions)
                    
                    # Draw lines from center to each limb
                    for l, (lx, ly) in limb_positions.items():
                        if l.endswith('H'):  # Hand
                            ax.plot([center_x, lx], [center_y, ly], 'k-', alpha=0.3, linewidth=1)
                        else:  # Foot
                            ax.plot([center_x, lx], [center_y, ly], 'k--', alpha=0.3, linewidth=1)
                
                # Label move number
                ax.text(x+10, y, str(i+1), fontsize=10, ha='left', va='center')
        
        # Set plot properties
        ax.set_xlim(min(x_coords) - 50, max(x_coords) + 50)
        ax.set_ylim(min(y_coords) - 50, max(y_coords) + 50)
        ax.set_title(f"Climb: {climb.get('name', 'Unnamed')}")
        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        
        # Create a legend
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='green', markersize=10, label='Start'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Intermediate'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Finish'),
            plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='Foot')
        ]
        ax.legend(handles=legend_elements, loc='upper right')
        
        plt.tight_layout()
        plt.show()
    
    def save_generated_climb(self, climb: Dict, output_dir: str = '.') -> str:
        """Save a generated climb to a JSON file"""
        os.makedirs(output_dir, exist_ok=True)
        filename = f"{output_dir}/generated_climb_{climb.get('id')}.json"
        
        with open(filename, 'w') as f:
            json.dump(climb, f, indent=2)
        
        print(f"Saved climb to {filename}")
        return filename
    
    def generate_batch(self, 
                num_climbs: int = 5,
                min_holds: int = 10,
                max_holds: int = 20,
                min_height: float = None,
                max_height: float = None,
                compute_sequences: bool = True,
                output_dir: str = None) -> List[Dict]:

        print(f"Generating batch of {num_climbs} climbs...")
        generated_climbs = []
        
        for i in tqdm(range(num_climbs), desc="Generating climbs"):
            # Randomly determine number of holds for this climb
            num_holds = random.randint(min_holds, max_holds)
            
            # Generate the climb
            try:
                climb = self.generate_climb(
                    num_holds=num_holds,
                    target_difficulty=None,  # Random difficulty
                    min_height=min_height,
                    max_height=max_height
                )
                
                # Compute sequence if requested
                if compute_sequences:
                    climb = self.compute_sequence(climb)
                
                # Save to file if output directory provided
                if output_dir:
                    self.save_generated_climb(climb, output_dir)
                
                # Add to results list
                generated_climbs.append(climb)
                
            except Exception as e:
                print(f"Error generating climb {i+1}: {e}")
                continue
        
        print(f"Successfully generated {len(generated_climbs)} climbs")
        return generated_climbs

    def export_climbs_to_database(self, climbs: List[Dict], db_path: str = None) -> None:
        """
        Export generated climbs to a SQLite database.
        
        Args:
            climbs: List of climb dictionaries to export
            db_path: Path to the SQLite database (if None, uses the instance's db_path)
        """
        if db_path is None:
            db_path = self.db_path
        
        print(f"Exporting {len(climbs)} climbs to database: {db_path}")
        
        # Connect to database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        try:
            # Create tables if they don't exist
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_climbs (
                id TEXT PRIMARY KEY,
                name TEXT,
                parameters TEXT,
                generated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_holds (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                climb_id TEXT,
                hole_id INTEGER,
                role_id INTEGER,
                FOREIGN KEY (climb_id) REFERENCES generated_climbs(id)
            )
            """)
            
            cursor.execute("""
            CREATE TABLE IF NOT EXISTS generated_sequences (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                climb_id TEXT,
                sequence_data TEXT,
                score REAL,
                FOREIGN KEY (climb_id) REFERENCES generated_climbs(id)
            )
            """)
            
            # Insert data for each climb
            for climb in climbs:
                # Skip climbs with errors
                if 'error' in climb:
                    continue
                    
                # Insert climb data
                cursor.execute(
                    "INSERT OR REPLACE INTO generated_climbs (id, name, parameters) VALUES (?, ?, ?)",
                    (
                        climb.get('id'),
                        climb.get('name'),
                        json.dumps(climb.get('parameters', {}))
                    )
                )
                
                # Insert hold data
                for hold in climb.get('holds', []):
                    cursor.execute(
                        "INSERT INTO generated_holds (climb_id, hole_id, role_id) VALUES (?, ?, ?)",
                        (
                            climb.get('id'),
                            hold.get('hole_id'),
                            hold.get('role_id')
                        )
                    )
                
                # Insert sequence data if available
                if 'best_sequence' in climb:
                    cursor.execute(
                        "INSERT INTO generated_sequences (climb_id, sequence_data, score) VALUES (?, ?, ?)",
                        (
                            climb.get('id'),
                            json.dumps(climb.get('best_sequence')),
                            climb.get('best_sequence', {}).get('evaluation', {}).get('score', 0)
                        )
                    )
            
            # Commit changes
            conn.commit()
            print(f"Successfully exported climbs to database")
            
        except Exception as e:
            conn.rollback()
            print(f"Error exporting to database: {e}")
        
        finally:
            conn.close()

    def run_generation_pipeline(self, 
                            num_climbs: int = 10,
                            min_holds: int = 12,
                            max_holds: int = 25,
                            output_dir: str = 'generated_climbs',
                            export_to_db: bool = True,
                            visualize_sample: int = 3) -> None:
        """
        Run a complete generation pipeline: analyze, generate, compute, save, and visualize.
        
        Args:
            num_climbs: Number of climbs to generate
            min_holds: Minimum number of holds per climb
            max_holds: Maximum number of holds per climb
            output_dir: Directory to save generated climbs
            export_to_db: Whether to export climbs to database
            visualize_sample: Number of sample climbs to visualize (0 to disable)
        """
        # Make sure we've analyzed the data
        if not self.holds_distribution:
            print("Analyzing existing climbs first...")
            self.analyze_climbs()
        
        # Generate batch of climbs
        generated_climbs = self.generate_batch(
            num_climbs=num_climbs,
            min_holds=min_holds,
            max_holds=max_holds,
            compute_sequences=True,
            output_dir=output_dir
        )
        
        # Export to database if requested
        if export_to_db:
            self.export_climbs_to_database(generated_climbs)
        
        # Visualize sample climbs if requested
        if visualize_sample > 0:
            sample_size = min(visualize_sample, len(generated_climbs))
            print(f"Visualizing {sample_size} sample climbs...")
            
            # Pick successful climbs (without errors)
            successful_climbs = [c for c in generated_climbs if 'error' not in c]
            
            if not successful_climbs:
                print("No successful climbs to visualize")
                return
                
            for climb in random.sample(successful_climbs, sample_size):
                self.visualize_climb(climb)
        
        print(f"Generation pipeline complete!")
        print(f"Generated {len(generated_climbs)} climbs")
        print(f"Saved to: {output_dir}")
        
        # Summarize generation results
        successful = sum(1 for c in generated_climbs if 'error' not in c)
        print(f"Success rate: {successful}/{len(generated_climbs)} ({successful/len(generated_climbs)*100:.1f}%)")


    if __name__ == "__main__":
        # Example usage
        DB_PATH = "climbing_db.sqlite"
        RESULTS_FILE = "climb_results.json"
        
        # Initialize and run the model
        model = ClimbGenerativeModel(DB_PATH, RESULTS_FILE)
        
        # Full pipeline
        model.run_generation_pipeline(
            num_climbs=20,
            min_holds=12,
            max_holds=20,
            output_dir="generated_climbs",
            export_to_db=True,
            visualize_sample=3
        )