import os
import sys
import argparse
import json
import torch
import random
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from collections import defaultdict
from lstm import ClimbGenerator, ClimbLSTM

class ClimbCreator:
    def __init__(self, model_path, data_file):
        self.data_file = data_file
        
        # Load data for hold layouts
        print(f"Loading data from {data_file}")
        self.wall_layouts = self._load_wall_layouts(data_file)
        print(f"Loaded {len(self.wall_layouts)} wall layouts")
        
        # Load generator to utilize its dataset
        self.generator = ClimbGenerator(data_file)
        if not self.generator.load_model(model_path):
            print("Warning: Failed to load model. Some functionality may be limited.")
            
        # Role mappings
        self.role_mapping = {
            12: 'Start',
            13: 'Hand',
            14: 'Finish',
            15: 'Foot'
        }
        self.reverse_role_mapping = {v: k for k, v in self.role_mapping.items()}
        
    def _load_wall_layouts(self, data_file):
        with open(data_file) as f:
            data = json.load(f)
            
        wall_layouts = {}
        
        for climb in data.get('results', []):
            if 'best_sequence' not in climb:
                continue
                
            # Get wall bounds
            holds = climb['best_sequence'].get('holds', [])
            if not holds:
                continue
                
            # Create a hash key based on the hold positions
            hold_positions = []
            for hold in holds:
                if 'hole_id' in hold and 'x' in hold and 'y' in hold:
                    hold_positions.append((hold['hole_id'], hold['x'], hold['y']))
            
            if hold_positions:
                # Sort to ensure same wall pattern with different hold IDs matches
                sorted_positions = sorted(hold_positions, key=lambda h: (h[1], h[2]))
                wall_key = '_'.join([f"{x}_{y}" for _, x, y in sorted_positions])
                
                if wall_key not in wall_layouts:
                    # Create wall layout entry
                    wall_layouts[wall_key] = {
                        'id': climb.get('id', f"wall_{len(wall_layouts)}"),
                        'name': climb.get('name', f"Wall Layout {len(wall_layouts) + 1}"),
                        'holds': holds,
                        'bounds': self._get_wall_bounds(holds)
                    }
        
        return wall_layouts
    
    def _get_wall_bounds(self, holds):
        if not holds:
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
            
        x_values = [hold['x'] for hold in holds if 'x' in hold]
        y_values = [hold['y'] for hold in holds if 'y' in hold]
        
        if not x_values or not y_values:
            return {'min_x': 0, 'max_x': 100, 'min_y': 0, 'max_y': 100}
            
        return {
            'min_x': min(x_values),
            'max_x': max(x_values),
            'min_y': min(y_values),
            'max_y': max(y_values)
        }
        
    def list_wall_layouts(self):
        if not self.wall_layouts:
            print("No wall layouts found in the data file")
            return []
            
        layouts = []
        for i, (key, layout) in enumerate(self.wall_layouts.items()):
            layouts.append({
                'index': i+1,
                'id': layout['id'],
                'name': layout['name'],
                'hold_count': len(layout['holds']),
                'bounds': layout['bounds']
            })
            
        return layouts
    
    def create_climb(self, wall_index=None, difficulty='moderate', start_holds=None, finish_holds=None, 
                     min_moves=6, max_moves=15):
        # Select a wall layout
        if wall_index is None:
            # Pick a random layout
            layout_key = random.choice(list(self.wall_layouts.keys()))
            layout = self.wall_layouts[layout_key]
        else:
            # Convert index to 0-based
            adj_index = max(0, min(wall_index - 1, len(self.wall_layouts) - 1))
            layout_key = list(self.wall_layouts.keys())[adj_index]
            layout = self.wall_layouts[layout_key]
        
        print(f"Creating new climb on wall layout: {layout['name']}")
        
        # Select appropriate holds for start and finish
        all_holds = layout['holds']
        wall_bounds = layout['bounds']
        
        # Process holds by role
        holds_by_role = defaultdict(list)
        for hold in all_holds:
            role_id = hold.get('role_id', 13)  # Default to hand
            holds_by_role[role_id].append(hold)
            
        # Choose start holds - prefer lower on the wall
        if not start_holds:
            # Sort holds by height (y-value)
            sorted_holds = sorted(all_holds, key=lambda h: h.get('y', 0))
            
            # Take 2 holds from the lower third of the wall
            lower_third = sorted_holds[:max(2, len(sorted_holds) // 3)]
            
            # If there are designated start holds, prefer those
            designated_starts = holds_by_role.get(12, [])
            if designated_starts:
                start_candidates = designated_starts
            else:
                start_candidates = lower_third
                
            # Select 2 start holds
            if len(start_candidates) >= 2:
                start_holds = random.sample(start_candidates, 2)
            else:
                # Use all available and possibly duplicate
                start_holds = start_candidates * 2 if start_candidates else sorted_holds[:2]
                
            # Mark them as start holds
            for hold in start_holds:
                hold['role_id'] = 12
        
        # Choose finish holds - prefer higher on the wall
        if not finish_holds:
            # Sort holds by height (y-value), descending
            sorted_holds = sorted(all_holds, key=lambda h: h.get('y', 0), reverse=True)
            
            # Take holds from the upper third of the wall
            upper_third = sorted_holds[:max(1, len(sorted_holds) // 3)]
            
            # If there are designated finish holds, prefer those
            designated_finishes = holds_by_role.get(14, [])
            if designated_finishes:
                finish_candidates = designated_finishes
            else:
                finish_candidates = upper_third
                
            # Select 1 finish hold
            if finish_candidates:
                finish_holds = [random.choice(finish_candidates)]
            else:
                finish_holds = [sorted_holds[0]] if sorted_holds else []
                
            # Mark it as a finish hold
            for hold in finish_holds:
                hold['role_id'] = 14
        
        # Select intermediate holds based on difficulty
        difficulty_map = {
            'easy': {'hand_ratio': 0.2, 'foot_ratio': 0.3, 'max_distance': 30},
            'moderate': {'hand_ratio': 0.3, 'foot_ratio': 0.2, 'max_distance': 40},
            'hard': {'hand_ratio': 0.4, 'foot_ratio': 0.1, 'max_distance': 50}
        }
        
        params = difficulty_map.get(difficulty, difficulty_map['moderate'])
        
        # Select additional hand holds
        available_holds = [h for h in all_holds 
                          if h not in start_holds and h not in finish_holds]
        
        # Determine how many hand and foot holds to use
        total_additional = min(len(available_holds), random.randint(min_moves, max_moves) - len(start_holds) - len(finish_holds))
        hand_count = max(2, int(total_additional * params['hand_ratio']))
        foot_count = max(1, int(total_additional * params['foot_ratio']))
        
        # Spread holds throughout the wall
        if available_holds:
            # Sort by y-coordinate
            sorted_by_y = sorted(available_holds, key=lambda h: h.get('y', 0))
            step_y = max(1, len(sorted_by_y) // (hand_count + 1))
            
            hand_holds = []
            # Select hand holds with good spacing
            for i in range(hand_count):
                index = min((i + 1) * step_y, len(sorted_by_y) - 1)
                hold = sorted_by_y[index]
                hold['role_id'] = 13  # Mark as hand hold
                hand_holds.append(hold)
            
            # Select foot holds
            remaining = [h for h in available_holds if h not in hand_holds]
            foot_holds = random.sample(remaining, min(foot_count, len(remaining)))
            for hold in foot_holds:
                hold['role_id'] = 15  # Mark as foot hold
        else:
            hand_holds = []
            foot_holds = []
        
        # Combine all selected holds
        selected_holds = start_holds + hand_holds + finish_holds + foot_holds
        
        # Create a sequence from the holds
        sequence = self._create_sequence(selected_holds)
        
        # Create the climb data
        new_climb = {
            "id": f"generated_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "name": f"Generated {difficulty.capitalize()} Climb",
            "best_sequence": {
                "holds": selected_holds,
                "sequence": sequence
            },
            "difficulty": difficulty,
            "wall_name": layout['name'],
            "creation_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return new_climb
    
    def _create_sequence(self, holds):
        # Sort holds by role and height
        sorted_holds = []
        
        # First add start holds
        start_holds = [h for h in holds if h.get('role_id') == 12]
        sorted_holds.extend(start_holds)
        
        # Add hand holds, sorted by height
        hand_holds = sorted([h for h in holds if h.get('role_id') == 13], 
                          key=lambda h: h.get('y', 0))
        sorted_holds.extend(hand_holds)
        
        # Add finish holds
        finish_holds = [h for h in holds if h.get('role_id') == 14]
        sorted_holds.extend(finish_holds)
        
        # Create sequence with alternating hands
        sequence = []
        current_hand = 'RH'
        
        for i, hold in enumerate(sorted_holds):
            if hold.get('role_id') in [12, 13, 14]:  # Start, Hand, or Finish hold
                sequence.append({
                    "hold": hold.get('hole_id'),
                    "limb": current_hand
                })
                # Switch hands
                current_hand = 'LH' if current_hand == 'RH' else 'RH'
        
        # Add foot holds at appropriate points if we have the generator model
        if hasattr(self, 'generator') and self.generator.model:
            # Use the model to suggest foot placements
            pass
            
        return sequence
    
    def visualize_climb(self, climb, save_path=None):
        if not climb or 'best_sequence' not in climb:
            print("Invalid climb data")
            return
            
        holds = climb['best_sequence'].get('holds', [])
        sequence = climb['best_sequence'].get('sequence', [])
        
        if not holds or not sequence:
            print("Climb has no holds or sequence")
            return
        
        plt.figure(figsize=(10, 12))
        
        # Draw grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Draw all holds
        for hold in holds:
            x, y = hold.get('x', 0), hold.get('y', 0)
            role_id = hold.get('role_id', 13)
            name = hold.get('name', '')
            
            color = 'gray'
            if role_id == 12:  # Start
                color = 'blue'
            elif role_id == 14:  # Finish
                color = 'red'
            elif role_id == 15:  # Foot
                color = 'green'
                
            plt.scatter(x, y, color=color, alpha=0.5, s=80 if role_id != 15 else 50)
            
            # Add hold name or ID
            label = name if name else f"{hold.get('hole_id', '')}"
            plt.text(x, y - 2, label, fontsize=6, ha='center', alpha=0.7)
        
        # Draw sequence
        colors = {'RH': 'red', 'LH': 'blue', 'RF': 'green', 'LF': 'purple'}
        
        # Create hold ID to hold mapping
        hold_map = {hold.get('hole_id'): hold for hold in holds if 'hole_id' in hold}
        
        # Draw the sequence path
        sequence_holds = []
        for i, move in enumerate(sequence):
            hold_id = move.get('hold')
            limb = move.get('limb', 'RH')
            
            if hold_id in hold_map:
                hold = hold_map[hold_id]
                x, y = hold.get('x', 0), hold.get('y', 0)
                
                plt.scatter(x, y, color=colors.get(limb, 'black'), s=100, zorder=10)
                plt.text(x, y, f"{i+1}", fontsize=10, ha='center', va='center', 
                       bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
                
                sequence_holds.append((x, y, limb))
        
        # Connect holds with arrows
        for i in range(1, len(sequence_holds)):
            prev_x, prev_y, _ = sequence_holds[i-1]
            curr_x, curr_y, curr_limb = sequence_holds[i]
            
            plt.arrow(prev_x, prev_y, curr_x-prev_x, curr_y-prev_y, 
                    head_width=0.5, head_length=0.7, fc=colors.get(curr_limb, 'gray'), 
                    ec=colors.get(curr_limb, 'gray'), alpha=0.6)
        
        # Add legend
        limb_labels = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=limb)
                     for limb, color in colors.items()]
        
        plt.legend(handles=limb_labels, loc='upper right')
        
        # Set labels and title
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title(f"{climb.get('name', 'New Climb')} - {climb.get('difficulty', 'moderate')}")
        
        # Add info
        info_text = (
            f"Wall: {climb.get('wall_name', 'Unknown')}\n"
            f"Difficulty: {climb.get('difficulty', 'moderate')}\n"
            f"Moves: {len(sequence)}\n"
            f"Start Holds: {len([h for h in holds if h.get('role_id') == 12])}\n"
            f"Hand Holds: {len([h for h in holds if h.get('role_id') == 13])}\n"
            f"Finish Holds: {len([h for h in holds if h.get('role_id') == 14])}"
        )
        plt.figtext(0.02, 0.02, info_text, fontsize=10, 
                  bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
    
    def export_climbs(self, climbs, output_file="generated_climbs.json"):
        output = {"results": climbs}
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Exported {len(climbs)} climbs to {output_file}")
        return output_file
    
    def batch_generate(self, count=5, wall_indices=None, difficulties=None, output_file=None):
        if not difficulties:
            difficulties = ['easy', 'moderate', 'hard']
            
        if not wall_indices:
            # Use all available walls
            wall_indices = list(range(1, len(self.wall_layouts) + 1))
            
        generated_climbs = []
        
        for i in range(count):
            # Cycle through wall layouts and difficulties
            wall_index = wall_indices[i % len(wall_indices)]
            difficulty = difficulties[i % len(difficulties)]
            
            try:
                new_climb = self.create_climb(
                    wall_index=wall_index,
                    difficulty=difficulty,
                    min_moves=5 + (2 if difficulty == 'easy' else 4 if difficulty == 'moderate' else 6),
                    max_moves=10 + (2 if difficulty == 'easy' else 5 if difficulty == 'moderate' else 8)
                )
                generated_climbs.append(new_climb)
                
                print(f"Generated climb {i+1}/{count}: {new_climb['name']} ({difficulty})")
            except Exception as e:
                print(f"Error generating climb {i+1}: {e}")
        
        # Export if requested
        if output_file and generated_climbs:
            self.export_climbs(generated_climbs, output_file)
            
        return generated_climbs

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create new climbing routes")
    parser.add_argument("--model", required=True, help="Path to the trained model")
    parser.add_argument("--data", required=True, help="Path to the climbing data JSON")
    parser.add_argument("--count", type=int, default=5, help="Number of routes to generate")
    parser.add_argument("--difficulty", choices=['easy', 'moderate', 'hard', 'mixed'], default='mixed',
                       help="Difficulty level of the routes")
    parser.add_argument("--wall", type=int, help="Specific wall layout index to use")
    parser.add_argument("--output", default="generated_climbs.json", help="Output file path")
    parser.add_argument("--visualize", action="store_true", help="Visualize the generated routes")
    parser.add_argument("--output-dir", default="visualizations", help="Directory for visualizations")
    
    args = parser.parse_args()
    
    # Create output directory if needed
    if args.visualize:
        os.makedirs(args.output_dir, exist_ok=True)
    
    # Initialize the creator
    creator = ClimbCreator(args.model, args.data)
    
    # Show available wall layouts
    layouts = creator.list_wall_layouts()
    if not layouts:
        print("No wall layouts found in the data file. Exiting.")
        sys.exit(1)
        
    print("\nAvailable wall layouts:")
    for layout in layouts:
        print(f"{layout['index']}. {layout['name']} - {layout['hold_count']} holds")
    
    # Set up difficulty options
    difficulties = ['easy', 'moderate', 'hard'] if args.difficulty == 'mixed' else [args.difficulty]
    
    # Generate climbs
    wall_indices = [args.wall] if args.wall else None
    climbs = creator.batch_generate(args.count, wall_indices, difficulties, args.output)
    
    # Visualize if requested
    if args.visualize and climbs:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        for i, climb in enumerate(climbs):
            save_path = os.path.join(args.output_dir, f"climb_{timestamp}_{i+1}.png")
            creator.visualize_climb(climb, save_path)