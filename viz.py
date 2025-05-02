import argparse
import json
import os
import matplotlib.pyplot as plt
from collections import defaultdict
from datetime import datetime

def visualize_and_save_climb(climb_data, climb_id, output_dir="climb_visualizations"):
    """Visualize a specific climb by ID and automatically save the image"""
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find the climb in the data
    climb = None
    for result in climb_data.get('results', []):
        if result.get('id') == climb_id:
            climb = result
            break
    
    if not climb:
        print(f"Climb with ID '{climb_id}' not found")
        return False
    
    climb_name = climb.get('name', 'Unnamed').replace(' ', '_')
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(output_dir, f"{climb_name}_{timestamp}.png")
    
    print(f"Visualizing climb: {climb.get('name', 'Unnamed')} (ID: {climb_id})")
    
    holds = climb.get('best_sequence', {}).get('holds', [])
    sequence = climb.get('best_sequence', {}).get('sequence', [])
    
    if not holds or not sequence:
        print("Climb has no holds or sequence")
        return False
    
    # Create visualization
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
    plt.title(f"{climb.get('name', 'Climb')} (ID: {climb_id})")
    
    # Add climb info
    start_count = len([h for h in holds if h.get('role_id') == 12])
    hand_count = len([h for h in holds if h.get('role_id') == 13])
    finish_count = len([h for h in holds if h.get('role_id') == 14])
    foot_count = len([h for h in holds if h.get('role_id') == 15])
    
    info_text = (
        f"Moves: {len(sequence)}\n"
        f"Start Holds: {start_count}\n"
        f"Hand Holds: {hand_count}\n"
        f"Finish Holds: {finish_count}\n"
        f"Foot Holds: {foot_count}"
    )
    plt.figtext(0.02, 0.02, info_text, fontsize=10, 
              bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
    
    plt.tight_layout()
    
    # Save the image
    plt.savefig(save_path)
    print(f"Visualization saved to {save_path}")
    plt.close()
    
    return True

def main():
    parser = argparse.ArgumentParser(description="Visualize a specific climb and save the image")
    parser.add_argument("--data", required=True, help="Path to the climbing data JSON file")
    parser.add_argument("--id", required=True, help="ID of the climb to visualize")
    parser.add_argument("--output-dir", default="climb_visualizations", 
                       help="Directory to save visualizations")
    
    args = parser.parse_args()
    
    # Load the data
    try:
        with open(args.data, 'r') as f:
            data = json.load(f)
    except Exception as e:
        print(f"Error loading data file: {e}")
        return
    
    # Visualize and save the climb
    visualize_and_save_climb(data, args.id, args.output_dir)

if __name__ == "__main__":
    main()