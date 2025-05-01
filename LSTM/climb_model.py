import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict

class ClimbDataset(Dataset):
    def __init__(self, json_file, max_sequence_length=50):
        with open(json_file) as f:
            data = json.load(f)
        
        self.sequences = []
        self.hold_mapping = {'PAD': 0}
        self.reverse_hold_mapping = {0: 'PAD'}
        self.limb_mapping = {'RH': 0, 'LH': 1, 'RF': 2, 'LF': 3}
        self.max_sequence_length = max_sequence_length
        self.hold_info = {}

        # First pass to build vocabulary
        hold_counter = 1
        for result in data['results']:
            if 'best_sequence' in result:
                sequence = result['best_sequence']['sequence']
                for move in sequence:
                    hold_id = move['hold']
                    if hold_id not in self.hold_mapping:
                        self.hold_mapping[hold_id] = hold_counter
                        self.reverse_hold_mapping[hold_counter] = hold_id
                        hold_counter += 1

        # Second pass to create sequences and collect hold info
        for result in data['results']:
            if 'best_sequence' in result:
                sequence = result['best_sequence']['sequence']
                holds = result['best_sequence'].get('holds', [])
                
                # Store hold information
                for h in holds:
                    self.hold_info[h['hole_id']] = {
                        'x': h['x'],
                        'y': h['y'],
                        'role_id': h['role_id']
                    }
                
                encoded_sequence = []
                for move in sequence:
                    if move['hold'] not in self.hold_mapping:
                        continue
                        
                    hold_token = self.hold_mapping[move['hold']]
                    limb_token = self.limb_mapping[move['limb']]
                    combined_token = hold_token * len(self.limb_mapping) + limb_token + 1
                    encoded_sequence.append(combined_token)
                
                if not encoded_sequence:
                    continue
                    
                if len(encoded_sequence) < self.max_sequence_length:
                    encoded_sequence = encoded_sequence + [0] * (self.max_sequence_length - len(encoded_sequence))
                else:
                    encoded_sequence = encoded_sequence[:self.max_sequence_length]
                
                self.sequences.append(encoded_sequence)
        
        self.vocab_size = (len(self.hold_mapping) * len(self.limb_mapping)) + 1

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return torch.tensor(sequence[:-1], dtype=torch.long), torch.tensor(sequence[1:], dtype=torch.long)
    
    def decode_sequence(self, encoded_sequence):
        decoded = []
        for token in encoded_sequence:
            if token == 0:
                continue
            token -= 1
            hold_token = token // len(self.limb_mapping)
            limb_token = token % len(self.limb_mapping)
            
            if hold_token in self.reverse_hold_mapping and limb_token in self.limb_mapping.values():
                hold_id = self.reverse_hold_mapping[hold_token]
                limb = list(self.limb_mapping.keys())[list(self.limb_mapping.values()).index(limb_token)]
                role_id = self.hold_info.get(hold_id, {}).get('role_id', -1)
                
                decoded.append({
                    'hold': hold_id,
                    'limb': limb,
                    'role_id': role_id,
                    'x': self.hold_info.get(hold_id, {}).get('x', 0),
                    'y': self.hold_info.get(hold_id, {}).get('y', 0)
                })
        return decoded

class ClimbLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2):
        super(ClimbLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=num_layers, 
                           batch_first=True, dropout=0.2 if num_layers > 1 else 0)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        
    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        output = self.fc(lstm_out)
        return output, hidden

class ClimbGenerator:
    def __init__(self, json_file):
        self.json_file = json_file
        self.dataset = ClimbDataset(json_file)
        self.model = None
        self.climb_db = self._build_climb_database()
        self.role_mapping = {
            12: 'Start',
            13: 'Hand',
            14: 'Finish',
            15: 'Foot'
        }

    def train(self, num_epochs=20, batch_size=32, learning_rate=0.001):
        train_data, val_data = train_test_split(self.dataset, test_size=0.2, random_state=42)
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ClimbLSTM(self.dataset.vocab_size).to(device)
        criterion = nn.CrossEntropyLoss(ignore_index=0)
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        best_val_loss = float('inf')
        for epoch in range(num_epochs):
            self.model.train()
            train_loss = 0
            for inputs, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                inputs, targets = inputs.to(device), targets.to(device)
                
                if (inputs >= self.dataset.vocab_size).any() or (targets >= self.dataset.vocab_size).any():
                    continue
                
                optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = criterion(outputs.reshape(-1, self.dataset.vocab_size), 
                               targets.reshape(-1))
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
            
            self.model.eval()
            val_loss = 0
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs, targets = inputs.to(device), targets.to(device)
                    if (inputs >= self.dataset.vocab_size).any() or (targets >= self.dataset.vocab_size).any():
                        continue
                    outputs, _ = self.model(inputs)
                    loss = criterion(outputs.reshape(-1, self.dataset.vocab_size), 
                                   targets.reshape(-1))
                    val_loss += loss.item()
            
            train_loss /= len(train_loader)
            val_loss /= len(val_loader)
            print(f"Epoch {epoch+1}: Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(self.model.state_dict(), 'lstm.pth')

    def load_model(self, model_path):
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = ClimbLSTM(self.dataset.vocab_size).to(device)
        self.model.load_state_dict(torch.load(model_path))
        self.model.eval()

    def generate(self, climb_ids, num_sequences=1, temperature=0.7):
        if not self.model:
            raise ValueError("Model not loaded or trained")
            
        start_holds, common_holds = self._analyze_examples(climb_ids)
        device = next(self.model.parameters()).device
        
        generated_climbs = []
        for i in range(num_sequences):
            try:
                # Generate sequence
                sequence = []
                hidden = None
                current_tokens = []
                
                # Convert start holds to tokens
                for hold_id, limb in start_holds:
                    if hold_id not in self.dataset.hold_mapping:
                        continue
                    hold_token = self.dataset.hold_mapping[hold_id]
                    limb_token = self.dataset.limb_mapping[limb]
                    combined_token = (hold_token * len(self.dataset.limb_mapping)) + limb_token + 1
                    current_tokens.append(combined_token)
                
                if not current_tokens:
                    raise ValueError("No valid start holds")
                
                sequence = current_tokens.copy()
                
                self.model.eval()
                with torch.no_grad():
                    for _ in range(self.dataset.max_sequence_length - len(start_holds)):
                        input_seq = torch.tensor([sequence[-self.dataset.max_sequence_length:]], 
                                                dtype=torch.long).to(device)
                        
                        if (input_seq >= self.dataset.vocab_size).any():
                            break
                            
                        output, hidden = self.model(input_seq, hidden)
                        logits = output[0, -1, :] / temperature
                        probs = torch.softmax(logits, dim=-1)
                        next_token = torch.multinomial(probs, num_samples=1).item()
                        
                        if next_token == 0:
                            break
                            
                        sequence.append(next_token)
                
                decoded = self.dataset.decode_sequence(sequence)
                
                # Create output
                generated_climb = {
                    'id': f"generated_{i+1}",
                    'name': f"Inspired by {', '.join([self.climb_db[cid]['name'] for cid in climb_ids if cid in self.climb_db])}",
                    'sequence': decoded,
                    'holds': [{
                        'hole_id': move['hold'],
                        'x': move['x'],
                        'y': move['y'],
                        'role_id': move['role_id'],
                        'role_name': self.role_mapping.get(move['role_id'], 'Unknown')
                    } for move in decoded]
                }
                
                generated_climbs.append(generated_climb)
                
                # Print results
                print(f"\nGenerated Climb {i+1}:")
                for move in decoded:
                    role_name = self.role_mapping.get(move['role_id'], 'Unknown')
                    print(f"{move['limb']} -> Hold {move['hold']} ({role_name}) at ({move['x']:.1f}, {move['y']:.1f})")
                
            except Exception as e:
                print(f"Error generating sequence: {str(e)}")
        
        return generated_climbs

    def _build_climb_database(self):
        climb_db = {}
        with open(self.json_file) as f:
            data = json.load(f)
            for result in data['results']:
                if 'best_sequence' in result:
                    climb_db[result['id']] = {
                        'name': result.get('name', 'Unnamed').strip(),
                        'sequence': result['best_sequence']['sequence'],
                        'holds': result['best_sequence'].get('holds', [])
                    }
        return climb_db

    def _analyze_examples(self, climb_ids):
        start_holds = []
        hold_counts = defaultdict(int)
        
        for climb_id in climb_ids:
            if climb_id not in self.climb_db:
                print(f"Warning: Climb ID {climb_id} not found")
                continue
                
            climb = self.climb_db[climb_id]
            sequence = climb['sequence']
            
            # Get starting holds (first 2 moves)
            if len(sequence) >= 2:
                start_holds.append((sequence[0]['hold'], sequence[0]['limb']))
                start_holds.append((sequence[1]['hold'], sequence[1]['limb']))
            
            # Count hold frequencies
            for move in sequence:
                hold_counts[move['hold']] += 1
        
        # Get most common holds
        common_holds = [hold for hold, count in sorted(hold_counts.items(), 
                                                     key=lambda x: x[1], 
                                                     reverse=True)[:10]]
        
        # Default start holds if none found
        if not start_holds:
            start_holds = [
                (list(self.dataset.hold_mapping.keys())[1], 'RH'),
                (list(self.dataset.hold_mapping.keys())[2], 'LH')
            ]
        
        return start_holds[:4], common_holds