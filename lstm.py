import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F

class ClimbDataset(Dataset):
    def __init__(self, json_file, max_sequence_length=50):
        print(f"Loading dataset from {json_file}")
        with open(json_file) as f:
            data = json.load(f)
        
        self.sequences = []
        self.hold_mapping = {'PAD': 0}
        self.reverse_hold_mapping = {0: 'PAD'}
        self.limb_mapping = {'RH': 0, 'LH': 1, 'RF': 2, 'LF': 3}
        self.role_mapping = {12: 'Start', 13: 'Hand', 14: 'Finish', 15: 'Foot'}
        self.max_sequence_length = max_sequence_length
        self.hold_info = {}

        self._build_vocabulary(data)
        self._create_sequences(data)
        self.augment_sequences()
        
        self.vocab_size = (len(self.hold_mapping) * len(self.limb_mapping)) + 1
        print(f"Vocabulary size: {self.vocab_size}")

    def _build_vocabulary(self, data):
        hold_counter = 1
        
        #Process all sequences to build vocab and hold info
        for result in data['results']:
            if 'best_sequence' not in result:
                continue
                
            #Add sequence holds to vocabulary
            for move in result['best_sequence']['sequence']:
                hold_id = move['hold']
                if hold_id not in self.hold_mapping:
                    self.hold_mapping[hold_id] = hold_counter
                    self.reverse_hold_mapping[hold_counter] = hold_id
                    hold_counter += 1

            #Add hold information (coordinates, roles)
            for h in result['best_sequence'].get('holds', []):
                hold_id = h.get('hole_id')
                if hold_id:
                    self.hold_info[hold_id] = {
                        'x': h.get('x', 0),
                        'y': h.get('y', 0),
                        'role_id': h.get('role_id', 13),  #Default to hand hold
                        'name': h.get('name', ''),
                        'climb_id': result.get('id', '')
                    }
        
        print(f"Built vocabulary with {len(self.hold_mapping)} holds")
        print(f"Loaded information for {len(self.hold_info)} holds")

    def _create_sequences(self, data):
        total_sequences = valid_sequences = 0
        
        for result in data['results']:
            if 'best_sequence' not in result:
                continue
                
            total_sequences += 1
            sequence = result['best_sequence']['sequence']
            encoded_sequence = []
            
            for move in sequence:
                if move['hold'] not in self.hold_mapping:
                    continue
                    
                hold_token = self.hold_mapping[move['hold']]
                limb_token = self.limb_mapping.get(move['limb'], 0)  #Default to RH if limb not found
                encoded_sequence.append((hold_token * len(self.limb_mapping)) + limb_token + 1)
            
            if encoded_sequence:
                valid_sequences += 1
                #Pad or truncate sequence
                if len(encoded_sequence) < self.max_sequence_length:
                    encoded_sequence += [0] * (self.max_sequence_length - len(encoded_sequence))
                else:
                    encoded_sequence = encoded_sequence[:self.max_sequence_length]
                
                self.sequences.append(encoded_sequence)
        
        print(f"Processed {total_sequences} sequences, {valid_sequences} valid")

    def augment_sequences(self):
        original_count = len(self.sequences)
        new_sequences = []
        
        #mirroring technique
        for seq in self.sequences[:original_count]:
            mirrored = []
            for token in seq:
                if token == 0:  #Padding
                    mirrored.append(0)
                    continue
                    
                token -= 1
                hold_token = token // len(self.limb_mapping)
                limb_token = token % len(self.limb_mapping)
                
                #Swap left and right limbs
                if limb_token in [self.limb_mapping['LH'], self.limb_mapping['RH']]:
                    limb_token = 1 - limb_token  #Swap RH(0) and LH(1)
                elif limb_token in [self.limb_mapping['LF'], self.limb_mapping['RF']]:
                    limb_token = 5 - limb_token  #Swap LF(2) and RF(3)
                    
                new_token = (hold_token * len(self.limb_mapping)) + limb_token + 1
                mirrored.append(new_token)
            new_sequences.append(mirrored)
        
        #Subsequence creation
        min_subseq_length = 8
        for seq in self.sequences[:original_count]:
            actual_seq = [t for t in seq if t != 0]
            if len(actual_seq) >= min_subseq_length * 2:
                mid_point = len(actual_seq) // 2
                for half in [actual_seq[:mid_point], actual_seq[mid_point:]]:
                    if len(half) >= min_subseq_length:
                        new_sequences.append(half + [0] * (self.max_sequence_length - len(half)))
        
        #jitter
        jitter_prob = 0.1
        for seq in self.sequences[:original_count]:
            if seq.count(0) <= len(seq) * 0.5:  # Skip mostly padded sequences
                valid_tokens = list(set(t for t in seq if t != 0))
                if valid_tokens:
                    jittered = [t if random.random() > jitter_prob else random.choice(valid_tokens) 
                               for t in seq]
                    new_sequences.append(jittered)
        
        self.sequences.extend(new_sequences)
        print(f"Augmented dataset from {original_count} to {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        return (
            torch.tensor(sequence[:-1], dtype=torch.long),
            torch.tensor(sequence[1:], dtype=torch.long)
        )
    
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
                limb = next(k for k, v in self.limb_mapping.items() if v == limb_token)
                role_id = self.hold_info.get(hold_id, {}).get('role_id', -1)
                
                decoded.append({
                    'hold': hold_id,
                    'limb': limb,
                    'role_id': role_id,
                    'x': self.hold_info.get(hold_id, {}).get('x', 0),
                    'y': self.hold_info.get(hold_id, {}).get('y', 0),
                    'name': self.hold_info.get(hold_id, {}).get('name', '')
                })
        return decoded

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None, ignore_index=0):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.ignore_index = ignore_index
        
    def forward(self, input, target):
        ce_loss = F.cross_entropy(input, target, reduction='none', ignore_index=self.ignore_index)
        pt = torch.exp(-ce_loss)
        return ((1 - pt) ** self.gamma * ce_loss).mean()

class ClimbLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embedding_dim, hidden_dim, 
            num_layers=num_layers,
            batch_first=True, 
            dropout=dropout if num_layers > 1 else 0
        )
        self.bn = nn.BatchNorm1d(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)
        self._init_weights()
        
    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        
    def forward(self, x, hidden=None):
        batch_size, seq_len = x.size()
        embedded = self.embedding(x)
        lstm_out, hidden = self.lstm(embedded, hidden)
        
        # Apply batch norm
        reshaped = lstm_out.contiguous().view(-1, lstm_out.size(-1))
        normalized = self.bn(reshaped)
        reshaped = normalized.view(batch_size, seq_len, -1)
        
        return self.fc(reshaped), hidden

class EarlyStopping:
    def __init__(self, patience=7, min_delta=0.001, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.counter = 0
        self.best_loss = float('inf')
        self.best_weights = None
        self.stopped_epoch = 0
        
    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.stopped_epoch = self.counter
                return True
            return False
    
    def restore_weights(self, model):
        if self.restore_best_weights and self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            print(f"Restored model to best weights from epoch {self.stopped_epoch - self.patience}")

class ClimbGenerator:
    def __init__(self, json_file):
        print(f"Initializing ClimbGenerator")
        print(f"Data file: {json_file}")
        
        self.dataset = ClimbDataset(json_file)
        self.model = None
        self.climb_data = self._load_climb_data(json_file)
        self.role_mapping = {
            12: 'Start',
            13: 'Hand',
            14: 'Finish',
            15: 'Foot'
        }
        
    def _load_climb_data(self, json_file):
        with open(json_file) as f:
            data = json.load(f)
        
        climb_data = {}
        for result in data['results']:
            if 'id' in result and 'best_sequence' in result:
                climb_id = result['id']
                
                climb_data[climb_id] = {
                    'name': result.get('name', f"Climb {climb_id}"),
                    'holds': result['best_sequence'].get('holds', []),
                    'sequences': [result['best_sequence'].get('sequence', [])]
                }
        
        print(f"Loaded {len(climb_data)} climbs from {json_file}")
        return climb_data

    def load_model(self, model_path):
        try:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            
            self.model = ClimbLSTM(
                vocab_size=self.dataset.vocab_size,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                dropout=0.2
            ).to(device)
            
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train(self, num_epochs=30, batch_size=32, learning_rate=0.001, save_path='lstm_model.pth'):
        train_data, val_data = train_test_split(self.dataset, test_size=0.2, random_state=42, shuffle=True)
        
        #Set up data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Training on: {device}")
        
        #Initialize model
        self.model = ClimbLSTM(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2
        ).to(device)
        
        #Set up loss, optimizer and schedulers
        criterion = FocalLoss(gamma=2.0, ignore_index=0)
        optimizer = optim.AdamW(self.model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        #Learning rate warm-up and reduction
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=learning_rate,
            steps_per_epoch=len(train_loader),
            epochs=num_epochs
        )
        
        early_stopping = EarlyStopping(patience=7, restore_best_weights=True)
        
        #Training loop
        train_losses, val_losses = [], []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for inputs, targets in progress_bar:
                inputs, targets = inputs.to(device), targets.to(device)
                
                #Skip batch if it contains invalid tokens
                if (inputs >= self.dataset.vocab_size).any() or (targets >= self.dataset.vocab_size).any():
                    print("Skipping batch with invalid token IDs")
                    continue
                
                optimizer.zero_grad()
                outputs, _ = self.model(inputs)
                loss = criterion(outputs.reshape(-1, self.dataset.vocab_size), targets.reshape(-1))
                loss.backward()
                
                #Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()
                
                epoch_loss += loss.item()
                progress_bar.set_postfix({"loss": f"{loss.item():.4f}"})
            
            #validation
            val_loss = self._evaluate(val_loader, criterion, device)
            
            #Log metrics
            avg_train_loss = epoch_loss / len(train_loader)
            avg_val_loss = val_loss / len(val_loader)
            train_losses.append(avg_train_loss)
            val_losses.append(avg_val_loss)
            
            print(f"Epoch {epoch+1}: Train Loss={avg_train_loss:.4f}, Val Loss={avg_val_loss:.4f}")
            
            #Check for early stopping
            if early_stopping(avg_val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                early_stopping.restore_weights(self.model)
                break
            
            #Generate a sample sequence every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._generate_sample(device)
        
        #Ensure best weights are used
        if not early_stopping.stopped_epoch and early_stopping.best_weights is not None:
            self.model.load_state_dict(early_stopping.best_weights)
        
        #Save the model
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")
        
        #Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses

    def _evaluate(self, loader, criterion, device):
        """Evaluate the model on a data loader"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets in loader:
                inputs, targets = inputs.to(device), targets.to(device)
                
                #Skip batch if it contains invalid tokens
                if (inputs >= self.dataset.vocab_size).any() or (targets >= self.dataset.vocab_size).any():
                    continue
                
                outputs, _ = self.model(inputs)
                loss = criterion(outputs.reshape(-1, self.dataset.vocab_size), targets.reshape(-1))
                total_loss += loss.item()
        
        return total_loss

    def _generate_sample(self, device, temp=0.8):
        self.model.eval()
        
        with torch.no_grad():
            #Get a random climb ID
            if self.climb_data:
                climb_id = random.choice(list(self.climb_data.keys()))
                print(f"\nGenerating sample sequence for climb: {self.climb_data[climb_id]['name']}")
                
                #Get start holds from the climb
                start_holds = []
                for hold in self.climb_data[climb_id]['holds']:
                    hold_id = hold.get('hole_id')
                    if hold_id and hold.get('role_id') == 12:  # Start hold
                        start_holds.append(hold_id)
                
                #If no start holds found, get the lowest 2 holds
                if not start_holds:
                    sorted_holds = sorted(self.climb_data[climb_id]['holds'], 
                                          key=lambda h: h.get('y', 0))
                    start_holds = [h.get('hole_id') for h in sorted_holds[:2]]
                
                #Generate sequence using up to 2 start holds
                sequence = []
                for i, hold_id in enumerate(start_holds[:2]):
                    if hold_id in self.dataset.hold_mapping:
                        limb = 'RH' if i == 0 else 'LH'
                        hold_token = self.dataset.hold_mapping[hold_id]
                        limb_token = self.dataset.limb_mapping[limb]
                        sequence.append((hold_token * len(self.dataset.limb_mapping)) + limb_token + 1)
                
                #Generate the rest of the sequence
                if sequence:
                    input_seq = torch.tensor([sequence], dtype=torch.long).to(device)
                    output, hidden = self.model(input_seq)
                    
                    for _ in range(20):  #Generate up to 20 more moves
                        logits = output[0, -1, :] / temp
                        logits[0] = float('-inf')   ##Don't predict padding
                        
                        #Sample from the distribution
                        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
                        
                        if next_token == 0:
                            break
                            
                        sequence.append(next_token)
                        
                        input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)
                        output, hidden = self.model(input_seq, hidden)
                    
                    #Decode and print the sequence
                    decoded = self.dataset.decode_sequence(sequence)
                    print(f"Generated sequence with {len(decoded)} moves:")
                    for i, move in enumerate(decoded[:10]):  # Print first 10 moves
                        print(f"{i+1}. {move['limb']} on hold {move['hold']} ({move.get('name', '')})")
                    
                    if len(decoded) > 10:
                        print(f"... and {len(decoded) - 10} more moves")
                
            else:
                print("\nNo climb data available for sample generation")
        
        self.model.train()

    def _plot_training_history(self, train_losses, val_losses):
        plt.figure(figsize=(10, 6))
        plt.plot(train_losses, label='Training Loss')
        plt.plot(val_losses, label='Validation Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.grid(True)
        plt.savefig('training_history.png')
        plt.close()

    def generate(self, climb_id=None, num_sequences=3, temperature=0.8, max_length=30):
        if not self.model:
            print("Error: Model not loaded")
            return []
        
        device = next(self.model.parameters()).device
        sequences = []
        
        #Get climb IDs to generate 
        climb_ids = [climb_id] if climb_id else list(self.climb_data.keys())
        if not climb_ids:
            print("No climb data available")
            return []
        
        print(f"Generating sequences for {len(climb_ids)} climbs, temperature={temperature}")
        
        for i in range(num_sequences):
            #Pick a random climb if multiple are available
            current_id = climb_id if climb_id else random.choice(climb_ids)
            print(f"\nGenerating sequence {i+1} for climb: {self.climb_data[current_id]['name']}")
            
            #Get start holds
            start_holds = []
            for hold in self.climb_data[current_id]['holds']:
                hold_id = hold.get('hole_id')
                if hold_id and hold.get('role_id') == 12:  # Start hold
                    start_holds.append((hold_id, 'RH' if len(start_holds) == 0 else 'LH'))
            
            #If no start holds found, use any holds from vocabulary
            if not start_holds:
                #Try to get from the lowest holds
                if self.climb_data[current_id]['holds']:
                    sorted_holds = sorted(self.climb_data[current_id]['holds'], 
                                        key=lambda h: h.get('y', 0))
                    start_holds = [(h.get('hole_id'), 'RH' if i == 0 else 'LH') 
                                for i, h in enumerate(sorted_holds[:2])]
                
                #If still no valid holds, use any holds from vocabulary
                if not start_holds:
                    valid_holds = list(self.dataset.hold_mapping.keys())
                    if 'PAD' in valid_holds:
                        valid_holds.remove('PAD')
                    
                    if valid_holds:
                        random.shuffle(valid_holds)
                        start_holds = [(valid_holds[0], 'RH'), 
                                      (valid_holds[min(1, len(valid_holds)-1)], 'LH')]
            
            if not start_holds:
                print(f"No valid start holds found for climb {current_id}")
                continue
                
            #Generate sequence using start holds
            current_tokens = []
            for hold_id, limb in start_holds:
                if hold_id in self.dataset.hold_mapping:
                    hold_token = self.dataset.hold_mapping[hold_id]
                    limb_token = self.dataset.limb_mapping[limb]
                    current_tokens.append((hold_token * len(self.dataset.limb_mapping)) + limb_token + 1)
            
            if not current_tokens:
                print(f"Could not create start tokens for climb {current_id}")
                continue
                
            print(f"Using {len(current_tokens)} start tokens")
                
            #Generate the sequence
            sequence_tokens = current_tokens.copy()
            
            try:
                with torch.no_grad():
                    input_seq = torch.tensor([current_tokens], dtype=torch.long).to(device)
                    output, hidden = self.model(input_seq)
                    
                    for _ in range(max_length):
                        logits = output[0, -1, :] / temperature
                        logits[0] = float('-inf')
                        
                        #Top-p (nucleus) sampling
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        
                        sorted_indices_to_remove = cumulative_probs > 0.9
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        
                        indices_to_remove = sorted_indices[sorted_indices_to_remove]
                        logits[indices_to_remove] = float('-inf')
                        
                        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
                        
                        if next_token == 0:
                            break
                            
                        sequence_tokens.append(next_token)
                        
                        input_seq = torch.tensor([[next_token]], dtype=torch.long).to(device)
                        output, hidden = self.model(input_seq, hidden)
                
                #Decode
                decoded_sequence = self.dataset.decode_sequence(sequence_tokens)
                
                #Validate and clean
                valid_sequence = self._validate_sequence(decoded_sequence, current_id)
                
                sequences.append({
                    'climb_id': current_id,
                    'climb_name': self.climb_data[current_id]['name'],
                    'sequence': valid_sequence,
                    'is_valid': len(valid_sequence) >= 3  #min 3 moves
                })
                
                #sequence summary
                print(f"Generated sequence with {len(valid_sequence)} moves")
                valid_moves = 0
                for move in valid_sequence[:5]:
                    valid_moves += 1
                    print(f"{valid_moves}. {move['limb']} on hold {move['hold']} ({move.get('name', '')})")
                if len(valid_sequence) > 5:
                    print(f"... and {len(valid_sequence) - 5} more moves")
                    
            except Exception as e:
                print(f"Error generating sequence: {e}")
        
        valid_count = sum(1 for s in sequences if s['is_valid'])
        print(f"\nGenerated {len(sequences)} sequences, {valid_count} valid")
        
        return sequences

    def _validate_sequence(self, sequence, climb_id):
        if not sequence:
            return []
        
        #make sure all holds exist in the climb
        valid_hold_ids = set()
        start_holds = set()
        finish_holds = set()
        
        if climb_id in self.climb_data and self.climb_data[climb_id]['holds']:
            valid_hold_ids = {h.get('hole_id') for h in self.climb_data[climb_id]['holds']}
            start_holds = {h.get('hole_id') for h in self.climb_data[climb_id]['holds'] 
                          if h.get('role_id') == 12}
            finish_holds = {h.get('hole_id') for h in self.climb_data[climb_id]['holds'] 
                           if h.get('role_id') == 14}
        
        if valid_hold_ids:
            cleaned = [m for m in sequence if m['hold'] in valid_hold_ids]
        else:
            cleaned = sequence.copy()
        
        #make sure there is hand alternation
        last_hand = None
        alternated = []
        
        for move in cleaned:
            if move['limb'] in ['RH', 'LH']:
                if last_hand == move['limb']:
                    move['limb'] = 'LH' if move['limb'] == 'RH' else 'RH'
                last_hand = move['limb']
            alternated.append(move)
        
        #makes sure the sequence ends with a finish hold
        has_finish = any(m['hold'] in finish_holds for m in alternated[-2:]) if finish_holds else False
        
        #add finish if there is noen
        if not has_finish and finish_holds:
            last_hand = alternated[-1]['limb'] if alternated and alternated[-1]['limb'] in ['RH', 'LH'] else 'RH'
            next_hand = 'LH' if last_hand == 'RH' else 'RH'
            
            finish_hold = next(iter(finish_holds))
            alternated.append({
                'hold': finish_hold,
                'limb': next_hand,
                'role_id': 14,
                'x': self.dataset.hold_info.get(finish_hold, {}).get('x', 0),
                'y': self.dataset.hold_info.get(finish_hold, {}).get('y', 0),
                'name': self.dataset.hold_info.get(finish_hold, {}).get('name', '')
            })
        
        return alternated

    def visualize_sequence(self, sequence, climb_id=None, save_path=None):
        if not sequence:
            print("No sequence to visualize")
            return
        
        plt.figure(figsize=(10, 12))
        
        #Draw grid
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Get all holds for this climb
        all_holds = []
        if climb_id and climb_id in self.climb_data:
            all_holds = [(h.get('hole_id'), h.get('x', 0), h.get('y', 0), h.get('role_id', -1))
                         for h in self.climb_data[climb_id]['holds']]
        else:
            all_holds = [(h_id, info.get('x', 0), info.get('y', 0), info.get('role_id', -1))
                         for h_id, info in self.dataset.hold_info.items()]
        
        # Draw all holds as gray circles
        for hold_id, x, y, role_id in all_holds:
            color = 'gray'
            if role_id == 12:  # Start
                color = 'blue'
            elif role_id == 14:  # Finish
                color = 'red'
                
            plt.scatter(x, y, color=color, alpha=0.3, s=50)
        
        # Draw sequence
        colors = {'RH': 'red', 'LH': 'blue', 'RF': 'green', 'LF': 'purple'}
        
        for i, move in enumerate(sequence):
            x, y = move.get('x', 0), move.get('y', 0)
            
            plt.scatter(x, y, color=colors.get(move['limb'], 'black'), s=100, zorder=10)
            plt.text(x, y, f"{i+1}", fontsize=10, ha='center', va='center', 
                   bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))
        
        # Connect holds with arrows
        for i in range(1, len(sequence)):
            prev_x, prev_y = sequence[i-1].get('x', 0), sequence[i-1].get('y', 0)
            curr_x, curr_y = sequence[i].get('x', 0), sequence[i].get('y', 0)
            
            plt.arrow(prev_x, prev_y, curr_x-prev_x, curr_y-prev_y, 
                    head_width=0.5, head_length=0.7, fc=colors.get(sequence[i]['limb'], 'gray'), 
                    ec=colors.get(sequence[i]['limb'], 'gray'), alpha=0.6)
        
        # Add legend
        limb_labels = [plt.Line2D([0], [0], marker='o', color='w', 
                               markerfacecolor=color, markersize=10, label=limb)
                     for limb, color in colors.items()]
        
        plt.legend(handles=limb_labels, loc='upper right')
        
        # Set labels and title
        plt.xlabel('X Position')
        plt.ylabel('Y Position')
        plt.title('Climbing Sequence Visualization')
        
        # Add sequence statistics
        stats = self._calculate_sequence_stats(sequence)
        if stats:
            stat_text = "\n".join([f"{k}: {v:.1f}" if isinstance(v, float) else f"{k}: {v}" 
                                for k, v in stats.items()])
            plt.figtext(0.02, 0.02, stat_text, fontsize=10, 
                      bbox=dict(facecolor='white', alpha=0.7, edgecolor='gray'))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
            print(f"Visualization saved to {save_path}")
        else:
            plt.show()
            
        plt.close()
        
    def _calculate_sequence_stats(self, sequence):
        if not sequence:
            return {}
        
        coords = [(move['x'], move['y']) for move in sequence 
                 if 'x' in move and 'y' in move]
        
        if len(coords) < 2:
            return {}
            
        #distances between consecutive holds
        distances = []
        for i in range(1, len(coords)):
            x1, y1 = coords[i-1]
            x2, y2 = coords[i]
            dist = np.sqrt((x2-x1)**2 + (y2-y1)**2)
            distances.append(dist)
        
        #limb usage count
        limb_counts = defaultdict(int)
        for move in sequence:
            limb_counts[move['limb']] += 1
        unique_holds = len(set(move['hold'] for move in sequence))
        
        return {
            'avg_distance': np.mean(distances) if distances else 0,
            'max_distance': np.max(distances) if distances else 0,
            'unique_holds': unique_holds,
            'sequence_length': len(sequence),
            'hand_moves': limb_counts.get('RH', 0) + limb_counts.get('LH', 0),
            'foot_moves': limb_counts.get('RF', 0) + limb_counts.get('LF', 0),
            'height_gain': coords[-1][1] - coords[0][1] if len(coords) > 1 else 0,
            'width_span': max(x for x, _ in coords) - min(x for x, _ in coords) if coords else 0,
        }
        
    def export_sequences(self, sequences, output_file="generated_sequences.json"):
        output = {
            "results": [
                {
                    "id": f"{seq['climb_id']}_gen_{i}",
                    "name": f"Generated for {seq['climb_name']}",
                    "best_sequence": {
                        "sequence": [
                            {"hold": move["hold"], "limb": move["limb"]} 
                            for move in seq["sequence"]
                        ]
                    }
                }
                for i, seq in enumerate(sequences)
                if seq['is_valid']
            ]
        }
        
        with open(output_file, 'w') as f:
            json.dump(output, f, indent=2)
            
        print(f"Exported {len(output['results'])} sequences to {output_file}")
        return output_file