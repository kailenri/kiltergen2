import json
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from collections import defaultdict
import matplotlib.pyplot as plt
import random
from torch.nn import functional as F
from viz import plot_climb_sequence, plot_sequence_cycle, plot_reachability_map, plot_hold_density
from config import (
    MAX_HAND_REACH,
    MAX_FOOT_REACH,
    X_SPACING,
    FOOT_CUT_ENABLED,
    FOOT_REPLANT_MAX_STEPS,
)
from kinematics import (
    estimate_hip as _kin_estimate_hip,
    estimate_shoulder as _kin_estimate_shoulder,
    ellipse_dnorm as _kin_ellipse_dnorm,
    convex_hull as _kin_convex_hull,
    point_in_poly as _kin_point_in_poly,
    is_dynamic_hand_move,
    choose_cut_foot,
)

# ---------------------------------------------------------------------------
# Difficulty conditioning helpers
# ---------------------------------------------------------------------------

NUM_DIFFICULTY_BUCKETS = 5


def difficulty_to_bucket(d):
    """Map a raw Kilter difficulty integer (1–39) to a 1-based bucket (1–5).
    Bucket 0 is reserved as the 'unknown' / no-conditioning padding index.

    Mapping (from difficulty_grades table, is_listed grades):
      1 → V0–V2  (difficulty  1–15)
      2 → V3–V5  (difficulty 16–20)
      3 → V6–V8  (difficulty 21–24)
      4 → V9–V12 (difficulty 25–28)
      5 → V13+   (difficulty 29+)
    """
    if d is None:
        return 0
    d = int(round(float(d)))
    if d <= 15: return 1
    if d <= 20: return 2
    if d <= 24: return 3
    if d <= 28: return 4
    return 5


class ClimbDataset(Dataset):
    def __init__(self, json_file, max_sequence_length=50, vocab_path=None):
        print(f"Loading dataset from {json_file}")
        with open(json_file) as f:
            data = json.load(f)
        
        self.sequences = []
        self.sequence_difficulties = []   # parallel list: difficulty bucket per sequence
        self.hold_mapping = {'PAD': 0}
        self.reverse_hold_mapping = {0: 'PAD'}
        self.limb_mapping = {'RH': 0, 'LH': 1, 'RF': 2, 'LF': 3}
        self.role_mapping = {12: 'Start', 13: 'Hand', 14: 'Finish', 15: 'Foot'}
        self.max_sequence_length = max_sequence_length
        self.hold_info = {}

        if vocab_path:
            # load an existing vocabulary to ensure reproducible token ids
            self.load_vocab(vocab_path)
        else:
            self._build_vocabulary(data)
        self._populate_hold_info(data)
        self._create_sequences(data)
        self.augment_sequences()
        
        self.vocab_size = (len(self.hold_mapping) * len(self.limb_mapping)) + 1
        print(f"Vocabulary size: {self.vocab_size}")

    def save_vocab(self, path):
        payload = {
            'hold_mapping': {str(k): v for k, v in self.hold_mapping.items()},
            'reverse_hold_mapping': {str(k): v for k, v in self.reverse_hold_mapping.items()},
            'limb_mapping': self.limb_mapping,
            'role_mapping': self.role_mapping
        }
        with open(path, 'w') as f:
            json.dump(payload, f, indent=2)

    def load_vocab(self, path):
        with open(path) as f:
            payload = json.load(f)

        # JSON keys are strings; convert to ints where appropriate
        hold_mapping = {}
        for k, v in payload.get('hold_mapping', {}).items():
            try:
                key = int(k)
            except Exception:
                key = k
            hold_mapping[key] = v

        reverse_mapping = {}
        for k, v in payload.get('reverse_hold_mapping', {}).items():
            try:
                key = int(k)
            except Exception:
                key = k
            reverse_mapping[int(key)] = v

        self.hold_mapping = hold_mapping
        self.reverse_hold_mapping = reverse_mapping
        self.limb_mapping = payload.get('limb_mapping', self.limb_mapping)
        self.role_mapping = payload.get('role_mapping', self.role_mapping)
        self.vocab_size = (len(self.hold_mapping) * len(self.limb_mapping)) + 1
        print(f"Loaded vocab from {path}. Vocab size: {self.vocab_size}")

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

        print(f"Built vocabulary with {len(self.hold_mapping)} holds")

    def _populate_hold_info(self, data):
        """Populate hold spatial info (x, y, role_id) from the data file.
        Called after vocab is built or loaded so it works in both paths.
        """
        for result in data['results']:
            if 'best_sequence' not in result:
                continue
            for h in result['best_sequence'].get('holds', []):
                hold_id = h.get('hole_id')
                if hold_id and hold_id not in self.hold_info:
                    self.hold_info[hold_id] = {
                        'x': h.get('x', 0),
                        'y': h.get('y', 0),
                        'role_id': h.get('role_id', 13),
                        'name': h.get('name', ''),
                        'climb_id': result.get('id', '')
                    }
        print(f"Loaded spatial info for {len(self.hold_info)} holds")

    def build_spatial_features(self):
        """Build a (vocab_size, 10) float tensor of per-token spatial features:
          [x_norm, y_norm, role_start, role_hand, role_finish, role_foot,
           limb_RH, limb_LH, limb_RF, limb_LF]
        Token 0 (PAD) is all zeros.  Returns None if hold_info is empty.
        """
        if not self.hold_info:
            return None

        x_vals = [info['x'] for info in self.hold_info.values()]
        y_vals = [info['y'] for info in self.hold_info.values()]
        x_min, x_max = min(x_vals), max(x_vals)
        y_min, y_max = min(y_vals), max(y_vals)
        x_range = (x_max - x_min) or 1.0
        y_range = (y_max - y_min) or 1.0

        role_order = [12, 13, 14, 15]   # Start, Hand, Finish, Foot
        limb_order = [0, 1, 2, 3]       # RH, LH, RF, LF
        n_limbs = len(self.limb_mapping)

        features = torch.zeros(self.vocab_size, 10)

        for hold_id, hold_token in self.hold_mapping.items():
            if hold_id == 'PAD':
                continue
            info = self.hold_info.get(hold_id, {})
            x_norm = (info.get('x', x_min) - x_min) / x_range
            y_norm = (info.get('y', y_min) - y_min) / y_range
            role_id = info.get('role_id', 13)
            role_oh = [float(role_id == r) for r in role_order]
            for limb_token in limb_order:
                token_id = (hold_token * n_limbs) + limb_token + 1
                if 0 < token_id < self.vocab_size:
                    limb_oh = [float(limb_token == lt) for lt in limb_order]
                    features[token_id] = torch.tensor([x_norm, y_norm] + role_oh + limb_oh)

        print(f"Built spatial feature tensor: {features.shape}")
        return features

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
                self.sequence_difficulties.append(difficulty_to_bucket(result.get('difficulty')))
        
        print(f"Processed {total_sequences} sequences, {valid_sequences} valid")

    def augment_sequences(self):
        original_count = len(self.sequences)
        new_sequences = []
        new_difficulties = []
        original_difficulties = list(self.sequence_difficulties)

        # --- Build spatial mirror lookup for spatially-correct mirroring ---
        # Mirror: limb L↔R AND hold position flipped about the wall's centre X.
        x_vals = [info['x'] for info in self.hold_info.values() if 'x' in info]
        wall_center_x = (min(x_vals) + max(x_vals)) / 2.0 if x_vals else None

        # (x, y) → hold_token  and  hold_token → (x, y)
        xy_to_token: dict = {}
        token_to_xy: dict = {}
        for hold_id, info in self.hold_info.items():
            token = self.hold_mapping.get(hold_id)
            if token is not None:
                xy = (info.get('x', 0), info.get('y', 0))
                xy_to_token[xy] = token
                token_to_xy[token] = xy

        def mirror_hold_token(hold_token):
            """Return the hold_token at the spatially mirrored position, or None."""
            if wall_center_x is None:
                return hold_token  # no spatial data; keep as-is
            xy = token_to_xy.get(hold_token)
            if xy is None:
                return None
            mirror_xy = (2.0 * wall_center_x - xy[0], xy[1])
            return xy_to_token.get(mirror_xy)  # None if no mirror hold exists

        # --- Spatially-correct mirroring ---
        # Swap L↔R limbs AND remap each hold to its mirrored counterpart.
        # Skip any sequence whose holds lack a mirror partner in the vocab.
        n_limbs = len(self.limb_mapping)
        for i, seq in enumerate(self.sequences[:original_count]):
            mirrored = []
            valid = True
            for token in seq:
                if token == 0:  # Padding
                    mirrored.append(0)
                    continue
                token -= 1
                hold_token = token // n_limbs
                limb_token = token % n_limbs

                m_hold_token = mirror_hold_token(hold_token)
                if m_hold_token is None:
                    valid = False
                    break

                # Swap left ↔ right limbs
                if limb_token in (self.limb_mapping['LH'], self.limb_mapping['RH']):
                    limb_token = 1 - limb_token          # RH(0) ↔ LH(1)
                elif limb_token in (self.limb_mapping['RF'], self.limb_mapping['LF']):
                    limb_token = 5 - limb_token          # RF(2) ↔ LF(3)

                new_token = (m_hold_token * n_limbs) + limb_token + 1
                mirrored.append(new_token)

            if valid:
                new_sequences.append(mirrored)
                new_difficulties.append(original_difficulties[i] if original_difficulties else 0)

        # --- Subsequence creation: valid prefixes only ---
        # Only keep the first half so every sub-sequence starts from the beginning
        # with known limb positions. The second half (mid-sequence onwards) is
        # omitted because it would start with unknown limb context.
        min_subseq_length = 8
        for i, seq in enumerate(self.sequences[:original_count]):
            actual_seq = [t for t in seq if t != 0]
            if len(actual_seq) >= min_subseq_length * 2:
                first_half = actual_seq[:len(actual_seq) // 2]
                if len(first_half) >= min_subseq_length:
                    padded = first_half + [0] * (self.max_sequence_length - len(first_half))
                    new_sequences.append(padded)
                    new_difficulties.append(original_difficulties[i] if original_difficulties else 0)

        # Jitter augmentation removed: randomly replacing tokens with arbitrary
        # tokens from the same climb ignores ordering constraints and produces
        # physically invalid transitions that pollute the training distribution.

        self.sequences.extend(new_sequences)
        self.sequence_difficulties.extend(new_difficulties)
        print(f"Augmented dataset from {original_count} to {len(self.sequences)} sequences")

    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        sequence = self.sequences[idx]
        difficulty = self.sequence_difficulties[idx] if self.sequence_difficulties else 0
        return (
            torch.tensor(sequence[:-1], dtype=torch.long),
            torch.tensor(sequence[1:], dtype=torch.long),
            torch.tensor(difficulty, dtype=torch.long),
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


class HoldEncoder(nn.Module):
    """Factorised spatial encoder for (hold, limb) token pairs.

    Rather than projecting a flat 10-dim per-token feature vector that
    duplicates hold geometry across all four limbs on the same hold, this
    module learns a hold-level representation shared across limbs and
    combines it with a learned kinematic limb embedding.

    Architecture:
        hold_mlp : (hold_feat_dim=6) → 32 → hold_embed_dim  (per physical hold)
        limb_emb : Embedding(4, limb_embed_dim)              (per kinematic limb)
        output   : cat(hold_emb, limb_emb)  →  hold_embed_dim + limb_embed_dim
    """

    def __init__(
        self,
        hold_feat_dim: int = 6,
        hold_embed_dim: int = 32,
        num_limbs: int = 4,
        limb_embed_dim: int = 8,
    ):
        super().__init__()
        self.hold_mlp = nn.Sequential(
            nn.Linear(hold_feat_dim, 32),
            nn.ReLU(),
            nn.Linear(32, hold_embed_dim),
            nn.ReLU(),
        )
        self.limb_embedding = nn.Embedding(num_limbs, limb_embed_dim)
        self.output_dim = hold_embed_dim + limb_embed_dim

    def forward(self, hold_features: torch.Tensor, limb_indices: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hold_features : (..., hold_feat_dim) float — spatial hold attributes
            limb_indices  : (...) long — 0=RH, 1=LH, 2=RF, 3=LF
        Returns:
            (..., output_dim) float
        """
        hold_emb = self.hold_mlp(hold_features)       # (..., hold_embed_dim)
        limb_emb = self.limb_embedding(limb_indices)   # (..., limb_embed_dim)
        return torch.cat([hold_emb, limb_emb], dim=-1) # (..., output_dim)


class ClimbLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256, num_layers=2, dropout=0.2,
                 spatial_features=None, spatial_proj_dim=32,
                 num_difficulty_buckets=0, difficulty_embed_dim=16,
                 num_holds=0):
        super().__init__()
        self.num_layers = num_layers
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)

        # Spatial branch: HoldEncoder factorises spatial conditioning into a
        # hold-level MLP (shared across all limbs on the same hold) and a
        # learned limb identity embedding, replacing the flat per-token
        # spatial_proj with a more expressive, parameter-efficient module.
        #
        # hold_features buffer: (vocab_size, 6) = first 6 dims of spatial_features
        #   [x_norm, y_norm, role_start, role_hand, role_finish, role_foot]
        # Limb identity is captured by HoldEncoder's own Embedding rather than
        # the last 4 one-hot columns of the original 10-dim feature.
        _HOLD_FEAT_DIM  = 6
        _HOLD_EMBED_DIM = 32
        _LIMB_EMBED_DIM = 8
        lstm_input_dim = embedding_dim
        if spatial_features is not None:
            hold_feats = spatial_features[:, :_HOLD_FEAT_DIM].clone()
            self.register_buffer('hold_features', hold_feats)  # (vocab_size, 6)
            self.hold_encoder = HoldEncoder(
                hold_feat_dim=_HOLD_FEAT_DIM,
                hold_embed_dim=_HOLD_EMBED_DIM,
                num_limbs=4,
                limb_embed_dim=_LIMB_EMBED_DIM,
            )
            lstm_input_dim = embedding_dim + self.hold_encoder.output_dim  # 128+40=168
        else:
            self.hold_features = None
            self.hold_encoder  = None
        # Kept as None so any code that checks spatial_proj / spatial_features
        # will not crash when loading a Phase D checkpoint.
        self.spatial_features = None
        self.spatial_proj     = None

        self.lstm = nn.LSTM(
            lstm_input_dim, hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )
        # LayerNorm is correct for sequence models: it normalises per time-step
        # using statistics from that step alone, so batch-size=1 inference
        # behaves identically to batch training (unlike BatchNorm1d).
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

        # Auxiliary output heads: separate hold and limb predictors provide
        # factorised gradient signal during training without changing inference.
        if num_holds > 0:
            self.fc_hold = nn.Linear(hidden_dim, num_holds)
            self.fc_limb = nn.Linear(hidden_dim, 4)
        else:
            self.fc_hold = None
            self.fc_limb = None

        # Difficulty conditioning: condition the LSTM's initial (h, c) on a
        # bucketed difficulty level.  Padding index 0 = 'unknown' → zero vector.
        if num_difficulty_buckets > 0:
            self.difficulty_embedding = nn.Embedding(
                num_difficulty_buckets + 1, difficulty_embed_dim, padding_idx=0
            )
            self.difficulty_h_proj = nn.Linear(difficulty_embed_dim, num_layers * hidden_dim)
            self.difficulty_c_proj = nn.Linear(difficulty_embed_dim, num_layers * hidden_dim)
        else:
            self.difficulty_embedding = None
            self.difficulty_h_proj = None
            self.difficulty_c_proj = None

        self._init_weights()

    def _init_weights(self):
        init_range = 0.1
        self.embedding.weight.data.uniform_(-init_range, init_range)
        self.fc.weight.data.uniform_(-init_range, init_range)
        self.fc.bias.data.zero_()
        if self.hold_encoder is not None:
            self.hold_encoder.limb_embedding.weight.data.uniform_(-init_range, init_range)
        if self.fc_hold is not None:
            self.fc_hold.weight.data.uniform_(-init_range, init_range)
            self.fc_hold.bias.data.zero_()
            self.fc_limb.weight.data.uniform_(-init_range, init_range)
            self.fc_limb.bias.data.zero_()

    def forward(self, x, hidden=None, difficulty=None, return_aux=False):
        embedded = self.embedding(x)

        if self.hold_features is not None and self.hold_encoder is not None:
            hold_feat   = self.hold_features[x]                     # (batch, seq_len, 6)
            limb_idx    = (x - 1).clamp(min=0) % 4                 # (batch, seq_len) long
            spatial_emb = self.hold_encoder(hold_feat, limb_idx)   # (batch, seq_len, 40)
            embedded = torch.cat([embedded, spatial_emb], dim=-1)

        # Condition initial hidden state on difficulty (first step only — when
        # hidden is None — so subsequent generation steps are unaffected).
        if self.difficulty_embedding is not None and difficulty is not None and hidden is None:
            d_emb = self.difficulty_embedding(difficulty)   # (batch, diff_dim)
            h0 = self.difficulty_h_proj(d_emb).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
            c0 = self.difficulty_c_proj(d_emb).view(-1, self.num_layers, self.hidden_dim).permute(1, 0, 2).contiguous()
            hidden = (h0, c0)

        lstm_out, hidden = self.lstm(embedded, hidden)
        out = self.dropout(self.layer_norm(lstm_out))
        composite = self.fc(out)
        if return_aux and self.fc_hold is not None:
            return composite, self.fc_hold(out), self.fc_limb(out), hidden
        return composite, hidden

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
    def __init__(self, json_file, vocab_path=None):
        print(f"Initializing ClimbGenerator")
        print(f"Data file: {json_file}")
        
        self.dataset = ClimbDataset(json_file, vocab_path=vocab_path)
        self.model = None
        self.climb_data = self._load_climb_data(json_file)
        self.role_mapping = {
            12: 'Start',
            13: 'Hand',
            14: 'Finish',
            15: 'Foot'
        }
        self.debug_reachability = os.getenv('LSTM_REACH_DEBUG', '').lower() in ('1', 'true', 'yes')
        if self.debug_reachability:
            print("Reachability debug logging enabled")
        
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
            if torch.cuda.is_available():
                device = torch.device('cuda')
            elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
                device = torch.device('mps')
            else:
                device = torch.device('cpu')
            
            self.model = ClimbLSTM(
                vocab_size=self.dataset.vocab_size,
                embedding_dim=128,
                hidden_dim=256,
                num_layers=2,
                dropout=0.2,
                spatial_features=self.dataset.build_spatial_features(),
                num_difficulty_buckets=NUM_DIFFICULTY_BUCKETS,
                num_holds=len(self.dataset.hold_mapping),
            ).to(device)
            
            state_dict = torch.load(model_path, map_location=device)
            self.model.load_state_dict(state_dict)
            self.model.eval()
            
            print(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            print(f"Error loading model: {e}")
            return False

    def train(self, num_epochs=30, batch_size=32, learning_rate=0.001, save_path='lstm_model.pth', checkpoint_dir=None, checkpoint_freq=5, tb_logdir=None, reinforce_weight=0.0):
        train_data, val_data = train_test_split(self.dataset, test_size=0.2, random_state=42, shuffle=True)
        
        #Set up data loaders
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size)
        
        if torch.cuda.is_available():
            device = torch.device('cuda')
        elif getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available():
            device = torch.device('mps')
        else:
            device = torch.device('cpu')
        print(f"Training on: {device}")
        
        #Initialize model
        self.model = ClimbLSTM(
            vocab_size=self.dataset.vocab_size,
            embedding_dim=128,
            hidden_dim=256,
            num_layers=2,
            dropout=0.2,
            spatial_features=self.dataset.build_spatial_features(),
            num_difficulty_buckets=NUM_DIFFICULTY_BUCKETS,
            num_holds=len(self.dataset.hold_mapping),
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

        # Setup checkpointing and TensorBoard
        writer = None
        if tb_logdir:
            try:
                from torch.utils.tensorboard import SummaryWriter
                writer = SummaryWriter(tb_logdir)
                print(f"TensorBoard logging to: {tb_logdir}")
            except Exception as e:
                print(f"Warning: failed to initialize TensorBoard writer: {e}")

        if checkpoint_dir:
            os.makedirs(checkpoint_dir, exist_ok=True)
        
        #Training loop
        train_losses, val_losses = [], []
        
        for epoch in range(num_epochs):
            self.model.train()
            epoch_loss = 0
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
            for batch_idx, (inputs, targets, difficulties) in enumerate(progress_bar):
                inputs, targets, difficulties = inputs.to(device), targets.to(device), difficulties.to(device)
                
                #Raise on invalid tokens — catches data pipeline bugs early
                if (inputs >= self.dataset.vocab_size).any() or (targets >= self.dataset.vocab_size).any():
                    raise ValueError(
                        f"Invalid token ID found in training batch "
                        f"(vocab_size={self.dataset.vocab_size}). "
                        "Re-build vocabulary or re-export training data."
                    )
                
                optimizer.zero_grad()

                # Scheduled sampling: teacher-forcing ratio decays linearly
                # from 1.0 → 0.5 over training, reducing exposure bias.
                tf_ratio = max(0.5, 1.0 - 0.5 * epoch / num_epochs)
                seq_len = inputs.shape[1]
                hidden = None
                logits_list, hold_list, limb_list = [], [], []
                current_input = inputs[:, 0:1]
                for t in range(seq_len):
                    logit_t, hold_t, limb_t, hidden = self.model(
                        current_input, hidden, difficulty=difficulties, return_aux=True
                    )
                    logits_list.append(logit_t)
                    hold_list.append(hold_t)
                    limb_list.append(limb_t)
                    if t < seq_len - 1:
                        if random.random() < tf_ratio:
                            current_input = inputs[:, t + 1:t + 2]
                        else:
                            with torch.no_grad():
                                current_input = logit_t.argmax(dim=-1)

                outputs     = torch.cat(logits_list, dim=1)
                hold_logits = torch.cat(hold_list,   dim=1)
                limb_logits = torch.cat(limb_list,   dim=1)

                loss_main = criterion(
                    outputs.reshape(-1, self.dataset.vocab_size), targets.reshape(-1)
                )

                # Auxiliary factorised losses on non-PAD positions.
                # Encoding: token = (hold_token * 4) + limb_token + 1
                # → hold_token = (token-1)//4,  limb_token = (token-1)%4
                num_holds = len(self.dataset.hold_mapping)
                hold_targets = torch.zeros_like(targets)
                limb_targets = torch.full_like(targets, -1)
                valid_mask = targets > 0
                hold_targets[valid_mask] = (targets[valid_mask] - 1) // 4
                limb_targets[valid_mask] = (targets[valid_mask] - 1) % 4
                loss_hold = F.cross_entropy(
                    hold_logits.reshape(-1, num_holds), hold_targets.reshape(-1), ignore_index=0
                )
                loss_limb = F.cross_entropy(
                    limb_logits.reshape(-1, 4), limb_targets.reshape(-1), ignore_index=-1
                )

                loss = loss_main + 0.3 * loss_hold + 0.3 * loss_limb
                loss.backward()
                
                #Gradient clipping
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                
                optimizer.step()
                scheduler.step()

                # REINFORCE step: every 10 batches when reinforce_weight > 0.
                # Separate backward pass so CE gradients are not disturbed.
                if reinforce_weight > 0.0 and batch_idx % 10 == 0:
                    self._reinforce_update(device, optimizer, reinforce_weight)

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

            # TensorBoard logging
            if writer:
                try:
                    writer.add_scalar('loss/train', avg_train_loss, epoch + 1)
                    writer.add_scalar('loss/val', avg_val_loss, epoch + 1)
                    # log learning rate
                    try:
                        lr = scheduler.get_last_lr()[0]
                    except Exception:
                        lr = optimizer.param_groups[0]['lr']
                    writer.add_scalar('learning_rate', lr, epoch + 1)
                except Exception as e:
                    print(f"Warning: TensorBoard write failed: {e}")

            # Periodic checkpointing
            if checkpoint_dir and ((epoch + 1) % checkpoint_freq == 0):
                chk_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
                try:
                    torch.save({
                        'epoch': epoch + 1,
                        'model_state': self.model.state_dict(),
                        'optimizer_state': optimizer.state_dict(),
                        'scheduler_state': scheduler.state_dict(),
                        'val_loss': avg_val_loss
                    }, chk_path)
                    print(f"Saved checkpoint: {chk_path}")
                except Exception as e:
                    print(f"Warning: failed to save checkpoint: {e}")

            #Check for early stopping
            if early_stopping(avg_val_loss, self.model):
                print(f"Early stopping triggered at epoch {epoch+1}")
                # save best weights as checkpoint
                if checkpoint_dir:
                    best_path = os.path.join(checkpoint_dir, 'checkpoint_best.pth')
                    try:
                        torch.save({
                            'epoch': epoch + 1,
                            'model_state': early_stopping.best_weights,
                            'optimizer_state': optimizer.state_dict(),
                            'scheduler_state': scheduler.state_dict(),
                            'val_loss': early_stopping.best_loss
                        }, best_path)
                        print(f"Saved best checkpoint: {best_path}")
                    except Exception as e:
                        print(f"Warning: failed to save best checkpoint: {e}")

                early_stopping.restore_weights(self.model)
                break
            
            #Generate a sample sequence every 5 epochs
            if (epoch + 1) % 5 == 0:
                self._generate_sample(device, writer=writer, epoch=epoch + 1)
        
        #Ensure best weights are used
        if not early_stopping.stopped_epoch and early_stopping.best_weights is not None:
            self.model.load_state_dict(early_stopping.best_weights)
        
        #Save the model
        torch.save(self.model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

        # Save a final checkpoint
        if checkpoint_dir:
            final_path = os.path.join(checkpoint_dir, 'checkpoint_final.pth')
            try:
                torch.save({
                    'epoch': epoch + 1,
                    'model_state': self.model.state_dict(),
                    'optimizer_state': optimizer.state_dict(),
                    'scheduler_state': scheduler.state_dict(),
                    'val_loss': val_losses[-1] if val_losses else None
                }, final_path)
                print(f"Saved final checkpoint: {final_path}")
            except Exception as e:
                print(f"Warning: failed to save final checkpoint: {e}")

        # Close TensorBoard writer
        if writer:
            try:
                writer.close()
            except Exception:
                pass
        
        #Plot training history
        self._plot_training_history(train_losses, val_losses)
        
        return train_losses, val_losses

    def _evaluate(self, loader, criterion, device):
        """Evaluate the model on a data loader"""
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for inputs, targets, difficulties in loader:
                inputs, targets, difficulties = inputs.to(device), targets.to(device), difficulties.to(device)
                
                #Skip batch if it contains invalid tokens
                if (inputs >= self.dataset.vocab_size).any() or (targets >= self.dataset.vocab_size).any():
                    raise ValueError(
                        f"Invalid token ID found in validation batch "
                        f"(vocab_size={self.dataset.vocab_size}). "
                        "Re-build vocabulary or re-export training data."
                    )
                
                outputs, _ = self.model(inputs, difficulty=difficulties)
                loss = criterion(outputs.reshape(-1, self.dataset.vocab_size), targets.reshape(-1))
                total_loss += loss.item()
        
        return total_loss

    def _generate_sample(self, device, temp=0.8, writer=None, epoch=None):
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

                    # Log quality metrics to TensorBoard when available
                    if writer is not None and epoch is not None:
                        hand_moves = [m for m in decoded if m.get('limb') in ('RH', 'LH')]
                        has_finish = any(m.get('role_id') == 14 for m in decoded[-2:])
                        coords = [(m.get('x', 0), m.get('y', 0)) for m in hand_moves]
                        dists = [
                            math.hypot(coords[i][0] - coords[i-1][0], coords[i][1] - coords[i-1][1])
                            for i in range(1, len(coords))
                        ]
                        role_valid = sum(1 for m in hand_moves if m.get('role_id') in {12, 13, 14})
                        writer.add_scalar('sample/validity_rate', float(has_finish), epoch)
                        writer.add_scalar('sample/avg_hand_move_dist',
                                          sum(dists) / len(dists) if dists else 0.0, epoch)
                        writer.add_scalar('sample/hand_role_accuracy',
                                          role_valid / len(hand_moves) if hand_moves else 0.0, epoch)
                        writer.add_scalar('sample/seq_length', len(decoded), epoch)

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

    def generate(self, climb_id=None, num_sequences=3, temperature=0.8, max_length=30, difficulty=None):
        if not self.model:
            print("Error: Model not loaded")
            return []
        
        device = next(self.model.parameters()).device
        diff_bucket = difficulty_to_bucket(difficulty) if difficulty is not None else 0
        diff_tensor = torch.tensor([diff_bucket], dtype=torch.long).to(device)
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

            finish_token_ids = set()
            finish_bias = 1.25
            if current_id in self.climb_data:
                finish_holds = [
                    h.get('hole_id')
                    for h in self.climb_data[current_id]['holds']
                    if h.get('role_id') == 14
                ]
                for hold_id in finish_holds:
                    hold_token = self.dataset.hold_mapping.get(hold_id)
                    if hold_token is None:
                        continue
                    for limb in ('RH', 'LH'):
                        limb_token = self.dataset.limb_mapping.get(limb)
                        if limb_token is None:
                            continue
                        token_id = (hold_token * len(self.dataset.limb_mapping)) + limb_token + 1
                        finish_token_ids.add(token_id)

            hold_map = {h.get('hole_id'): h for h in self.climb_data[current_id]['holds']}
            finish_holds = {
                h.get('hole_id')
                for h in self.climb_data[current_id]['holds']
                if h.get('role_id') == 14
            }
            
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
            limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
            used_hold_ids = set()
            cut_feet = set()
            for hold_id, limb in start_holds:
                if hold_id in self.dataset.hold_mapping:
                    hold_token = self.dataset.hold_mapping[hold_id]
                    limb_token = self.dataset.limb_mapping[limb]
                    current_tokens.append((hold_token * len(self.dataset.limb_mapping)) + limb_token + 1)
                hold = hold_map.get(hold_id)
                if hold:
                    limb_positions[limb] = (hold.get('x', 0), hold.get('y', 0))
                    used_hold_ids.add(hold_id)

            if not current_tokens:
                print(f"Could not create start tokens for climb {current_id}")
                continue
                
            print(f"Using {len(current_tokens)} start tokens")
            # Minimum moves that must be generated before the finish-seeker
            # is allowed to terminate the sequence.  Without this guard the
            # model can finish after just 1-2 hand moves whenever a finish
            # hold happens to be reachable from the start position.
            num_start_tokens = len(current_tokens)
            min_moves_before_finish = num_start_tokens + 6

            rev_limb_mapping = {v: k for k, v in self.dataset.limb_mapping.items()}
            last_hand = None

            def decode_token(token_id):
                if token_id == 0:
                    return None
                token_id -= 1
                hold_token = token_id // len(self.dataset.limb_mapping)
                limb_token = token_id % len(self.dataset.limb_mapping)
                hold_id = self.dataset.reverse_hold_mapping.get(hold_token)
                limb = rev_limb_mapping.get(limb_token)
                if hold_id is None or limb is None:
                    return None
                hold = hold_map.get(hold_id, {})
                return {
                    'hold': hold_id,
                    'limb': limb,
                    'role_id': hold.get('role_id'),
                    'x': hold.get('x', 0),
                    'y': hold.get('y', 0),
                    'name': hold.get('name', '')
                }

            #Generate the sequence
            sequence_tokens = current_tokens.copy()
            
            try:
                with torch.no_grad():
                    input_seq = torch.tensor([current_tokens], dtype=torch.long).to(device)
                    output, hidden = self.model(input_seq, difficulty=diff_tensor)
                    
                    for _ in range(max_length):
                        logits = output[0, -1, :] / temperature
                        logits[0] = float('-inf')

                        if finish_token_ids:
                            for token_id in finish_token_ids:
                                if token_id < logits.shape[0]:
                                    logits[token_id] += finish_bias

                        # Finish-seeking: force a reachable finish when possible,
                        # but only once enough moves have been generated so we
                        # don't short-circuit to the finish from the start.
                        next_hand = 'LH' if last_hand == 'RH' else 'RH'
                        finish_move = None
                        if len(sequence_tokens) >= min_moves_before_finish:
                            finish_move = self._select_reachable_finish(
                            next_hand,
                            limb_positions,
                            hold_map,
                            finish_holds,
                            reach_scale=1.1,
                        )
                        if finish_move:
                            hold_token = self.dataset.hold_mapping.get(finish_move['hold'])
                            limb_token = self.dataset.limb_mapping.get(next_hand)
                            if hold_token is not None and limb_token is not None:
                                next_token = (hold_token * len(self.dataset.limb_mapping)) + limb_token + 1
                                sequence_tokens.append(next_token)
                                limb_positions[next_hand] = (finish_move.get('x', 0), finish_move.get('y', 0))
                                used_hold_ids.add(finish_move.get('hold'))
                                last_hand = next_hand
                            # Finish reached — stop generating.
                            break
                        
                        # Constrained decoding: mask all invalid tokens in one
                        # O(|climb_holds| × 4) pass — no rejection loop needed.
                        valid_mask = self._build_valid_token_mask(
                            limb_positions, last_hand, hold_map, used_hold_ids, cut_feet
                        )
                        device_mask = valid_mask.to(logits.device)
                        logits[~device_mask] = float('-inf')

                        if not device_mask.any():
                            # Replant deadlock: no cut foot can reach any hold.
                            # Treat as toe-match — clear cut_feet and rebuild
                            # the mask so generation can continue with hands.
                            if cut_feet:
                                cut_feet.clear()
                                valid_mask = self._build_valid_token_mask(
                                    limb_positions, last_hand, hold_map, used_hold_ids, cut_feet
                                )
                                device_mask = valid_mask.to(logits.device)
                                logits[~device_mask] = float('-inf')
                            if not device_mask.any():
                                break

                        # Top-p (nucleus) sampling within the valid set
                        sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                        cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        sorted_indices_to_remove = cumulative_probs > 0.9
                        sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                        sorted_indices_to_remove[..., 0] = 0
                        logits[sorted_indices[sorted_indices_to_remove]] = float('-inf')

                        next_token = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1).item()
                        if next_token == 0:
                            break

                        move = decode_token(next_token)
                        if move:
                            mv_limb = move['limb']
                            new_xy = (move.get('x', 0), move.get('y', 0))

                            # Foot-cut / replant bookkeeping mirroring the beam
                            # generator: a dynamic hand move may cut a foot,
                            # and the next foot move re-plants it.
                            if FOOT_CUT_ENABLED:
                                if mv_limb in ('RH', 'LH'):
                                    prev_hand_xy = limb_positions[mv_limb]
                                    foot_xys = [limb_positions[f] for f in ('RF', 'LF')
                                                if limb_positions[f] is not None]
                                    other_hand = 'LH' if mv_limb == 'RH' else 'RH'
                                    other_hand_xy = limb_positions[other_hand]
                                    if is_dynamic_hand_move(prev_hand_xy, new_xy, foot_xys,
                                                            mv_limb, other_hand_xy):
                                        foot_dict = {f: limb_positions[f] for f in ('RF', 'LF')}
                                        cut = choose_cut_foot(mv_limb, foot_dict)
                                        if cut is not None:
                                            cut_feet.add(cut)
                                            limb_positions[cut] = None
                                elif mv_limb in cut_feet:
                                    cut_feet.discard(mv_limb)

                            limb_positions[mv_limb] = new_xy
                            used_hold_ids.add(move.get('hold'))
                            if mv_limb in ('RH', 'LH'):
                                last_hand = mv_limb

                            # Stop immediately if we landed on a finish hold.
                            if move.get('role_id') == 14:
                                sequence_tokens.append(next_token)
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
        
        #make sure there is hand alternation (filter, not relabel)
        # Relabeling the limb of a duplicate move is wrong: it doesn't update
        # the position tracking and may produce an unreachable move under a
        # different limb.  Dropping the move is safer; the reachability filter
        # below will prune any further invalid moves.
        last_hand = None
        alternated = []

        for move in cleaned:
            if move['limb'] in ('RH', 'LH'):
                if last_hand == move['limb']:
                    continue  # drop duplicate-hand move instead of relabeling
                last_hand = move['limb']
            alternated.append(move)

        #reachability filtering per limb
        if climb_id in self.climb_data and self.climb_data[climb_id]['holds']:
            hold_map = {h.get('hole_id'): h for h in self.climb_data[climb_id]['holds']}
        else:
            hold_map = {hid: info for hid, info in self.dataset.hold_info.items()}

        limb_positions = {'RH': None, 'LH': None, 'RF': None, 'LF': None}
        reachable = []

        for move in alternated:
            hold = hold_map.get(move['hold'])
            if hold:
                move['x'] = hold.get('x', move.get('x', 0))
                move['y'] = hold.get('y', move.get('y', 0))
                move['role_id'] = hold.get('role_id', move.get('role_id'))
                move['name'] = hold.get('name', move.get('name', ''))

            limb = move.get('limb')
            if limb in ('RH', 'LH'):
                if hold and hold.get('role_id') not in {12, 13, 14}:
                    self._log_reachability(move, 'hand_role_invalid')
                    continue
                ok, reason = self._dynamic_hand_reachable(hold, limb, limb_positions)
                if not ok:
                    self._log_reachability(move, reason)
                    continue
                max_reach = MAX_HAND_REACH
            else:
                max_reach = MAX_FOOT_REACH

            prev_pos = limb_positions.get(limb)
            if prev_pos is not None and hold:
                dist = math.hypot(hold.get('x', 0) - prev_pos[0], hold.get('y', 0) - prev_pos[1])
                if dist > max_reach:
                    self._log_reachability(move, f"limb_dist {dist:.2f} > {max_reach:.2f}")
                    continue

            if hold:
                limb_positions[limb] = (hold.get('x', 0), hold.get('y', 0))
            reachable.append(move)
        
        #makes sure the sequence ends with a finish hold
        # Re-apply alternation after reachability filtering may have exposed
        # consecutive same-limb moves that the first pass already cleared.
        last_hand_re = None
        reachable_alt = []
        for move in reachable:
            if move['limb'] in ('RH', 'LH'):
                if last_hand_re == move['limb']:
                    continue
                last_hand_re = move['limb']
            reachable_alt.append(move)
        reachable = reachable_alt

        has_finish = any(m['hold'] in finish_holds for m in reachable[-2:]) if finish_holds else False
        
        #add finish if there is none
        if not has_finish and finish_holds:
            last_hand = reachable[-1]['limb'] if reachable and reachable[-1]['limb'] in ['RH', 'LH'] else 'RH'
            next_hand = 'LH' if last_hand == 'RH' else 'RH'

            finish_move = self._select_reachable_finish(next_hand, limb_positions, hold_map, finish_holds)
            if finish_move:
                reachable.append(finish_move)
            else:
                self._log_reachability({'limb': next_hand, 'hold': '?'}, 'finish_unreachable')
                fallback = self._select_progress_hand_hold(next_hand, limb_positions, hold_map, finish_holds)
                if fallback:
                    reachable.append(fallback)

        # If no finish is reached, allow up to four additional reachable moves to reach it.
        has_finish = any(m['hold'] in finish_holds for m in reachable[-2:]) if finish_holds else False
        if finish_holds and not has_finish:
            for _ in range(4):
                last_hand = reachable[-1]['limb'] if reachable and reachable[-1]['limb'] in ['RH', 'LH'] else 'RH'
                next_hand = 'LH' if last_hand == 'RH' else 'RH'
                finish_move = self._select_reachable_finish(next_hand, limb_positions, hold_map, finish_holds)
                if finish_move:
                    reachable.append(finish_move)
                    break

                progress_move = self._select_progress_hand_hold(next_hand, limb_positions, hold_map, finish_holds)
                if not progress_move:
                    break
                reachable.append(progress_move)
                limb_positions[next_hand] = (progress_move.get('x', 0), progress_move.get('y', 0))

        has_finish = any(m['hold'] in finish_holds for m in reachable[-2:]) if finish_holds else False
        if finish_holds and not has_finish:
            self._log_reachability({'limb': '?', 'hold': '?'}, 'finish_required')
            return []
        
        return reachable

    def _build_valid_token_mask(self, limb_positions, last_hand, hold_map, used_hold_ids, cut_feet=None):
        """Build a boolean mask of shape (vocab_size,) where True marks a token as a
        valid next move.  Replaces the 30-attempt rejection-sampling loop with a
        single O(|climb_holds| × 4) pass that writes directly into the logit tensor
        before sampling.

        When ``cut_feet`` is non-empty and ``FOOT_CUT_ENABLED`` is True, only foot
        re-plants for the cut limbs are allowed (hand tokens are masked) so that
        the climber must restore foot tension before reaching again.
        """
        n_limbs = len(self.dataset.limb_mapping)
        vocab_size = self.dataset.vocab_size
        valid = torch.zeros(vocab_size, dtype=torch.bool)
        gate_active = bool(cut_feet) and FOOT_CUT_ENABLED

        for hold in hold_map.values():
            hold_id = hold.get('hole_id')
            if hold_id is None:
                continue
            hold_token = self.dataset.hold_mapping.get(hold_id)
            if hold_token is None:
                continue
            role_id = hold.get('role_id')
            hx = hold.get('x', 0)
            hy = hold.get('y', 0)

            for limb_name, limb_token in self.dataset.limb_mapping.items():
                if limb_name in ('RH', 'LH'):
                    if gate_active:
                        continue  # hands blocked until cut feet are replanted
                    if role_id not in {12, 13, 14}:
                        continue
                    if last_hand == limb_name:
                        continue
                    if hold_id in used_hold_ids and role_id != 14:
                        continue
                    ok, _ = self._dynamic_hand_reachable(hold, limb_name, limb_positions)
                    if not ok:
                        continue
                    max_reach = MAX_HAND_REACH
                else:  # RF / LF
                    if gate_active and limb_name not in cut_feet:
                        continue
                    if role_id not in {15}:  # feet must go to foot holds only
                        continue
                    if hold_id in used_hold_ids and role_id != 14:
                        continue
                    max_reach = MAX_FOOT_REACH

                prev_pos = limb_positions.get(limb_name)
                if prev_pos is not None:
                    dist = math.hypot(hx - prev_pos[0], hy - prev_pos[1])
                    if dist > max_reach * 0.95:
                        continue

                token_id = (hold_token * n_limbs) + limb_token + 1
                if 0 < token_id < vocab_size:
                    valid[token_id] = True

        return valid

    def _dynamic_hand_reachable(self, hold, limb, limb_positions):
        if not hold:
            return False, 'missing_hold'

        foot_positions = [p for k, p in limb_positions.items() if k in ('RF', 'LF') and p is not None]
        # If no feet planted, fall back to a simple reach check.
        if not foot_positions:
            prev_pos = limb_positions.get(limb)
            if prev_pos is None:
                return True, None
            dist = math.hypot(hold.get('x', 0) - prev_pos[0], hold.get('y', 0) - prev_pos[1])
            if dist > MAX_HAND_REACH:
                return False, f"hand_dist {dist:.2f} > {MAX_HAND_REACH:.2f}"
            return True, None

        hip = self._estimate_hip(foot_positions)
        shoulder = self._estimate_shoulder(hip)
        rx = MAX_HAND_REACH * 0.9
        ry = MAX_HAND_REACH * 0.6 + (X_SPACING * 1.2)

        hold_xy = (hold.get('x', 0), hold.get('y', 0))
        dnorm = self._ellipse_dnorm(hold_xy, shoulder, rx, ry)
        if dnorm > 1.15:
            return False, f"ellipse_dnorm {dnorm:.2f} > 1.15"

        # stability: COM projection vs support polygon
        com = (hip[0], hip[1] + X_SPACING * 1.2)
        support_pts = list(foot_positions)
        for k in ('RH', 'LH'):
            if limb_positions.get(k) is not None and k != limb:
                support_pts.append(limb_positions[k])
        support_pts.append(hold_xy)

        if len(support_pts) >= 3:
            hull = self._convex_hull(support_pts)
            if not self._point_in_poly(com, hull) and dnorm > 1.05:
                return False, 'com_outside_support'

        return True, None

    def _select_reachable_finish(self, limb, limb_positions, hold_map, finish_holds, reach_scale=1.0):
        if not finish_holds:
            return None

        prev_pos = limb_positions.get(limb)
        candidates = []

        for hold_id in finish_holds:
            hold = hold_map.get(hold_id)
            if not hold:
                continue
            ok, _ = self._dynamic_hand_reachable(hold, limb, limb_positions)
            if not ok:
                continue
            if prev_pos is not None:
                dist = math.hypot(hold.get('x', 0) - prev_pos[0], hold.get('y', 0) - prev_pos[1])
                if dist > MAX_HAND_REACH * reach_scale:
                    continue
            candidates.append(hold)

        if not candidates:
            return None

        def sort_key(h):
            if prev_pos is None:
                return (-h.get('y', 0), 0.0)
            return (math.hypot(h.get('x', 0) - prev_pos[0], h.get('y', 0) - prev_pos[1]), -h.get('y', 0))

        best = sorted(candidates, key=sort_key)[0]
        return {
            'hold': best.get('hole_id'),
            'limb': limb,
            'role_id': 14,
            'x': best.get('x', 0),
            'y': best.get('y', 0),
            'name': best.get('name', '')
        }

    def _select_progress_hand_hold(self, limb, limb_positions, hold_map, finish_holds):
        occupied = {pos for pos in limb_positions.values() if pos is not None}
        prev_pos = limb_positions.get(limb)

        finish_positions = []
        for hold_id in finish_holds or []:
            hold = hold_map.get(hold_id)
            if hold:
                finish_positions.append((hold.get('x', 0), hold.get('y', 0)))

        def dist_to_finish(xy):
            if not finish_positions:
                return float('inf')
            return min(math.hypot(xy[0] - fx, xy[1] - fy) for fx, fy in finish_positions)

        candidates = []
        for hold in hold_map.values():
            if hold.get('role_id') not in {12, 13, 14}:
                continue
            hold_xy = (hold.get('x', 0), hold.get('y', 0))
            if hold_xy in occupied:
                continue
            ok, _ = self._dynamic_hand_reachable(hold, limb, limb_positions)
            if not ok:
                continue
            if prev_pos is not None:
                move_dist = math.hypot(hold_xy[0] - prev_pos[0], hold_xy[1] - prev_pos[1])
                if move_dist > MAX_HAND_REACH:
                    continue
            candidates.append(hold)

        if not candidates:
            self._log_reachability({'limb': limb, 'hold': '?'}, 'fallback_unavailable')
            return None

        def sort_key(h):
            h_xy = (h.get('x', 0), h.get('y', 0))
            move_dist = 0.0 if prev_pos is None else math.hypot(h_xy[0] - prev_pos[0], h_xy[1] - prev_pos[1])
            return (dist_to_finish(h_xy), -h.get('y', 0), move_dist)

        best = sorted(candidates, key=sort_key)[0]
        self._log_reachability({'limb': limb, 'hold': best.get('hole_id')}, 'fallback_selected')
        return {
            'hold': best.get('hole_id'),
            'limb': limb,
            'role_id': best.get('role_id', 13),
            'x': best.get('x', 0),
            'y': best.get('y', 0),
            'name': best.get('name', '')
        }

    def _log_reachability(self, move, reason):
        if not self.debug_reachability:
            return
        limb = move.get('limb', '?') if isinstance(move, dict) else '?'
        hold_id = move.get('hold', '?') if isinstance(move, dict) else '?'
        print(f"Reachability reject: limb={limb} hold={hold_id} reason={reason}")

    def _estimate_hip(self, foot_positions):
        return _kin_estimate_hip(list(foot_positions))

    def _estimate_shoulder(self, hip_xy):
        return _kin_estimate_shoulder(hip_xy)

    def _ellipse_dnorm(self, hold_xy, shoulder_xy, rx, ry):
        return _kin_ellipse_dnorm(hold_xy, shoulder_xy, rx, ry)

    def _convex_hull(self, points):
        return _kin_convex_hull(points)

    def _point_in_poly(self, pt, poly):
        return _kin_point_in_poly(pt, poly)

    # ------------------------------------------------------------------
    # REINFORCE helpers
    # ------------------------------------------------------------------

    def _compute_reinforce_loss(self, device):
        """REINFORCE episode: sample a sequence, score it, return policy-gradient loss.

        Two-phase approach for MPS compatibility and numerical stability:
          Phase 1 — sample tokens with torch.no_grad() (no autograd graph built).
          Phase 2 — recompute log-probs in ONE teacher-forced forward pass with
                    gradients, so the autograd graph is shallow and MPS-safe.
        """
        climb_ids = list(self.climb_data.keys())
        if not climb_ids:
            return None

        climb_id     = random.choice(climb_ids)
        climb        = self.climb_data.get(climb_id, {})
        hold_map     = {h['hole_id']: h for h in climb.get('holds', [])}
        finish_holds = {h['hole_id'] for h in climb.get('holds', []) if h.get('role_id') == 14}
        start_holds  = [h['hole_id'] for h in climb.get('holds', []) if h.get('role_id') == 12]

        if not start_holds or not finish_holds:
            return None

        # Build seed tokens from start holds (up to 2).
        n_limbs     = len(self.dataset.limb_mapping)
        init_tokens = []
        for i, hold_id in enumerate(start_holds[:2]):
            ht = self.dataset.hold_mapping.get(hold_id)
            if ht is None:
                continue
            lt = self.dataset.limb_mapping.get('RH' if i == 0 else 'LH', i)
            init_tokens.append((ht * n_limbs) + lt + 1)

        if not init_tokens:
            return None

        # ----------------------------------------------------------------
        # Phase 1: autoregressively sample a sequence — no autograd graph.
        # ----------------------------------------------------------------
        sampled_tokens = list(init_tokens)
        was_training   = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                input_seq      = torch.tensor([init_tokens], dtype=torch.long).to(device)
                output, hidden = self.model(input_seq)

                for _ in range(20):
                    logits    = output[0, -1, :].clone()
                    logits[0] = float('-inf')
                    probs     = F.softmax(logits, dim=-1)

                    # Guard against degenerate distributions on MPS.
                    if not probs.isfinite().all() or probs.sum() <= 0:
                        break

                    next_token = torch.multinomial(probs, num_samples=1).item()
                    sampled_tokens.append(next_token)

                    if next_token == 0:
                        break

                    t          = next_token - 1
                    hold_token = t // n_limbs
                    hold_id    = self.dataset.reverse_hold_mapping.get(hold_token)
                    if hold_map.get(hold_id, {}).get('role_id') == 14:
                        break

                    input_seq      = torch.tensor([[next_token]], dtype=torch.long).to(device)
                    output, hidden = self.model(input_seq, hidden)
        finally:
            if was_training:
                self.model.train()

        if len(sampled_tokens) <= len(init_tokens):
            return None  # nothing was sampled

        # Flush MPS queue and release Phase 1 GPU tensors before Phase 2.
        if device.type == 'mps' and hasattr(torch, 'mps'):
            torch.mps.synchronize()
        del output, hidden  # allow MPS to reuse memory

        # ----------------------------------------------------------------
        # Compute reward.
        # ----------------------------------------------------------------
        decoded   = self.dataset.decode_sequence(sampled_tokens)
        reward    = self._compute_sequence_reward(decoded, climb_id)
        advantage = reward - 0.5  # fixed baseline: midpoint of [0, 1]

        if advantage == 0.0:
            return None  # no gradient signal; skip

        # ----------------------------------------------------------------
        # Phase 2: recompute log-probs in ONE vectorised teacher-forced
        # forward pass (no per-token Python indexing — MPS-safe).
        #
        # sampled_tokens layout: [*init_tokens, tok_1, tok_2, ..., tok_K]
        # x (input)            : sampled_tokens[:-1]  → length L-1
        # output_all           : (1, L-1, vocab_size)  with grad
        # Target slice         : output_all[0, n_seed-1 : n_seed-1+K, :]
        #   predicts           : sampled_tokens[n_seed : n_seed+K]
        # ----------------------------------------------------------------
        n_seed            = len(init_tokens)
        # Exclude trailing PAD (0) token if sampling stopped on it.
        sampled_after_seed = [t for t in sampled_tokens[n_seed:] if t != 0]
        if not sampled_after_seed:
            return None

        K         = len(sampled_after_seed)
        x         = torch.tensor([sampled_tokens[:-1]], dtype=torch.long).to(device)
        output_all, _ = self.model(x)          # (1, L-1, vocab_size) — grad

        start_pos  = n_seed - 1
        end_pos    = start_pos + K
        if end_pos > output_all.shape[1]:
            end_pos = output_all.shape[1]
            sampled_after_seed = sampled_after_seed[:end_pos - start_pos]
            if not sampled_after_seed:
                return None

        # (K, vocab_size) slice — gradient flows here.
        logits_slice = output_all[0, start_pos:end_pos, :]
        # Targets as a 1-D LongTensor on the same device.
        tgt = torch.tensor(sampled_after_seed, dtype=torch.long, device=device)

        # F.cross_entropy = -log_softmax[tgt], vectorised, MPS-safe, no
        # per-element Python indexing that triggers MPS assertions.
        neg_lp = F.cross_entropy(logits_slice, tgt, reduction='none')  # (K,)
        return neg_lp.mean() * advantage  # REINFORCE loss (positive → minimise)

    def _reinforce_update(self, device, optimizer, reinforce_weight):
        """Complete REINFORCE update step.

        When *device* is MPS the gradient computation is done on CPU to avoid
        MPS C-level assertion failures (uncatchable by Python try/except).
        CPU gradients are collected, transferred to MPS, and the main MPS
        optimizer performs the parameter update — preserving Adam state.
        """
        needs_cpu = (device.type == 'mps')
        work_dev  = torch.device('cpu') if needs_cpu else device

        try:
            if needs_cpu:
                self.model.to(work_dev)   # move params + buffers to CPU

            rl_loss = self._compute_reinforce_loss(work_dev)
            if rl_loss is None:
                return

            optimizer.zero_grad()
            (reinforce_weight * rl_loss).backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            if needs_cpu:
                # Snapshot CPU gradients, restore model to MPS, apply grads.
                cpu_grads = {
                    n: p.grad.clone() if p.grad is not None else None
                    for n, p in self.model.named_parameters()
                }
                self.model.to(device)   # back to MPS
                optimizer.zero_grad()   # clear stale grad references
                for n, p in self.model.named_parameters():
                    g = cpu_grads.get(n)
                    if g is not None:
                        p.grad = g.to(device)
                # Second clip pass now that grads are on MPS.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

            optimizer.step()

        except Exception:
            pass  # non-fatal — CE training continues
        finally:
            # Always ensure model is on the training device and in train mode.
            if next(self.model.parameters()).device != device:
                self.model.to(device)
            self.model.train()

    def _compute_sequence_reward(self, decoded, climb_id):
        """Scalar reward in [0, 1] measuring generated sequence quality.

        Components (weights sum to 1.0):
          R_finish      (0.5): whether the last two moves include a finish hold
          R_alternation (0.3): fraction of consecutive hand pairs that alternate
          R_length      (0.2): reaches at least 6 moves (min viable sequence)
        """
        if not decoded:
            return 0.0

        climb        = self.climb_data.get(climb_id, {})
        finish_holds = {h['hole_id'] for h in climb.get('holds', []) if h.get('role_id') == 14}

        r_finish = 1.0 if finish_holds and any(
            m['hold'] in finish_holds for m in decoded[-2:]
        ) else 0.0

        hands  = [m['limb'] for m in decoded if m.get('limb') in ('RH', 'LH')]
        n_alt  = sum(1 for i in range(1, len(hands)) if hands[i] != hands[i - 1])
        r_alternation = n_alt / (len(hands) - 1) if len(hands) > 1 else 0.0

        r_length = min(1.0, len(decoded) / 6.0)

        return 0.5 * r_finish + 0.3 * r_alternation + 0.2 * r_length

    def visualize_sequence(self, sequence, climb_id=None, save_path=None, viz_type='path'):
        if not sequence:
            print("No sequence to visualize")
            return

        if climb_id and climb_id in self.climb_data:
            holds = self.climb_data[climb_id]["holds"]
        else:
            holds = [
                {
                    "hole_id": h_id,
                    "x": info.get("x", 0),
                    "y": info.get("y", 0),
                    "role_id": info.get("role_id", -1),
                    "name": info.get("name", ""),
                }
                for h_id, info in self.dataset.hold_info.items()
            ]

        if viz_type == 'path':
            plot_climb_sequence(
                holds,
                sequence,
                title="Climbing Sequence Visualization",
                output_path=save_path,
                show=save_path is None,
            )
        elif viz_type == 'cycle':
            plot_sequence_cycle(
                holds,
                sequence,
                title="Climbing Sequence Visualization",
                output_path=None,
                show=True,
            )
        elif viz_type == 'reachability-hand':
            plot_reachability_map(holds, mode='hand', output_path=save_path, title="Hand Reachability", show=save_path is None)
        elif viz_type == 'reachability-foot':
            plot_reachability_map(holds, mode='foot', output_path=save_path, title="Foot Reachability", show=save_path is None)
        elif viz_type == 'hold-density':
            plot_hold_density(holds, output_path=save_path, title="Hold Density", show=save_path is None)
        
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