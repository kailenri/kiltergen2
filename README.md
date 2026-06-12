KilterGen2
==========

Generate, score, and visualize climbing sequences from a Kilter Board database, and train an LSTM model to propose new move sequences.

**Features**: Beam search with kinematics-driven scoring (footcuts, flags, dynamic moves) → LSTM training with difficulty conditioning and policy gradients → rich visualizations with beta panels and calibrated board overlays.

Overview
--------
- **Data Pipeline**: Parse Kilter Board SQLite DB → validate holds with kinematics → filter by count/grades
- **Sequence Generation**: Beam search with reachability, hand/foot role constraints, and move quality scoring
- **LSTM Training**: Sequence-to-sequence with focal loss, REINFORCE, difficulty conditioning, early stopping, checkpointing
- **Visualizations**: Board overlays with calibrated hold positions, 5 viz types (path, cycle, reachability maps, hold density), animated GIF support
- **Route Generation**: Create new climbing sequences from wall layouts using trained models
- **Production Models**: 8 trained model versions (v3–v8) with checkpoint history

Project Structure
-----------------
**Core Modules:**
- `main.py`: Batch sequence generation over 271k+ DB climbs with hold filtering
- `db_utils.py`: Parse frames column → structured holds; add `holds_in` column to DB
- `sequence_generator.py`: Beam search (width=100) with kinematics-driven scoring and move validation
- `kinematics.py`: Shared geometry library (2-bone IK, convex hull support polygon, ellipse reach, body joint tracking)
- `lstm.py`: `ClimbDataset`, `ClimbLSTM` (2-layer + difficulty conditioning), `ClimbGenerator` (train/generate/load)
- `lstm_main.py`: CLI interface with 3 modes (train/generate/visualize)
- `new_climbs.py`: Route generator integrating LSTM for sequence proposals with wall layouts
- `viz.py`: Visualization engine + CLI (board overlays, beta panel, 5 viz types)
- `config.py`: 40+ tuning parameters (reach, limb lengths, foot-cut/flag rules, beam width, hold filters)

**Utility Scripts (Advanced):**
- `compare_models.py`: N-way model comparison with foot metrics and stats
- `probe_foot_prob.py`: Teacher-force analysis of foot probability in LSTM
- `build_hold_orientations.py`: Extract hold quality/direction from board image
- `calibrate_board_image.py`: Compute affine board→pixel calibration with debug overlay
- `export_training.py`: Batch sequence export with cleaning and validation
- `build_vocab.py`: Generate hold-ID→token mappings from training data
- `analyze_data_distribution.py`: Dataset statistics and distribution analysis
- `spotcheck_beam_feet.py`: Validate foot move generation in beam search

**Test Suite:**
- `tests/`: 8 test modules covering kinematics, rules, hold orientation, visualization, and foot decoding

Requirements
------------
- Python 3.10+
- A local SQLite DB with Kilter Board tables
- PyTorch (install CPU or GPU variant per your setup)

Install dependencies:
	pip install -r requirements.txt

Suggested Python deps (in requirements.txt):
- numpy
- matplotlib
- torch
- tqdm
- scikit-learn
- jsonschema

Quickstart
----------
1) Set paths in config.py
	 - DB_PATH: SQLite DB file
	 - JSON_PATH: JSON output path for training
	 - LSTM_PATH: Path to trained model

2) Populate holds in the DB (optional but recommended)
	 python db_utils.py

3) Generate sequences from DB climbs
	 python main.py

4) Visualize a specific climb from a results JSON
	 python viz.py --data climb_results_*.json --id <climb_id>

Data Pipeline
-------------
1) db_utils.py parses the frames column into structured holds
2) main.py pulls climbs, validates holds, and runs sequence generation
3) Results are stored as JSON with metadata

Visualization
-------------
The visualization module creates richly detailed climb overlays. All visuals are saved to PNG, with optional animated GIF support.

**Visual Design**: Board overlay with calibrated hold positions on left; right-side beta panel shows move-by-move breakdown with distances (reachability) and stats (move count, height gain, average/max distances).

**5 Visualization Types:**

1. **path** (default): Board + move sequence on beta panel with text labels
	python viz.py --data climb_results_*.json --id <climb_id> --type path

2. **cycle**: Animated sequence showing limb positions on board (stick-figure legacy mode)
	python viz.py --data climb_results_*.json --id <climb_id> --type cycle

3. **reachability-hand**: Reachability heatmap for hand positions from each move
	python viz.py --data climb_results_*.json --id <climb_id> --type reachability-hand

4. **reachability-foot**: Reachability heatmap for foot positions
	python viz.py --data climb_results_*.json --id <climb_id> --type reachability-foot

5. **hold-density**: Heatmap of hold usage frequency across the sequence
	python viz.py --data climb_results_*.json --id <climb_id> --type hold-density

**Generating all visualizations for a climb:**
	python viz.py --data climb_results_*.json --id <climb_id> --type path
	python viz.py --data climb_results_*.json --id <climb_id> --type cycle
	python viz.py --data climb_results_*.json --id <climb_id> --type reachability-hand
	python viz.py --data climb_results_*.json --id <climb_id> --type reachability-foot
	python viz.py --data climb_results_*.json --id <climb_id> --type hold-density

**Route Generator Visualization** (using trained LSTM model):
	python new_climbs.py --model <model.pth> --data <data.json> --visualize --viz-type path
	python new_climbs.py --model <model.pth> --data <data.json> --visualize --viz-type reachability-hand

LSTM Training and Sampling
--------------------------

**Basic Training:**
	python lstm_main.py train --data <data.json> --epochs 30 --batch-size 32 --learning-rate 0.001 --output lstm_model.pth

**Generate Sequences from Model:**
	python lstm_main.py generate --model lstm_model.pth --data <data.json> --count 5 --temperature 0.8

**Visualize Generated Sequences:**
	python lstm_main.py visualize --model lstm_model.pth --data <data.json> --count 3 --output-dir visualizations

**Advanced Training Options:**

Focal Loss (for class imbalance in move tokens):
	python lstm_main.py train --data <data.json> --epochs 30 --focal-loss

REINFORCE Policy Gradient (optimize for move quality):
	python lstm_main.py train --data <data.json> --epochs 30 --reinforce-weight 0.01

Early Stopping (with patience and min improvement delta):
	python lstm_main.py train --data <data.json> --epochs 100 --early-stopping-patience 5 --min-delta 0.001

Model Checkpointing (save best + periodic):
	python lstm_main.py train --data <data.json> --epochs 30 --checkpoint-dir ./checkpoints --checkpoint-freq 5

TensorBoard Logging (for loss curves and metrics):
	python lstm_main.py train --data <data.json> --epochs 30 --tb-logdir ./logs/tb_run_1
	tensorboard --logdir ./logs

**Combined Advanced Example:**
	python lstm_main.py train --data <data.json> --epochs 100 --batch-size 64 --learning-rate 0.0005 \
		--focal-loss --reinforce-weight 0.02 --early-stopping-patience 10 --checkpoint-dir ./checkpoints \
		--checkpoint-freq 5 --tb-logdir ./logs --output best_model.pth

**Advanced Generation with Custom Temperature & Length:**
	python lstm_main.py generate --model lstm_model.pth --data <data.json> --count 10 \
		--temperature 0.9 --max-length 25 --output-dir generated_climbs --export-json sequences.json

**Training with Custom Vocabulary:**
	python lstm_main.py train --data <data.json> --vocab custom_vocab.json --output lstm_model.pth

Configuration Notes
-------------------

**Setup Required:**
All of these must be configured in `config.py` before running:
- `DB_PATH`: Path to Kilter Board SQLite database file
- `JSON_PATH`: Output directory for training JSON files
- `LSTM_PATH`: Path to save/load trained model files

**Reachability & Geometry:**
- `MAX_HAND_REACH`: Maximum arm reach for hand placement (inches, default ~36)
- `MAX_FOOT_REACH`: Maximum leg reach for foot placement (inches, default ~40)
- `X_SPACING` / `Y_SPACING`: Board hole spacing in pixels
- `UPPER_ARM_LEN`, `FOREARM_LEN`: Arm segment lengths (affects IK)
- `THIGH_LEN`, `SHIN_LEN`: Leg segment lengths
- `TORSO_LEN`, `HEAD_OFFSET`: Torso and head geometry

**Move Validation:**
- `BEAM_WIDTH`: Beam search breadth (default=100; lower for speed, higher for quality)
- `MIN_HOLDS_PER_CLIMB` / `MAX_HOLDS_PER_CLIMB`: Filter climbs by hold count (default 4–50)
- `MAX_CLIMBS_TO_PROCESS`: Limit batch processing (default=271502 for full DB)

**Advanced Move Types:**
- `FOOT_CUT_ENABLED`: Allow foot cuts (dynamic reachability boost)
- `DYNAMIC_REACH_FACTOR`: Multiplier for reach when foot-cutting (default=1.5x)
- `FOOT_REPLANT_MAX_STEPS`: Steps after foot-cut before must replant (default=2)
- `FLAG_ENABLED`: Allow flags with one hand/foot
- `FLAG_LEG_FACTOR`: Reach multiplier when flagging (default=1.3x)

**Hand/Foot Roles:**
- `HAND_ROLE_TUNING`: Role scoring weights (crimpy, jug, sloper, etc.)
- `FOOT_MOVE_BASE_SCORE`: Score boost for sequences with foot moves (Phase A: 0.5→2.5)
- `LAST_2_HANDS_FOOT_BONUS`: Conditional 1.5x boost if last 2 moves were hands

**Data Filtering:**
- `LAYOUT_ID`: Kilter Board layout to use (default=1 for standard)
- `MIN_GRADE_DIFFICULTY` / `MAX_GRADE_DIFFICULTY`: Climb difficulty range

Utility Scripts
---------------

**Model Comparison & Analysis:**
- `compare_models.py`: Compare N trained models side-by-side with foot metrics, move counts, and reachability stats
- `probe_foot_prob.py`: Teacher-force analysis of foot probability in LSTM (debug foot move generation)

**Board Calibration & Hold Analysis:**
- `calibrate_board_image.py`: Compute affine transformation from board physical space → pixel coordinates; outputs debug overlay
- `build_hold_orientations.py`: Extract hold metadata (quality, direction, confidence) from board image
- `analyze_data_distribution.py`: Dataset statistics, move frequency, hold usage heatmaps

**Data Preparation:**
- `build_vocab.py`: Generate hold-ID → token mappings from training sequences
- `export_training.py`: Batch export and clean training sequences from DB
- `spotcheck_beam_feet.py`: Validation script to check foot move generation in beam search

**Quick Tests:**
- `run_seq_cycle.py`: Single-climb demo showing sequence generation + visualization

Advanced Features
-----------------

**Kinematics Engine:**
- 2-bone inverse kinematics (arm & leg) with joint angle constraints
- Convex hull support polygon for balance validation
- Elliptical reachability modeling (accounts for reach variation by angle)
- Dynamic moves: foot-cuts (cross reachability boost), flags (single-limb support)
- Limb role constraints (hands prefer certain hold types; feet avoid crimps)

**LSTM Training Enhancements:**
- **Focal Loss**: Addresses class imbalance in move tokens (common moves vs rare transitions)
- **REINFORCE Policy Gradient**: Optional reward shaping to optimize move quality & diversity
- **Difficulty Conditioning**: 5 difficulty buckets from Kilter grades; model learns per-grade patterns
- **Early Stopping**: Patience + min improvement delta to prevent overfitting
- **Checkpointing**: Save best model + periodic snapshots for reproducibility
- **TensorBoard Integration**: Real-time loss/accuracy monitoring

**Data Preparation:**
- Vocabulary with role (hand/foot) and limb embeddings
- Sequence tokenization with balanced class weighting
- Support for multiple dataset versions (full vs subset) with different vocab
- Hold orientation quality scoring from board calibration

Testing
-------
Run the full test suite:
	pytest tests/

Specific test categories:
	pytest tests/test_kinematics.py      # IK, body geometry, reachability
	pytest tests/test_generator_rules.py # Beam search move validation
	pytest tests/test_hold_orientation.py # Hold metadata extraction
	pytest tests/test_foot_cut.py        # Foot-cut move generation
	pytest tests/test_flagging.py        # Flag move validation
	pytest tests/test_viz_cycle.py       # Visualization rendering
	pytest tests/test_decode_foot_emission.py # LSTM foot token decoding

Outputs
-------

**From `main.py` (Batch Sequence Generation):**
- `climb_results_<timestamp>.json`: Sequences for all DB climbs (filtered by hold count)
- Includes move sequence, reachability stats, hold list, success/failure flags

**From `lstm_main.py train`:**
- `<output>.pth`: Trained LSTM model weights
- `training_history.png`: Loss/accuracy curves (if using matplotlib backend)
- `./logs/tb_run_*/`: TensorBoard event files (if --tb-logdir specified)
- `./checkpoints/`: Per-epoch checkpoint files (if --checkpoint-dir specified)

**From `lstm_main.py generate`:**
- `./generated/`: Individual PNG visualizations per sequence (if --output-dir specified)
- `sequences.json`: Exported sequences in JSON format (if --export-json specified)

**From `lstm_main.py visualize`:**
- `./visualizations/`: PNG files for each generated climb
- Includes board overlays with beta panel (path type) or stick figures (cycle type)

**From `viz.py`:**
- `./climb_visualizations/`: PNG/GIF files for DB climbs
- Named by climb_id and viz type

**From `new_climbs.py`:**
- `./visualizations/`: Generated route visualizations
- `generated_climbs.json`: New climbing sequences (if saved)

**From utility scripts:**
- `calibrate_board_image.py`: Affine calibration matrix + debug overlay PNG
- `build_hold_orientations.py`: `hold_orientations.json` with quality/direction metadata
- `compare_models.py`: Comparison stats to console or JSON
- `export_training.py`: Cleaned training sequences JSON with train/val splits

Data & Models
-------------

**Datasets** (in `data/`):
- `climbs_all.json`: Full Kilter Board climb database
- `climbs_subset_5k*.json`: 5k-climb subsets for faster iteration
- `training_sequences_*.json`: 8 versions (v1–v8) with beam-generated sequences
- `vocab_*.json`: Hold-to-token mappings for each dataset version

**Pretrained Models** (in `models/`):
- `subset_v3_model.pth` – `subset_v8_model.pth`: Production model progression
- `full_train_model.pth`, `full_rebuild_model.pth`: Full-dataset models
- `checkpoints_v*`: Per-epoch checkpoints for training inspection

**Latest Model** (v8):
- 2,535 sequences
- 100% contain ≥1 foot move
- Median 2.0 feet/sequence
- Mean flag quality: 0.833

Workflow Examples
-----------------

**Scenario 1: Generate sequences from your DB**
1. Configure `DB_PATH` in config.py
2. Run: `python main.py`
3. Output: `climb_results_<timestamp>.json`
4. Visualize: `python viz.py --data climb_results_*.json --id 12345 --type path`

**Scenario 2: Fine-tune existing model on new data**
1. Export new sequences: `python export_training.py --climbs new_climbs.json --output new_training.json`
2. Build vocab: `python build_vocab.py --data new_training.json --output new_vocab.json`
3. Train: `python lstm_main.py train --data new_training.json --vocab new_vocab.json --epochs 10 --output finetuned.pth`
4. Generate: `python lstm_main.py generate --model finetuned.pth --data new_training.json --count 20`

**Scenario 3: Full analysis pipeline**
1. Calibrate board: `python calibrate_board_image.py --image board.jpg --output calibration.json`
2. Extract hold orientations: `python build_hold_orientations.py --calibration calibration.json`
3. Generate sequences: `python main.py`
4. Train model: `python lstm_main.py train --data climb_results_*.json --focal-loss --reinforce-weight 0.01 --epochs 50 --tb-logdir ./logs`
5. Compare models: `python compare_models.py --models model1.pth model2.pth --data climb_results_*.json`
6. Visualize results: `python lstm_main.py visualize --model model2.pth --data climb_results_*.json --count 10 --output-dir final_viz`
