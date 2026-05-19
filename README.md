KilterGen2
==========

Generate, score, and visualize climbing sequences from a Kilter Board database, and train an LSTM model to propose new move sequences.

Overview
--------
- Build hold datasets from the SQLite DB
- Generate sequences with a beam search evaluator
- Train and sample from an LSTM model
- Create visualizations for routes, reachability, and hold density

Project Structure
-----------------
- main.py: Batch sequence generation over DB climbs
- db_utils.py: Parse frames and populate holds
- sequence_generator.py: Beam search + sequence scoring
- lstm.py: Dataset, LSTM model, training, and sampling
- lstm_main.py: CLI for train/generate/visualize
- new_climbs.py: Route generator using wall layouts
- viz.py: Shared visualization helpers and CLI
- config.py: Paths and tuning constants

Requirements
------------
- Python 3.10+
- A local SQLite DB with Kilter Board tables

Suggested Python deps:
- numpy
- matplotlib
- torch
- tqdm
- scikit-learn

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
The visualization module supports multiple output types. All visuals are saved to PNG.

Path visualization (limb sequence + stats):
	python viz.py --data climb_results_*.json --id <climb_id> --type path

Reachability maps:
	python viz.py --data climb_results_*.json --id <climb_id> --type reachability-hand
	python viz.py --data climb_results_*.json --id <climb_id> --type reachability-foot

Hold density heatmap:
	python viz.py --data climb_results_*.json --id <climb_id> --type hold-density

The new_climbs.py generator also supports these types:
	python new_climbs.py --model <model.pth> --data <data.json> --visualize --viz-type path
	python new_climbs.py --model <model.pth> --data <data.json> --visualize --viz-type reachability-hand

LSTM Training and Sampling
--------------------------
Train:
	python lstm_main.py train --data <data.json> --epochs 30 --batch-size 32 --learning-rate 0.001 --output lstm_model.pth

Generate:
	python lstm_main.py generate --model lstm_model.pth --data <data.json> --count 5 --temperature 0.8

Visualize generated sequences:
	python lstm_main.py visualize --model lstm_model.pth --data <data.json> --count 3 --output-dir visualizations

Configuration Notes
-------------------
- MAX_HAND_REACH and MAX_FOOT_REACH drive reachability calculations
- BEAM_WIDTH affects the sequence search breadth
- MIN_HOLDS_PER_CLIMB and MAX_HOLDS_PER_CLIMB filter the DB query

Outputs
-------
- climb_results_<timestamp>.json: Sequence generation output from main.py
- training_history.png: Loss curves from LSTM training
- climb_visualizations/: Visual outputs from viz.py
- visualizations/: Visual outputs from new_climbs.py and lstm_main.py
