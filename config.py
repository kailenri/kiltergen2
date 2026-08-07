import os as _os
# Default to the DB that ships at the repo root, resolved relative to this
# file so it works regardless of the caller's working directory. Override by
# setting the env var or editing this path if your DB lives elsewhere.
DB_PATH = _os.environ.get(
    "KILTERGEN_DB_PATH",
    _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), "db.sqlite3"),
)
BEAM_WIDTH = 100 #beam search width 
X_SPACING = 18.66666666666 #spacing of holds from eachother on X axis 8 inches = 18.666666 X
Y_SPACING = 19.83333333333 #spacing on Y axis 8.5 inches 
MAX_HAND_REACH = 3.5 * X_SPACING #estimated reach of hands
MAX_FOOT_REACH = 3.2 * X_SPACING #estimated reach of feet/legs

# Dynamic-move / foot-cut rule constants.
# When a hand move exceeds DYNAMIC_REACH_FACTOR * MAX_HAND_REACH OR the post-move
# COM falls outside the support polygon, the move is "dynamic" and one foot is
# treated as cut. The cut foot must be re-planted within FOOT_REPLANT_MAX_STEPS
# moves before any further hand move is allowed.
FOOT_CUT_ENABLED = True
DYNAMIC_REACH_FACTOR = 0.85
DYNAMIC_DNORM_TRIGGER = 1.0    # ellipse dnorm above this also counts as dynamic
FOOT_REPLANT_MAX_STEPS = 2

# Phantom-flag rule. A "flag" extends the opposite foot into space (no hold)
# to balance a cross/side-reach. The foot has no hold but provides stance
# (it's a valid support point, unlike a cut). FLAG_LEG_FACTOR scales the
# flag reach as a fraction of total leg length.
FLAG_ENABLED = True
FLAG_LEG_FACTOR = 0.80

# Articulated body bone lengths (board units). Calibrated for a "median"
# climber: X_SPACING ~= 20 cm, so arm ~= 70 cm = 3.5*X_SPACING split as
# upper/forearm ~ 0.95/1.0, leg ~= 95 cm = 4.7*X_SPACING split as thigh/shin
# ~ 1.2/1.15. TORSO_LEN matches the existing shoulder offset of 2*X_SPACING.
UPPER_ARM_LEN = 1.7 * X_SPACING
FOREARM_LEN   = 1.8 * X_SPACING
THIGH_LEN     = 2.4 * X_SPACING
SHIN_LEN      = 2.3 * X_SPACING
TORSO_LEN     = 2.0 * X_SPACING
HEAD_OFFSET   = 1.0 * X_SPACING

MAX_CLIMBS_TO_PROCESS = 271502 #every climb in the DB can adjust for small batches 
MAX_HOLDS_PER_CLIMB = 50
MIN_HOLDS_PER_CLIMB = 4
JSON_PATH = "json-path-here" #for training 
LSTM_PATH = "lstm.pth" #enter full LSTM path here