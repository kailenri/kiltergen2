DB_PATH = "DB-path-here" #enter your local path to the DB
BEAM_WIDTH = 100 #beam search width 
X_SPACING = 18.66666666666 #spacing of holds from eachother on X axis 8 inches = 18.666666 X
Y_SPACING = 19.83333333333 #spacing on Y axis 8.5 inches 
MAX_HAND_REACH = 3.5 * X_SPACING #estimated reach of hands
MAX_FOOT_REACH = 3.2 * X_SPACING #estimated reach of feet/legs
MAX_CLIMBS_TO_PROCESS = 271502 #every climb in the DB can adjust for small batches 
MAX_HOLDS_PER_CLIMB = 50
MIN_HOLDS_PER_CLIMB = 4
JSON_PATH = "json-path-here" #for training 
LSTM_PATH = "lstm.pth" #enter full LSTM path here