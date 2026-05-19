import sys
import os

# Ensure repository root is on sys.path so top-level modules import correctly
ROOT = os.path.dirname(os.path.dirname(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)
