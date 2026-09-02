import sys
import os

# Add root directory to sys.path so it can find main.py and local modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from main import app
