import logging
from datetime import datetime
from pathlib import Path

# Create logs dir
BASE_DIR = Path(__file__).parent
LOGS_DIR = BASE_DIR / "logs"
LOGS_DIR.mkdir(exist_ok=True)

# Generate log filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_FILE = LOGS_DIR / f"secret_santa_debug_{timestamp}.log"

# Setup Logger
logger = logging.getLogger("SecretSantaDebug")
logger.setLevel(logging.DEBUG)

# File Handler
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setLevel(logging.DEBUG)
file_formatter = logging.Formatter('%(asctime)s - %(message)s')
file_handler.setFormatter(file_formatter)

# Console Handler
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO) # Keep console clean, file verbose
console_formatter = logging.Formatter('%(message)s')
console_handler.setFormatter(console_formatter)

# Avoid adding handlers multiple times if module is reloaded
if not logger.handlers:
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def log_section(title):
    msg = f"\n{'='*20} {title} {'='*20}"
    logger.info(msg)

def log_data(label, data):
    logger.info(f"\n[{label}]:\n{data}")


