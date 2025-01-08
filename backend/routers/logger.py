import os
import logging
import coloredlogs

LOG_DIR = "logs"
os.makedirs(LOG_DIR, exist_ok=True)

# Create a logger instance
logger = logging.getLogger("LY-PROJECT")

# Set the logging level
logger.setLevel(logging.INFO)

# Create handlers
console_handler = logging.StreamHandler()

# Create formatter
log_format = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

console_handler.setFormatter(log_format)

# Add handlers to the logger
logger.addHandler(console_handler)

# Set up coloredlogs
coloredlogs.install(
    level='INFO',
    logger=logger,
    fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    level_styles={
        'info': {'color': 'green'},
        'warning': {'color': 'yellow'},
        'error': {'color': 'red'},
        'critical': {'color': 'magenta'}
    },
    field_styles={
        'asctime': {'color': 'blue'},
        'name': {'color': 'cyan'},
        'levelname': {'bold': True, 'color': 'white'}
    }
)
# Prevent logging from being propagated to the root logger
logger.propagate = False