"""
Logging utilities for the baggage scanner project.
"""

import logging
import os
from datetime import datetime
from typing import Optional


def setup_logger(
    name: str,
    log_file: Optional[str] = None,
    level: int = logging.INFO,
    format_string: Optional[str] = None
) -> logging.Logger:
    """
    Set up a logger with both file and console handlers.
    
    Parameters
    ----------
    name : str
        Name of the logger
    log_file : str, optional
        Path to the log file. If None, only console logging is enabled
    level : int
        Logging level (e.g., logging.INFO, logging.DEBUG)
    format_string : str, optional
        Custom format string for log messages
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    # Create logger
    logger = logging.getLogger(name)
    logger.setLevel(level)
    
    # Remove existing handlers to avoid duplicates
    logger.handlers = []
    
    # Default format
    if format_string is None:
        format_string = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    
    formatter = logging.Formatter(format_string)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if log_file is provided)
    if log_file is not None:
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    return logger


def get_experiment_logger(experiment_name: str, output_dir: str = "outputs/logs") -> logging.Logger:
    """
    Create a logger for a specific experiment with timestamped log file.
    
    Parameters
    ----------
    experiment_name : str
        Name of the experiment
    output_dir : str
        Directory to save log files
        
    Returns
    -------
    logging.Logger
        Configured logger instance
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = f"{experiment_name}_{timestamp}.log"
    log_path = os.path.join(output_dir, log_filename)
    
    return setup_logger(
        name=experiment_name,
        log_file=log_path,
        level=logging.INFO
    )


class TqdmLoggingHandler(logging.Handler):
    """
    Custom logging handler that works well with tqdm progress bars.
    """
    
    def __init__(self, level=logging.NOTSET):
        super().__init__(level)
    
    def emit(self, record):
        try:
            msg = self.format(record)
            # Use tqdm.write to avoid conflicts with progress bars
            from tqdm import tqdm
            tqdm.write(msg)
        except Exception:
            self.handleError(record)
