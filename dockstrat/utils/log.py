import os
import logging
from typing import Dict, Any, Optional

def setup_base_logging(config: Dict[str, Any]) -> str:
    """
    Set up base logging based on the configuration.
    
    Args:
        config (dict): The resolved configuration
        
    Returns:
        str: Path to the log directory
    """
    # Determine log directory path from config
    if 'output_dir' in config:
        log_dir = config['log_dir']
    else:
        # Fallback to current directory
        log_dir = os.getcwd()
    
    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set up the default log file
    default_log_file = os.path.join(log_dir, 'gnina_timing.log')
    
    # Set up basic logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=[
            logging.FileHandler(default_log_file),
            logging.StreamHandler()  # Send log output to the console as well
        ]
    )
    
    logging.info(f"Base logging initialized to: {default_log_file}")
    
    return log_dir

def get_custom_logger(logger_name: str, config: Dict[str, Any], log_filename: Optional[str] = None) -> logging.Logger:
    """
    Returns a logger configured with a FileHandler that writes to a path determined by config.
    Any existing handlers attached to this logger will be removed.
    
    Args:
        logger_name (str): Name of the logger
        config (dict): The resolved configuration
        log_filename (str, optional): Custom filename for the log. If None, uses logger_name + '.log'
        
    Returns:
        logging.Logger: Configured logger instance
    """
    # Determine log directory path from config
    if 'log_dir' in config:
        log_dir = config['log_dir']
    else:
        # Fallback to current directory
        log_dir = os.getcwd()
    
    # Create the directory if it doesn't exist
    os.makedirs(log_dir, exist_ok=True)
    
    # Set the log filename
    if log_filename is None:
        log_filename = f"{logger_name}.log"
    
    # Create the full log path
    log_path = os.path.join(log_dir, log_filename)
    
    # Get or create the logger
    logger = logging.getLogger(logger_name)
    
    # Remove any existing handlers attached to this logger
    if logger.hasHandlers():
        logger.handlers.clear()
    
    logger.setLevel(logging.INFO)
    
    # Create file handler with the custom filename
    file_handler = logging.FileHandler(log_path)
    file_handler.setLevel(logging.INFO)
    
    # Define a formatter and set it for the file handler
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    file_handler.setFormatter(formatter)
    
    logger.addHandler(file_handler)
    
    # Optionally, also log to the console by adding a StreamHandler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Log where this specific logger is writing to
    logger.info(f"Logger '{logger_name}' initialized to write to: {log_path}")
    
    return logger