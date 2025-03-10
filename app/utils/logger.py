#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Logger: Provides logging functionality for the Crisis Sentinel system.
"""

import os
import logging
import datetime
from logging.handlers import RotatingFileHandler


def setup_logger(name: str = None, level: str = None, file_path: str = None, 
                max_size: int = 10 * 1024 * 1024, backup_count: int = 5) -> logging.Logger:
    """
    Set up a logger with console and file handlers.
    
    Args:
        name: Logger name, uses root logger if None
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        file_path: Path to log file
        max_size: Maximum log file size in bytes before rotation
        backup_count: Number of backup files to keep
        
    Returns:
        Configured logger
    """
    # Import here to avoid circular imports
    from app.utils.config_loader import ConfigLoader
    
    # Get configuration
    config = ConfigLoader().get('logging', default={})
    
    # Use parameters or fall back to config
    log_level = level or config.get('level', 'INFO')
    log_file = file_path or config.get('file', 'logs/crisis_sentinel.log')
    log_format = config.get('format', '%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Convert string level to numeric value
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Get logger
    logger = logging.getLogger(name)
    logger.setLevel(numeric_level)
    
    # Remove existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(log_format)
    
    # Create console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(numeric_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if a path is provided
    if log_file:
        try:
            # Ensure log directory exists
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            
            # Create rotating file handler
            file_handler = RotatingFileHandler(
                log_file, 
                maxBytes=max_size, 
                backupCount=backup_count
            )
            file_handler.setLevel(numeric_level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
            
            logger.debug(f"Log file configured at {log_file}")
        except Exception as e:
            logger.error(f"Failed to configure file logging: {str(e)}")
    
    return logger


class LoggerManager:
    """
    Manages loggers for the application to ensure consistent configuration.
    """
    
    _loggers = {}
    _initialized = False
    _default_config = None
    
    @classmethod
    def initialize(cls, config=None):
        """
        Initialize the logger manager with configuration.
        
        Args:
            config: Configuration dictionary
        """
        if cls._initialized:
            return
        
        cls._default_config = config
        cls._initialized = True
    
    @classmethod
    def get_logger(cls, name: str = None) -> logging.Logger:
        """
        Get a logger with the specified name.
        
        Args:
            name: Logger name
            
        Returns:
            Configured logger
        """
        if name in cls._loggers:
            return cls._loggers[name]
        
        logger = setup_logger(name)
        cls._loggers[name] = logger
        return logger 