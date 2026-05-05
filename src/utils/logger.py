import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime, timezone, timedelta

# Tạo timezone +7
VN_TZ = timezone(timedelta(hours=7))

class VNFormatter(logging.Formatter):
    def formatTime(self, record, datefmt=None):
        dt = datetime.fromtimestamp(record.created, VN_TZ)
        if datefmt:
            return dt.strftime(datefmt)
        return dt.isoformat()

def setup_logging():
    formatter = VNFormatter(
        "%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        "%Y-%m-%d %H:%M:%S"
    )

    handler = RotatingFileHandler(
        "app.log",
        maxBytes=5 * 1024 * 1024,
        backupCount=2
    )
    handler.setFormatter(formatter)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)

    if not root_logger.handlers:
        root_logger.addHandler(handler)
        root_logger.addHandler(console_handler)
    

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(name)