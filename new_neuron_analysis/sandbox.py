import sys
import traceback
import logging
try:
    asdf
except Exception as e:
    logging.error("Error:======", exc_info=e)