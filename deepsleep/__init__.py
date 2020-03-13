import os
import logging
import logging.config
import json

from .models import BuildGPUModel
from .model_callbacks import Histories, ResetModel
from .utils import Metrics, ThreadSafe, ResampleDataset
from .preprocess import PreProcessData
from .data_generator import HeartSequenceGenerator, HeartSequenceLoader, InferenceHeartSequenceLoader
from .training import Trainer
from .data_loader import DataLoader
from .prepare_dataset import PrepareDataset, PreparePretrainDataset

# Source : http://www.patricksoftwareblog.com/python-logging-tutorial/

ROOT = os.getcwd()
LOG_DIR = os.path.join(ROOT, "deepsleep/logs")
#print ROOT
#print LOG_DIR

# Check if log file already exists. If Yes, then remove it for fresh writing.
try:
    if os.path.isfile(LOG_DIR + "/debug.log") and os.path.isfile(LOG_DIR + "/debug1.log"):
        os.remove(LOG_DIR + "/debug.log")
        os.remove(LOG_DIR + "/debug1.log")

        with open("log_config.json", 'r') as log_config_file:
            config_dict = json.load(log_config_file)

        logging.config.dictConfig(config_dict)

    elif os.path.isfile(LOG_DIR + "/debug.log") and os.path.isfile(LOG_DIR + "/errors.log"):
        # os.remove(LOG_DIR + "/debug.log")
        # os.remove(LOG_DIR + "/errors.log")
        # Load log configuration file
        with open("log_config.json", 'r') as log_config_file:
            config_dict = json.load(log_config_file)
            handler = config_dict.get('handlers')
            debug_handler = handler.get('debug_handler')
            debug_handler['filename'] = LOG_DIR + '/debug1.log'

        logging.config.dictConfig(config_dict)

    else:
        with open("log_config.json", 'r') as log_config_file:
            config_dict = json.load(log_config_file)

        logging.config.dictConfig(config_dict)

except:
    print "Generating new log file ..."

# Log that the logger was configured
logger = logging.getLogger(__name__)
logger.info('Completed configuring logger()!')
