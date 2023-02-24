import configparser
import os
import logging

_DEFAULT_CONFIG_FILE = "config/config.ini"
_CONFIG_FILE_ENCODING = 'utf-8-sig'
ROOT_DIR = os.path.dirname(os.path.realpath(__file__))


def get_config(config_file=_DEFAULT_CONFIG_FILE):
    config = configparser.ConfigParser()
    config.read(os.path.join(ROOT_DIR, config_file), encoding=_CONFIG_FILE_ENCODING)
    return config


def set_log_level(loc_config):
    if loc_config["LOGGING"]["LOG_LEVEL"] == "DEBUG":
        log_level = logging.DEBUG
    elif loc_config["LOGGING"]["LOG_LEVEL"] == "INFO":
        log_level = logging.INFO
    elif loc_config["LOGGING"]["LOG_LEVEL"] == "WARNING":
        log_level = logging.WARNING
    elif loc_config["LOGGING"]["LOG_LEVEL"] == "ERROR":
        log_level = logging.ERROR
    else:
        raise ValueError(
            "Unknown log level provided in the configuration file: {}".format(loc_config["LOGGING"]["LOG_LEVEL"]))
    logging.getLogger().setLevel(log_level)


def get_base_path(args, loc_config):
    base_path = loc_config["PATHS"]["BASE_PATH"]
    return base_path