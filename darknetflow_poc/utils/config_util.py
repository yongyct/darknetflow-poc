import os
import sys
import json
import argparse
import logging


def get_user_conf():

    # Logging configs
    logging.basicConfig(level=getattr(logging, 'INFO', logging.INFO))

    # Program args
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--filename', nargs='?', required=True, help='filename of json config file')
    program_args = parser.parse_args()

    logging.info('Json config provided: {}'.format(program_args.filename))

    return get_json_conf(program_args.filename)


def get_json_conf(filename):
    """
    Parses json information in the filename, and return it as a dictionary
    :param filename:
    :return:
    """
    with open(filename) as json_file:
        conf = json.load(json_file)
    return conf
