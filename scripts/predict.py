import sys
import logging

from darknetflow_poc.utils.config_util import get_user_conf

def handle_error(e):
    """
    Error handling during the prediction dataflow
    """
    logging.error(str(e) + '\n...Exiting program...')
    sys.exit(0)


def main():
    conf = get_user_conf()
    handle_error(conf)


if __name__ == '__main__':
    main()
