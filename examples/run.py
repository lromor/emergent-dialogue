import sys
import logging

import torch

from signal.config import ConfigParser


_LOGGER = logging.getLogger(__name__)
_LOGGER.addHandler(logging.NullHandler())


def main(config):
    _LOGGER.info("loaded configuration: \n%s", str(config))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _LOGGER.info("Using %s.", "GPU acceleration" if device.type == 'cuda'
                 else "CPU")

    # Load the dataset

    # Start the training loop


if __name__ == "__main__":
    config_parser = ConfigParser.create()
    config = config_parser.parse_args()

    logging.basicConfig(level=logging.DEBUG)
    main(config)
