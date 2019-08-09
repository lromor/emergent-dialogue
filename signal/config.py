import argparse
import configargparse


class Config(argparse.Namespace):

    def __str__(self):
        """Return a pretty version of the parsed arguments."""
        s = '\nConfig\n'
        s += '---------------------------------------\n'
        for key, value in vars(self).items():
            s += f'{key}: {value}\n'
        s += '---------------------------------------\n'
        return s


class ConfigParser(configargparse.ArgParser):

    def parse_args(self, *args, **kwargs):
        return super().parse_args(namespace=Config(), **kwargs)

    @classmethod
    def create(cls):
        """Returns the configuration object that specifies the
        supported options for the training.
        """
        p = cls()
        p.add('-c', '--config', required=False, is_config_file=True,
              help='Config file to be used.')
        p.add('--max-epochs', type=int, default=10000)
        p.add('--embedding-size', type=int, metavar="I",
              default=64, help="Size of the input embeddings.")
        p.add('--hidden-size', type=int, metavar="H",
              default=64, help="Size of the hidden layers.")
        p.add('--seed', type=int, default=42,
              help="Base random seed value.")
        p.add("--batch-size", type=int, default=1024,
              help="Input batch size for training.")
        p.add("--max-message-length", type=int,
              default=10, help="Max sentence length allowed "
              "for communication between the agents.")
        p.add("--n-distractors", type=int, default=3,
              help="Number of distractors.")
        p.add("--vocab-size", type=int, default=25,
              help="Size of vocabulary")
        p.add("--lr", type=float, default=1e-3,
              help="Adam learning rate.")
        p.add("--freeze-sender", help="Freeze sender weights (do not train) ",
              action="store_true")
        p.add("--freeze-receiver", help="Freeze receiver weights (do not train) ",
              action="store_true")
        p.add("--tau", type=float, default=1.2,
              help="Temperature parameter for softmax distributions")

        return p
