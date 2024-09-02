import os
import logging
from datetime import datetime
from utils.logger import set_logger

class Trainer(object):
    def __init__(self, args):
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(args.save_dir, sub_dir)
        self.log_dir = os.path.join(args.log_dir, sub_dir)

        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        set_logger(os.path.join(self.log_dir, 'train.log'))

        for k,v in args.__dict__.items():
            logging.info("{}: {}".format(k, v))
        self.args = args

    def setup(self):
        """initial the datasets, model, loss and optimizer"""
        pass

    def train(self):
        """training one epoch"""
        pass