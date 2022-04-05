import logging
from argparse import ArgumentParser
from enum import Enum, EnumMeta

import torch
import os
import socket
import ujson as json
import time
import random
import numpy as np

logger = logging.getLogger(__name__)


def dist_initialize():
    """
    initializes torch distributed
    :return: local_rank, global_rank, world_size
    """
    if "RANK" not in os.environ:
        local_rank = -1
        global_rank = 0
        world_size = 1
    else:
        if torch.cuda.device_count() == 0:
            err = f'No CUDA on {socket.gethostname()}'
            logger.error(err)
            raise ValueError(err)
        global_rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        env_master_addr = os.environ['MASTER_ADDR']
        # Initializes the distributed backend which will take care of sychronizing nodes/GPUs
        if env_master_addr.startswith('file://'):
            torch.distributed.init_process_group(backend='nccl',
                                                 init_method=env_master_addr,
                                                 world_size=world_size,
                                                 rank=global_rank)
            logger.info("init-method file: {}".format(env_master_addr))
            local_rank = int(os.environ['LOCAL_RANK'])
        else:
            torch.distributed.init_process_group(backend='nccl')
            logger.info("init-method master_addr: {} master_port: {}".format(
                env_master_addr, os.environ['MASTER_PORT']))
            local_rank = int(global_rank % torch.cuda.device_count())
    cuda_devices = os.environ['CUDA_VISIBLE_DEVICES'] if 'CUDA_VISIBLE_DEVICES' in os.environ else 'NOT SET'
    logger.info(f"world_rank {global_rank} cuda_is_available {torch.cuda.is_available()} "
                f"cuda_device_cnt {torch.cuda.device_count()} on {socket.gethostname()},"
                f" CUDA_VISIBLE_DEVICES = {cuda_devices}")
    return local_rank, global_rank, world_size


class HypersBase:
    """
    This should be the base hyperparameters class, others should extend this.
    """
    def __init__(self):
        self.local_rank, self.global_rank, self.world_size = dist_initialize()
        # required parameters initialized to the datatype
        self.model_type = ''
        self.model_name_or_path = ''
        self.resume_from = ''  # to resume training from a checkpoint
        self.config_name = ''
        self.tokenizer_name = ''
        self.cache_dir = 'data/cache'
        self.do_lower_case = False
        self.gradient_accumulation_steps = 1
        self.learning_rate = 5e-5
        self.weight_decay = 0.0  # previous default was 0.01
        self.adam_epsilon = 1e-8
        self.max_grad_norm = 1.0
        self.warmup_instances = 0  # previous default was 0.1 of total
        self.num_train_epochs = 3
        self.no_cuda = False
        self.n_gpu = 1
        self.seed = 42
        self.fp16 = False
        self.fp16_opt_level = 'O1'  # previous default was O2
        self.full_train_batch_size = 8  # previous default was 32
        self.per_gpu_eval_batch_size = 8
        self.output_dir = ''  # where to save model
        self.save_total_limit = 1  # limit to number of checkpoints saved in the output dir
        self.save_steps = 0  # do we save checkpoints every N steps? (TODO: put in terms of hours instead)
        self.use_tensorboard = False
        self.log_on_all_nodes = False
        self.server_ip = ''
        self.server_port = ''
        self.__required_args__ = ['model_type', 'model_name_or_path']
        self._post_init()

    def set_seed(self, seed=None):
        if seed is None:
            seed = self.seed
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if self.n_gpu > 0:
            torch.cuda.manual_seed_all(seed)

    def set_gradient_accumulation_steps(self):
        """
        when searching for full_train_batch_size in hyperparameter tuning we need to update
        the gradient accumulation steps to stay within GPU memory constraints
        :return:
        """
        if self.n_gpu * self.world_size * self.per_gpu_train_batch_size > self.full_train_batch_size:
            self.per_gpu_train_batch_size = self.full_train_batch_size // (self.n_gpu * self.world_size)
            self.gradient_accumulation_steps = 1
        else:
            self.gradient_accumulation_steps = self.full_train_batch_size // \
                                               (self.n_gpu * self.world_size * self.per_gpu_train_batch_size)

    def _basic_post_init(self):
        # Setup CUDA, GPU
        if self.local_rank == -1 or self.no_cuda:
            # NOTE: changed "cuda" to "cuda:0"
            self.device = torch.device("cuda:0" if torch.cuda.is_available() and not self.no_cuda else "cpu")
            self.n_gpu = torch.cuda.device_count()
        else:
            torch.cuda.set_device(self.local_rank)
            self.device = torch.device("cuda", self.local_rank)
            self.n_gpu = 1

        if self.n_gpu > 0:
            self.per_gpu_train_batch_size = self.full_train_batch_size // \
                                            (self.n_gpu * self.world_size * self.gradient_accumulation_steps)
        else:
            self.per_gpu_train_batch_size = self.full_train_batch_size // self.gradient_accumulation_steps

        self.stop_time = None
        if 'TIME_LIMIT_MINS' in os.environ:
            self.stop_time = time.time() + 60 * (int(os.environ['TIME_LIMIT_MINS']) - 5)

    def _post_init(self):
        self._basic_post_init()

        self._setup_logging()

        # Setup distant debugging if needed
        if self.server_ip and self.server_port:
            # Distant debugging - see https://code.visualstudio.com/docs/python/debugging#_attach-to-a-local-script
            import ptvsd
            print("Waiting for debugger attach")
            ptvsd.enable_attach(address=(self.server_ip, self.server_port), redirect_output=True)
            ptvsd.wait_for_attach()

        logger.warning(
            "On %s, Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
            socket.gethostname(),
            self.local_rank,
            self.device,
            self.n_gpu,
            bool(self.local_rank != -1),
            self.fp16,
        )
        # logger.info(f'hypers:\n{self}')

    def _setup_logging(self):
        # force our logging style
        for handler in logging.root.handlers[:]:
            logging.root.removeHandler(handler)
        if self.log_on_all_nodes:
            grank = self.global_rank
            class HostnameFilter(logging.Filter):
                hostname = socket.gethostname()
                if '.' in hostname:
                    hostname = hostname[0:hostname.find('.')]  # the first part of the hostname

                def filter(self, record):
                    record.hostname = HostnameFilter.hostname
                    record.global_rank = grank
                    return True

            handler = logging.StreamHandler()
            handler.setLevel(logging.INFO)
            handler.addFilter(HostnameFilter())
            format = logging.Formatter('%(hostname)s[%(global_rank)d] %(filename)s:%(lineno)d - %(message)s',
                                       datefmt='%m/%d/%Y %H:%M:%S')
            handler.setFormatter(format)
            logging.getLogger('').addHandler(handler)
        else:
            logging.basicConfig(format='%(filename)s:%(lineno)d - %(message)s',
                                datefmt='%m/%d/%Y %H:%M:%S',
                                level=logging.INFO)
        if self.global_rank != 0 and not self.log_on_all_nodes:
            try:
                logging.getLogger().setLevel(logging.WARNING)
            except:
                pass

    def to_dict(self):
        d = self.__dict__.copy()
        del d['device']
        return d

    def from_dict(self, a_dict):
        fill_from_dict(self, a_dict)
        self._basic_post_init()  # setup device and per_gpu_batch_size
        return self

    def __str__(self):
        return json.dumps(self.to_dict(), indent=2)

    def fill_from_args(self):
        fill_from_args(self)
        self._post_init()
        return self


def fill_from_dict(defaults, a_dict):
    for arg, val in a_dict.items():
        d = defaults.__dict__[arg]
        if type(d) is tuple:
            d = d[0]
        if isinstance(d, Enum):
            defaults.__dict__[arg] = type(d)[val]
        elif isinstance(d, EnumMeta):
            defaults.__dict__[arg] = d[val]
        else:
            defaults.__dict__[arg] = val


def fill_from_args(defaults):
    """
    Builds an argument parser, parses the arguments, updates and returns the object 'defaults'
    :param defaults: an object with fields to be filled from command line arguments
    :return:
    """
    parser = ArgumentParser()
    # if defaults has a __required_args__ we set those to be required on the command line
    required_args = []
    if hasattr(defaults, '__required_args__'):
        required_args = defaults.__required_args__
        for reqarg in required_args:
            if reqarg not in defaults.__dict__:
                raise ValueError(f'argument "{reqarg}" is required, but not present in __init__')
            if reqarg.startswith('_'):
                raise ValueError(f'arguments should not start with an underscore ({reqarg})')
    for attr, value in defaults.__dict__.items():
        # ignore members that start with '_'
        if attr.startswith('_'):
            continue

        # if it is a tuple, we assume the second is the help string
        help_str = None
        if type(value) is tuple and len(value) == 2 and type(value[1]) is str:
            help_str = value[1]
            value = value[0]

        # check if it is a type we can take on the command line
        if type(value) not in [str, int, float, bool] and not isinstance(value, Enum) and not isinstance(value, type):
            raise ValueError(f'Error on {attr}: cannot have {type(value)} as argument')
        if type(value) is bool and value:
            raise ValueError(f'Error on {attr}: boolean arguments (flags) must be false by default')

        # also handle str to enum conversion
        t = type(value)
        if isinstance(value, Enum):
            t = str
            value = value.name
        elif isinstance(value, EnumMeta):
            t = type
            value = str

        if t is type:
            # indicate a required arg by specifying a type rather than value
            parser.add_argument('--'+attr, type=value, required=True, help=help_str)
        elif t is bool:
            # support bool with store_true (required false by default)
            parser.add_argument('--'+attr, default=False, action='store_true', help=help_str)
        else:
            parser.add_argument('--'+attr, type=t, default=value, help=help_str, required=(attr in required_args))
    args = parser.parse_args()
    # now update the passed object with the arguments
    fill_from_dict(defaults, args.__dict__)
    # call _post_argparse() if the method is defined
    try:
        defaults._post_argparse()
    except AttributeError:
        pass
    return defaults