from collections import OrderedDict
import copy
import re
import os
import shutil
import time
import hashlib
import json
import warnings

import torch
from torch.hub import download_url_to_file


class ModelSaver:
    """ Model weights + metadata saver

    In default, a single weight file is always saved on the local machine.
    Then according to the value of `util.save-model.target`, ModelSaver copy
    the local file.
    into the specified target.

    Params
    -----------
    config: Munch instance
        args.util.save_model (consists of `target`, `periods`, and `metadata`)
    checkpoint_dir: str (optional, default='./checkpoints')
        Directory to save local files
    """
    supported_targets = ('local')
    supported_periods = ('every_epoch', 'last', 'best', r'last(\d+)',
                         r'every_(\d+)_epochs', 'none')

    def __init__(self, checkpoint_dir='./checkpoints', target='local', periods=('last', 'best')):
        """ Constructor of ModelSaver
        """
        if target not in ModelSaver.supported_targets:
            raise ValueError(
                f'Unsupported --util.save_weights.target "{target}"; '
                f'should be one of [{", ".join(ModelSaver.supported_targets)}].'
            )

        for period in periods:
            matched = False
            for rule in ModelSaver.supported_periods:
                if re.match(rule, period):
                    matched = True

            if not matched:
                raise ValueError(
                    f'Unsupported --util.save_weights.period {periods}; '
                    'should be one of'
                    f'[{", ".join(ModelSaver.supported_periods)}]')

        self.target = target
        self.periods = periods
        self.checkpoint_dir = checkpoint_dir

        os.makedirs(checkpoint_dir, exist_ok=True)

    def save(self, step, num_steps, state, summary,
             is_best=False, prefix=''):
        """ Save model with state(weights), and summary data.

        Params
        ----------
        step: int
            start from 0
        num_steps: int
            total number of steps
        state: dict
            state dict of the model
        summary: dict
            result (accuracy, # params, etc) of the model
        is_best: bool (optional, default=False)
            true if current model has best accuracy
        prefix: str (optional, default='')
            prefix of the file
        """

        checkpoint_name = f'{prefix}{(step + 1)}'
        summary['checkpoint'] = checkpoint_name

        # Regardless of the upload target, always save three files on the local:
        # model weight(1), args dump file(2), and summary dump file(3).
        filepath = f'{self.checkpoint_dir}/checkpoint-{prefix}last.pth'

        # backup last d epochs
        num_last_buffer = 0
        for period_config in self.periods:
            # last[N]
            match = re.match(r'last(\d+)', period_config)
            if match:
                if int(match.group(1)) > num_last_buffer:
                    num_last_buffer = int(match.group(1))

        for i in range(num_last_buffer):
            if num_last_buffer - i - 1 == 0:
                filepath_src = f'{self.checkpoint_dir}/checkpoint-{prefix}last.pth'
            else:
                filepath_src = f'{self.checkpoint_dir}/checkpoint-{prefix}last{num_last_buffer-i-1}.pth'
            if os.path.isfile(filepath_src):
                copy_json = os.path.isfile(filepath_src + '.summary.json')
                checkpoint_name_last = f'{prefix}last{num_last_buffer-i}'
                self._copy_to_target(checkpoint_name_last, filepath_src, copy_json)

        torch.save(state, filepath)
        with open(filepath + '.summary.json', 'w') as fout:
            fout.write(json.dumps(summary))

        # Then copy to the target based on the period config
        if self._is_time_to_copy(step, num_steps):
            self._copy_to_target(checkpoint_name, filepath)

        # To handle `best` option on the `periods`,
        # we have to save temporal best checkpoint file on each epoch.
        # And on the last iteration, upload the preserved checkpoint file to
        # the target.
        if 'best' in self.periods:
            checkpoint_name = f'{prefix}best'
            summary['checkpoint'] = checkpoint_name
            best_filepath = \
                f'{self.checkpoint_dir}/checkpoint-{prefix}best.pth'

            if is_best:
                shutil.copyfile(filepath, best_filepath)
                with open(best_filepath + '.summary.json', 'w') as fout:
                    fout.write(json.dumps(summary))

            if step == num_steps - 1:
                self._copy_to_target(checkpoint_name, best_filepath)

        # Remove temporal local file if 'last' is not in the periods
        last_flag = False
        for period in self.periods:
            if period.startswith('last'):
                last_flag = True
                break
        if not last_flag and step == num_steps - 1:
            os.remove(filepath)
            os.remove(filepath + '.summary.json')

    def _is_time_to_copy(self, step, num_steps):
        """ Returns whether current epoch is the time to save the model
        """
        for period_config in self.periods:
            if period_config == 'every_epoch':
                return True
            # if period_config == 'last' and step == num_steps - 1:
            #     return True
            # every_[N]_epochs
            match = re.match(r'every_(\d+)_epochs', period_config)
            if match:
                if (step + 1) % int(match.group(1)) == 0 and (step + 1) > 0:
                    return True
        return False

    def _copy_to_target(self, checkpoint_name, filepath, copy_json=True):
        """ Copy temporal local file to specific target
        """
        if self.target == 'local':
            # Copy the specific checkpoint on the local
            new_filepath = f'{self.checkpoint_dir}/' \
                           f'checkpoint-{checkpoint_name}.pth'

            if filepath == new_filepath:
                return

            shutil.copyfile(filepath, new_filepath)
            if copy_json:
                shutil.copyfile(filepath + '.summary.json',
                                new_filepath + '.summary.json')

