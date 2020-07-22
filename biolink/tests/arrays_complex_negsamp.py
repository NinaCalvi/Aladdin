import itertools
import os
import os.path

import sys
import logging


def cartesian_product(dicts):
    return (dict(zip(dicts, x)) for x in itertools.product(*dicts.values()))


def summary(configuration):
    kvs = sorted([(k, v) for k, v in configuration.items()], key=lambda e: e[0])
    return '_'.join([('%s=%s' % (k, v)) for (k, v) in kvs if k not in {'c', 'd'}])


def to_cmd(c, _path=None):
    command = f'PYTHONPATH=. python ./test2.py ' \
        f'--data pse ' \
        f'--batch-size {c["batch"]} --epochs {c["epoch"]} '\
        f'--embedding-size {c["emb_size"]} --learning-rate {c["lr"]} ' \
        f'--regulariser {c["reg"]} --reg-weight {c["reg_weight"]} ' \
        f'--optimizer {c["optim"]} --loss pw_square --nb-negs 200 ' \
        f'--quiet --valid'
    return command


def to_logfile(c, path):
    outfile = "{}/complex_ng_covid{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        batch=[268, 512, 1024],
        epoch=[100],
        emb_size=[50, 100, 150, 200],
        lr=[0.1, 0.01],
        optim=['adagrad'],
        reg=['n3'],
        reg_weight=[0.001, 0.005, 0.01, 0.05, 0.0001, 0.0005]
    )

    configurations = list(cartesian_product(hyp_space))

    path = '/home/acalvi/Dissertation/Aladdin/pse_neg_sampl/'
    is_rc = False

    # Check that we are on the UCLCS cluster first
    if os.path.exists('/home/acalvi/'):
        is_rc = True
        # If the folder that will contain logs does not exist, create it
        if not os.path.exists(path):
            os.makedirs(path)

    command_lines = set()
    for cfg in configurations:
        logfile = to_logfile(cfg, path)

        completed = False
        if is_rc is True and os.path.isfile(logfile):
            with open(logfile, 'r', encoding='utf-8', errors='ignore') as f:
                content = f.read()
                completed = 'Training finished' in content

        if not completed:
            command_line = '{} > {} 2>&1'.format(to_cmd(cfg), logfile)
            command_lines |= {command_line}

    # Sort command lines and remove duplicates
    sorted_command_lines = sorted(command_lines)

    import random
    rng = random.Random(0)
    rng.shuffle(sorted_command_lines)

    nb_jobs = len(sorted_command_lines)

    header = """#!/bin/bash -l
#$ -cwd
#$ -S /bin/bash
#$ -o /dev/null
#$ -e $HOME/GRIDpse.err
#$ -t 1-{}
#$ -l tmem=15G
#$ -l h_rt=38:00:00
#$ -l gpu=true
conda activate libkge_new
export LANG="en_US.utf8"
export LANGUAGE="en_US:en"
cd $HOME/Dissertation/Aladdin/biolink/
""".format(nb_jobs)

    print(header)

    for job_id, command_line in enumerate(sorted_command_lines, 1):
        print('test $SGE_TASK_ID -eq {} && sleep 10 && {}'.format(job_id, command_line))


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main(sys.argv[1:])
