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
        f'--data covid, --model transe ' \
        f'--batch-size {c["batch"]} --epochs {c["epoch"]} '\
        f'--embedding-size {c["emb_size"]} --learning-rate {c["lr"]} ' \
        f'--regulariser {c["reg"]} --reg-weight {c["reg_weight"]} ' \
        f'--optimizer {c["optim"]} --transe_norm {c["transe_norm"]} ' \
        f'--loss_margin {c["loss_margin"]} --loss pair_hinge ' \
        f'--nb_negs {c["nb_negs"]} '\
        f'--valid --quiet'
    return command


def to_logfile(c, path):
    outfile = "{}/transe_prhinge_covid{}.log".format(path, summary(c).replace("/", "_"))
    return outfile


def main(argv):
    hyp_space = dict(
        batch=[1024, 4308],
        epoch=[1000],
        emb_size=[50, 100, 150, 200],
        lr=[0.1, 0.01],
        transe_norm=['l1', 'l2'],
        optim=['adagrad'],
        reg=['n3', 'f2'],
        reg_weight=[0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        nb_negs=[10, 15, 100],
        loss_margin=[1, 5, 9]
    )

    configurations = list(cartesian_product(hyp_space))

    path = '/home/acalvi/Dissertation/Aladdin/transe_covid'
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
#$ -e $HOME/GRIDtranse.err
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
