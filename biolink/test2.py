from testing import experiment_ComplEx
import biolink
import logging
import os

logger = logging.getLogger(os.path.basename(sys.argv[0]))

def main(argv):
    experiment_ComplEx.main(argv)

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    print(' '.join(sys.argv))
    main(sys.argv[1:])
