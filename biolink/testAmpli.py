from testing import ampligraph_test
# import biolink
import logging
import os
import sys

logger = logging.getLogger(os.path.basename(sys.argv[0]))

def main():
    ampligraph_test.main()

if __name__ == '__main__':
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)
    # print(' '.join(sys.argv))
    main()
