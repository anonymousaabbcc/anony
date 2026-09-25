import sys
from validate_semantics_v3 import main
if __name__ == '__main__':
    if '--block' not in sys.argv:
        sys.argv.extend(['--block', 'understanding'])
    main()
