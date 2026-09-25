import sys
from validate_semantics_v3 import main
if __name__ == '__main__':
    print('V3 note: validate_u7_u16.py now runs the full U1-U18 semantic validator.')
    if '--block' not in sys.argv:
        sys.argv.extend(['--block', 'understanding'])
    main()
