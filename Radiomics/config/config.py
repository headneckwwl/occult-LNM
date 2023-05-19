import os
import platform


class Config:
    # Where to save scripts.
    SCRIPTS_ROOT = 'bin'
    # Scripts' template
    SCRIPTS_TEMP = './config/scripts_with_argparse.onekey_algo'
    # Manual template
    MAN_TEMP = './config/man.onekey_algo'
    # Which module to generate as scripts.
    MODULE_DIRs = [os.path.join(os.getcwd(), 'core')]
    # PY_INTERPRETER
    PY_INTERPRETER = r'C:\Users\onekey\.conda\envs\onekey\python.exe' \
        if platform.system() == 'Windows' else None
    # Learning to rank HOME
    ONEKEY_HOME = os.path.dirname(os.getcwd())
