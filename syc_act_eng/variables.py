import os 
import pathlib

def _get_home_dir() -> pathlib.Path:
    return pathlib.Path.home().absolute()

def get_project_dir() -> pathlib.Path:
    return pathlib.Path(__file__).parent.absolute()

PROJECT_DIR = str(get_project_dir())
