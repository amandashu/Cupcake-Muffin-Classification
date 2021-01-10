import os
import shutil

def remove_ouputs():
    """
    Removes files specified in locations variable
    """
    locations = ['data', 'results']
    for l in locations:
        if os.path.exists(l):
            shutil.rmtree(l)
