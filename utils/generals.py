import json
import shutil

def sec2HMS(sec):
    h = sec // 3600
    m = (sec % 3600) // 60
    s = sec % 60
    return int(h), int(m), s

def check_and_delete(target):
    k = input('Delete? --> y/n: ')
    if k.lower() == 'y':
        shutil.rmtree(target)
        return True
    else:
        return False

def load_json(path):
    with open(path) as f:
        config = json.load(f)
    return config