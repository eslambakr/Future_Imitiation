import os
def create_dirs(lis):
    for name in lis:
        if not os.path.exists(name):
            os.mkdir(name)
