import os
from glob import glob

root = 'DeepLearning'


def fix_path():
    # os.chdir('/Users/imad/PycharmProjects/')
    current = os.getcwd()
    # print(os.getcwd())
    if root in current:
        fixed_path = os.path.join(current.split(root)[0], root)
        print(fixed_path)
        os.chdir(fixed_path)
    elif os.path.isdir(os.path.join(current, root)):
        fixed_path = os.path.join(current, root)
        # print(fixed_path)
        os.chdir(fixed_path)
    else:
        search = current + "/*/"
        if '//' in search:
            search = str(search).replace('//', '/')
        potentials = glob(search)
        potentials = [glob(search_i + "/*/") for search_i in potentials]
        # print(potentials)
        fixed_path = current
        for p in potentials:
            for path in p:
                if root in path:
                    fixed_path = os.path.join(path.split(root)[0], root)
                    # print(fixed_path)
                    os.chdir(fixed_path)
                    return fixed_path
    return fixed_path
