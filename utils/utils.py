import os
from glob import glob


def fix_path():
    # os.chdir('/Users/imad/PycharmProjects/')
    current = os.getcwd()
    # print(os.getcwd())
    if 'HIP-PROJECT' in current:
        fixed_path = os.path.join(current.split('HIP-PROJECT')[0], 'HIP-PROJECT')
        # print(fixed_path)
        os.chdir(fixed_path)
    elif os.path.isdir(os.path.join(current, 'HIP-PROJECT')):
        fixed_path = os.path.join(current, 'HIP-PROJECT')
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
                if 'HIP-PROJECT' in path:
                    fixed_path = os.path.join(path.split('HIP-PROJECT')[0], 'HIP-PROJECT')
                    # print(fixed_path)
                    os.chdir(fixed_path)
                    return fixed_path
    return fixed_path
