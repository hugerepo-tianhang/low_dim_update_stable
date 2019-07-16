import numpy as np
def comp_dict(a,b):
    if a is None and b is None:
        return True
    if a.keys() != b.keys():
        return False
    keys = a.keys()

    result = []
    for key in keys:
        re = (np.array(a[key]) == np.array(b[key])).all()

        result.append(re)
    return all(result)
