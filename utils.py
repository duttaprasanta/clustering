import numpy as np
import os
def export(filename, labels, foldername='.'):
    unique = list(set(labels))
    clusters =[]
    for u in unique:
        clusters.append(sorted(np.where(labels==u)[0]))
    clusters.sort()
    try:
        os.makedirs(foldername)
    except:
        pass
    f = open(f'{foldername}/{filename}','w')
    for c in clusters:
        f.write(str(c)[1:-1])
        f.write('\n')
    f.close()
    print(f'Exported successfully into {foldername}/{filename}')