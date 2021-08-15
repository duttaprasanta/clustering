import numpy as np
import os

def export(filename, labels, foldername='.'):
    '''
    This function exports the clustering results into a file inside a directory
    * Parameters :
        1. filename : str 
        2. labels : array containing labels
        3. foldername : str, default = current folder
    '''
    unique = list(set(labels)) # Getting unique labels
    clusters =[]
    for u in unique:    # For every label
        clusters.append(sorted(np.where(labels==u)[0])) # Getting indices for a label
    clusters.sort()
    try:
        os.makedirs(foldername) # Creating specified folder
    except:
        pass
    f = open(f'{foldername}/{filename}','w') # opening file
    for c in clusters:
        f.write(str(c)[1:-1])   # Writing indices into file
        f.write('\n')   # Adding a newline after each line 
    f.close()   # Closing file
    print(f'Exported successfully into {foldername}/{filename}')