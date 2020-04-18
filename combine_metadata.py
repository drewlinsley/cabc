import numpy as np
import os
import sys

def find_files(files, dirs=[], contains=[]):
    for d in dirs:
        onlyfiles = [os.path.join(d, f) for f in os.listdir(d) if os.path.isfile(os.path.join(d, f))]
        for i, part in enumerate(contains):
            files += [os.path.join(d, f) for f in onlyfiles if part in f]
        onlydirs = [os.path.join(d, dd) for dd in os.listdir(d) if os.path.isdir(os.path.join(d, dd))]
        if onlydirs:
            files += find_files([], onlydirs, contains)

    return files

if __name__=="__main__":

    dataset_path = str(sys.argv[1])
    subpath = 'metadata'
    metadata_fullpath = os.path.join(dataset_path, subpath)

    files = find_files([], dirs=[metadata_fullpath], contains=['.npy'])
    for i, fn in enumerate(files):
        if i == 0:
            metadata = np.load(fn)
        else:
            metadata = np.concatenate([metadata, np.load(fn)], axis=0)

    np.save(os.path.join(metadata_fullpath,'combined.npy'), metadata)
