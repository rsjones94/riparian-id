from objective_functions import *


def interpret(extensions, params):
    """
    Takes a list of file extensions and a list of obj fxn parameters and interprets the
    meaning

    Returns: a str representing the meaning of the vector
    """
    plist = params.copy()
    plist = [plist[x:x + 3] for x in range(0, len(plist), 3)]

    prints = []
    for ext, sub in zip(extensions,plist):
        out = f'{ext}: {num_to_filter(sub[0]).__name__}, args {sub[1]} and {sub[2]}'
        print(out)
        prints.append(out)
    return prints


def extracted_title(res, levels, extensions):
    """
    Makes a title given a res object and the levels

    Args:
        res: the res object
        levels: a vector where each entry corresponds to a level of res. If the entry is an integer, that level is
                held invariant at the value of the index specified. If 'vary', that level is varied. The length
                should be one short of the sublist lengths

    Returns:
        a str

    """
    vary_level = levels.index('vary')
    vary_length = res.shape[vary_level]
    vary_range = range(0,vary_length)
    slicer = []
    for i, val in enumerate(levels):
        if i == vary_level:
            slicer.append(vary_range)
        else:
            slicer.append(val)
    sublists = res[slicer]

    l1 = sublists[0][:-1]
    initial = interpret(extensions, l1)
    initial = '\n'.join(' '.join(sub) for sub in initial)

    varying_arg = vary_level % 3
    varying_ext = extensions[int(np.floor(vary_level/3))]
    ex = f'{varying_ext}, arg {varying_arg}'

    return initial, ex


def extract_param(res, levels):
    """
    Takes res and extracts the level specified keeping all other vector entries invariant

    Args:
        res: the res object
        levels: a vector where each entry corresponds to a level of res. If the entry is an integer, that level is
                held invariant at the value of the index specified. If 'vary', that level is varied. The length
                should be one short of the sublist lengths

    Returns:
        a tuple of lists where the first lis is a list of the varies parameter and the second is a list of the
        values of the objective function
    """
    vary_level = levels.index('vary')
    vary_length = res.shape[vary_level]
    vary_range = range(0,vary_length)
    slicer = []
    for i, val in enumerate(levels):
        if i == vary_level:
            slicer.append(vary_range)
        else:
            slicer.append(val)
    sublists = res[slicer]
    vary_vals = [sublist[vary_level] for sublist in sublists]
    obj_vals = [sublist[-1] for sublist in sublists]
    return vary_vals, obj_vals
