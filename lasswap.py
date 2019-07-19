import os


def create_swapped_las(directory, file, target_folder, lastools_bin_location):
    """
    Creates a .las file with the x and n values swapped
    The output name is ALWAYS xynz_swap.las
    """
    las2txt = os.path.join(lastools_bin_location, 'las2txt')
    fullfile = os.path.join(directory, file)
    command = r'-parse xynz -o'
    outname = os.path.join(target_folder, 'xyzn_orig.txt')
    convert = las2txt+' '+fullfile+' '+' '+command+' '+outname
    os.system(convert)

    txt2las = os.path.join(lastools_bin_location, 'txt2las')
    swapcommand = r'-parse xyzn -o'
    swapoutname = os.path.join(target_folder, 'xynz_swap.las')
    swap = txt2las+' '+outname+' '+' '+swapcommand+' '+swapoutname
    os.system(swap)

    os.remove(outname)