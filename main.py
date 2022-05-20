import numpy as np
from skimage import io

def initialise(name):
    return


def get_locs_in_range(sub_loc, dim):

    # print(sub_loc[1][:])
    # print(sub_loc.shape)

    col1 =[]
    col2 =[]

    for i in range(sub_loc.shape[1]):
        if (sub_loc[0][i]>0 and sub_loc[0][i] <= dim[0]) and  (sub_loc[1][i]>0 and sub_loc[1][i] <= dim[1]):
            col1.append(sub_loc[0][i])
            col2.append(sub_loc[1][i])

    sub_loc = np.array([col1, col2])
    # print(sub_loc)
    # return sub_loc # this is the official return value, similar to the MATLAB code
    return col1, col2


def sub2ind(array_shape, rows, cols):
    temp = np.add(rows, -1)
    temp = np.multiply(temp, array_shape[1])
    temp = np.add(temp, cols)
    return temp


def crossCorr(video):

    dim = [video.shape[0], video.shape[1]]
    # print(dim)

    t_len = video.shape[2]
    # print(t_len)

    mods = np.arange(t_len)*dim[0]*dim[1]
    # print(mods)

    image = np.zeros(dim)
    # print(image)

    video_temp = np.transpose(video)
    video_reshaped = np.reshape(video_temp, [ dim[0] * dim[1] * t_len])

    # print(video_reshaped[8192000-20:8192000])
    # print(video_reshaped.shape)

    for ii in range(dim[0]):
        for jj in range(dim[1]):
            sub_loc = np.array([ [ii, ii, ii, ii+1, ii+1, ii+2, ii+2, ii+2],
                        [jj, jj+1, jj+2, jj, jj+2, jj, jj+1, jj+2 ] ])


            sub_loc_col1, sub_loc_col2  = get_locs_in_range(sub_loc, dim)
            sub_loc_col1.insert(0, ii+1)
            sub_loc_col2.insert(0, jj+1)
            sub_loc = np.array([sub_loc_col1, sub_loc_col2])
            ind_loc = sub2ind(dim, sub_loc_col1, sub_loc_col2)
            vid_loc = np.zeros([ind_loc.size, t_len])
            for i in range(t_len):
                vid_loc[:,i] = ind_loc + mods[i]



    print(vid_loc)


    return


if __name__ == '__main__':

    video = io.imread('multipage_tif_resized.tif')     # (500,128,128)
    # print(video.shape)
    video = np.transpose(video, (1, 2, 0))    # (128,128,500)
    # print(video.shape)
    meanIm = np.mean(video, 2)
    # print(meanIm)

    crossCorr(video)