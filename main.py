import numpy as np
from skimage import io
import skimage
import matplotlib.pyplot as plt
import skimage.morphology as morph
from skimage.filters import threshold_otsu

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
    image = np.zeros((dim[0], dim[1]))

    for ii in range(dim[0]):
        for jj in range(dim[1]):
            sub_loc = np.array([ [ii, ii, ii, ii+1, ii+1, ii+2, ii+2, ii+2],
                        [jj, jj+1, jj+2, jj, jj+2, jj, jj+1, jj+2 ] ])


            sub_loc_col1, sub_loc_col2  = get_locs_in_range(sub_loc, dim)
            sub_loc_col1.insert(0, ii+1)
            sub_loc_col2.insert(0, jj+1)
            sub_loc = np.array([sub_loc_col1, sub_loc_col2])
            ind_loc = sub2ind(dim, sub_loc_col1, sub_loc_col2)
            # vid_loc = np.zeros([ind_loc.size * t_len])
            vid_loc = np.empty([], dtype='int')
            # vid_loc = np.zeros([ind_loc.size, t_len])
            for i in range(t_len):
                # vid_loc[:,i] = ind_loc + mods[i]
                vid_loc = np.append(vid_loc, ind_loc + mods[i] - 1)
                if i == 0:
                    vid_loc = np.delete(vid_loc, 0)

            raw_ts = np.take(video_reshaped, vid_loc)
            raw_ts_rs = np.reshape(raw_ts, (t_len, ind_loc.size))
            raw_ts_rs = np.transpose(raw_ts_rs)

            a_del = np.delete(raw_ts_rs, 0, 0)
            Y = np.mean(a_del, axis=0)
            X = raw_ts_rs[0, :]
            mu_Y = np.mean(Y)
            sig_Y = np.std(Y)
            Y = Y - mu_Y
            Y = Y/sig_Y
            X = (X - np.mean(X))/np.std(X)

            image[ii, jj] = np.dot((Y), np.transpose(X))

    image = image/t_len
    image = np.transpose(image)
    # print(raw_ts)
    # print(raw_ts.shape)

    # print(raw_ts_rs)
    # print(raw_ts_rs.shape)
    # print(a_del)
    # print(a_del.shape)
    # print(Y.shape)
    # print(Y.shape)
    # print(X.shape)
    # print(mu_Y)
    # print(sig_Y) ROUGHLY DIFFERENT: 1.4891315888426007 in python, 1.4906

    return image

def imextendedmax(img, H):
    mask = img.copy()
    seed = mask - H
    hmax = morph.reconstruction(seed, mask, method='dilation')
    hmax = img - hmax
    # threshold and convert into a binary image
    img_thresh_otsu = hmax > 0.6*(np.max(hmax))

    return img_thresh_otsu

def separate_mask(img, dim):
    # get the number of connected components
    number_components = np.max(img)

    # init the array
    masks = np.ones([dim[0], dim[1], number_components])
    # print(masks.shape)

    for i in range(number_components):
        masks[:, :, i] = img
        # replace values that are equal to the component number by -1
        masks[:, :, i][masks[:, :, i] != i + 1] = 0
        masks[:, :, i][masks[:, :, i] == i + 1] = -1
        masks[:, :, i][masks[:, :, i] == 0] = 1
        # replace values that are above or below the component number by 1
        # masks[:, :, i][masks[:,:,i] > i + 1] = 1
        # masks[:, :, i][masks[:, :, i] < i + 1] = 1

    return masks

def initialise(metric, radius, alpha, blur_radius):
    '''
    NOTE: the init method is not exactly as the MATLAB one.
    - maxSize is not used to filter out any ROI,
    - the np.std and std MATLAB functions do not match perfectly,
    - the MATLAB imextendedmax function was replicated to the best of my abilities
    but no 1 to 1 correspondance exist between python and MATLAB for this function.
    There is an extra parameter in the python imextendedmax function controlling the
    sensibility of the binary mask creation
    '''
    dim = metric.shape
    maxSize = round(np.pi * pow(radius, 2) * 1.5)
    # print(maxSize)

    # blur the image
    metric = skimage.filters.gaussian(metric, sigma=blur_radius, mode = 'nearest', truncate=2.0)
    metric = np.array(metric)
    # print(metric.shape)

    # find the peaks of metric
    h = np.sqrt(np.nanvar(metric))
    # print(h)

    # replace the possible NaN values by 0
    metric[np.isnan(metric)] = 0

    BW = imextendedmax(metric, alpha * h)

    '''fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(BW, aspect='auto', interpolation='nearest')
    plt.show()'''

    # perform connected component analysis
    labeled_image, count = skimage.measure.label(BW, connectivity=2, return_num=True)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(labeled_image, aspect='auto', interpolation='nearest')
    plt.show()

    masks = separate_mask(labeled_image, dim)
    '''fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.imshow(masks[:, :, 0], aspect='auto', interpolation='nearest')
    plt.show()'''

    return masks

if __name__ == '__main__':

    video = io.imread('multipage_tif_resized.tif')     # (500,128,128)
    # print(video.shape)
    video = np.transpose(video, (1, 2, 0))    # (128,128,500)
    # print(video.shape)
    meanIm = np.mean(video, 2)
    # print(meanIm)

    corrIm = crossCorr(video)
    print('Done producing the correlation image')

    # plt.imshow(corrIm, interpolation='nearest')
    # plt.show()

    '''Initialisation'''
    # Parameters
    radius = 7
    alpha = 0.55
    blur_radius = 1.5

    phi_0 = initialise(corrIm, radius, alpha, blur_radius)

