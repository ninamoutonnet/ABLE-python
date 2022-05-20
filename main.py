from skimage import io


def initialise(name):




if __name__ == '__main__':
    im = io.imread('an_image.tif')
    print(im.shape)