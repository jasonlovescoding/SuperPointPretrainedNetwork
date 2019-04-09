import cv2

def read_image(impath, img_size=None):
  """ Read image as grayscale and resize to img_size.
  Inputs
    impath: Path to input image.
    img_size: (H, W) tuple specifying resize size.
  Returns
    grayim: float32 numpy array sized H x W with values in range [0, 1].
  """
  grayim = cv2.imread(impath, cv2.IMREAD_GRAYSCALE)
  if grayim is None:
    raise Exception('Error reading image %s' % impath)
  # Image is resized via opencv.
  interp = cv2.INTER_AREA
  if img_size is not None:
    grayim = cv2.resize(grayim, (img_size[1], img_size[0]), interpolation=interp)
  grayim = (grayim.astype('float32') / 255.)
  return grayim