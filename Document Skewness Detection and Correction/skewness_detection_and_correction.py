"""
--- Personal
--- Created by : Netra Prasad Neupane
--- Created on : 11/17/23
--- Use Case:
"""
import cv2
import numpy as np


def _ensure_gray(image):
    try:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except cv2.error:
        pass
    return image


def _ensure_optimal_square(image):
    assert image is not None, image
    nw = nh = cv2.getOptimalDFTSize(max(image.shape[:2]))
    output_image = cv2.copyMakeBorder(
        src=image,
        top=0,
        bottom=nh - image.shape[0],
        left=0,
        right=nw - image.shape[1],
        borderType=cv2.BORDER_CONSTANT,
        value=255,
    )
    return output_image


def _get_fft_magnitude(image):
    gray = _ensure_gray(image)
    opt_gray = _ensure_optimal_square(gray)

    # thresh
    opt_gray = cv2.adaptiveThreshold(
        ~opt_gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 15, -10
    )

    # perform fft
    dft = np.fft.fft2(opt_gray)
    shifted_dft = np.fft.fftshift(dft)

    # get the magnitude (module)
    magnitude = np.abs(shifted_dft)
    return magnitude


def _get_angle_radial_projection(m, angle_max=None, num=None, W=None):
    """Get angle via radial projection.

    Arguments:
    ------------
    angle_max : float
    num : int
      number of angles to generate between 1 degree
    """
    assert m.shape[0] == m.shape[1]
    r = c = m.shape[0] // 2

    if angle_max is None:
        pass

    if num is None:
        num = 20

    tr = np.linspace(-1 * angle_max, angle_max, int(angle_max * num * 2)) / 180 * np.pi
    profile_arr = tr.copy()

    def f(t):
        _f = np.vectorize(
            lambda x: m[c + int(x * np.cos(t)), c + int(-1 * x * np.sin(t))]
        )
        _l = _f(range(0, r))
        val_init = np.sum(_l)
        return val_init

    vf = np.vectorize(f)
    li = vf(profile_arr)

    a = tr[np.argmax(li)] / np.pi * 180

    if a == -1 * angle_max:
        return 0
    return a


def get_skewed_angle(
        image: np.ndarray, vertical_image_shape: int = None, angle_max: float = None
):
    """Getting angle from a given document image.

    image : np.ndarray
    vertical_image_shape : int
      resize image as preprocessing
    angle_max : float
      maximum angle to searching
    """
    assert isinstance(image, np.ndarray), image

    # if vertical_image_shape is None:
    #     vertical_image_shape = 512

    if angle_max is None:
        angle_max = 15

    # resize
    if vertical_image_shape is not None:
        ratio = vertical_image_shape / image.shape[0]
        image = cv2.resize(image, None, fx=ratio, fy=ratio)

    m = _get_fft_magnitude(image)
    a = _get_angle_radial_projection(m, angle_max=angle_max)
    return a


def correct_text_skewness(image):
    """
    Method to rotate image by n degree
    @param image:
    return:
    """
    # cv2_imshow(image)
    h, w, c = image.shape
    x_center, y_center = (w // 2, h // 2)

    # Find angle to rotate image
    rotation_angle = get_skewed_angle(image)
    print(f"[INFO]: Rotation angle is {rotation_angle}")

    # Rotate the image by given n degree around the center of the image
    M = cv2.getRotationMatrix2D((x_center, y_center), rotation_angle, 1.0)
    borderValue = (255, 255, 255)

    rotated_image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=borderValue)
    return rotated_image


def rotate_image(image):
    (h, w) = image.shape[:2]
    (cX, cY) = (w // 2, h // 2)
    # rotate our image by 45 degrees around the center of the image
    M = cv2.getRotationMatrix2D((cX, cY), -15, 1.0)
    rotated_image = cv2.warpAffine(image, M, (w, h), borderValue=(255, 255, 255))
    cv2.imwrite("documents/skewed_image_document.png", rotated_image)
    cv2.imshow("15-degree rotated image", rotated_image)
    cv2.waitKey(0)



if __name__ == "__main__":
    image = cv2.imread("documents/skewed_image_document.png")
    cv2.imshow("Original Skewed Image", image)
    # rotate_image(image)
    skew_corrected_image = correct_text_skewness(image)
    cv2.imshow("Skew Corrected Image", skew_corrected_image)
    cv2.imwrite("documents/skew_corrected_image.png", skew_corrected_image)
    cv2.waitKey(0)




















