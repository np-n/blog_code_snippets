"""
--- Personal
--- Created by : Netra Prasad Neupane
--- Created on : 11/17/23
--- Use Case:
"""
import cv2





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
    image = cv2.imread("documents/org_image_document.png")
    cv2.imshow("Original", image)
    rotate_image(image)




















