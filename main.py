import cv2
import numpy as np
import argparse

def GetPerspective(img, FOV, THETA, PHI, height, width):

  equ_h = 512
  equ_w = 1024
  equ_cx = (equ_w - 1) / 2.0
  equ_cy = (equ_h - 1) / 2.0

  wFOV = FOV
  hFOV = float(height) / width * wFOV

  w_len = np.tan(np.radians(wFOV / 2.0))
  h_len = np.tan(np.radians(hFOV / 2.0))


  x_map = np.ones([height, width], np.float32)
  y_map = np.tile(np.linspace(-w_len, w_len,width), [height,1])
  z_map = -np.tile(np.linspace(-h_len, h_len,height), [width,1]).T

  D = np.sqrt(x_map**2 + y_map**2 + z_map**2)
  xyz = np.stack((x_map,y_map,z_map),axis=2)/np.repeat(D[:, :, np.newaxis], 3, axis=2)
        
  y_axis = np.array([0.0, 1.0, 0.0], np.float32)
  z_axis = np.array([0.0, 0.0, 1.0], np.float32)
  [R1, _] = cv2.Rodrigues(z_axis * np.radians(THETA))
  [R2, _] = cv2.Rodrigues(np.dot(R1, y_axis) * np.radians(-PHI))

  xyz = xyz.reshape([height * width, 3]).T
  xyz = np.dot(R1, xyz)
  xyz = np.dot(R2, xyz).T
  lat = np.arcsin(xyz[:, 2])
  lon = np.arctan2(xyz[:, 1] , xyz[:, 0])

  lon = lon.reshape([height, width]) / np.pi * 180
  lat = -lat.reshape([height, width]) / np.pi * 180

  lon = lon / 180 * equ_cx + equ_cx
  lat = lat / 90  * equ_cy + equ_cy

  persp = cv2.remap(img, lon.astype(np.float32), lat.astype(np.float32), cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
  return persp

def empty(a):
    pass

def create_panorama(file_path: str):

    title_window = '3d room panorama'

    cv2.namedWindow(title_window)
    img = cv2.imread(file_path)
    if img is None:
        print('Такого файла не существует или ошибка чтения')
        exit(1)
    
    original_img = img.copy()
    trackbar_x = 'rotate_x'
    cv2.createTrackbar(trackbar_x, title_window , 0, 180, empty)
    cv2.setTrackbarMin(trackbar_x, title_window, -180)
    cv2.setTrackbarMax(trackbar_x, title_window, 180)

    trackbar_y = 'rotate_y'
    cv2.createTrackbar(trackbar_y, title_window , 0, 180, empty)
    cv2.setTrackbarMin(trackbar_y, title_window, -180)
    cv2.setTrackbarMax(trackbar_y, title_window, 180)

    trackbar_f = 'F'
    cv2.createTrackbar(trackbar_f, title_window , 120, 140, empty)
    cv2.setTrackbarMin(trackbar_f, title_window, 80)
    cv2.setTrackbarMax(trackbar_f, title_window, 140)

    while True:


        x = cv2.getTrackbarPos(trackbar_x, title_window)
        y = cv2.getTrackbarPos(trackbar_y, title_window)
        f = cv2.getTrackbarPos(trackbar_f, title_window)
        img = GetPerspective(original_img, f, x, y, 360, 360)

        cv2.imshow(title_window, img)

        k = cv2.waitKey(1)
        if k == 27:
            break

    cv2.destroyAllWindows()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='запуск просмотрщика цилиндрических проекций')
    parser.add_argument(
        '--file_path',
        type=str,
        required=True,
        help='Путь к файлу'
    )

    args = parser.parse_args()
    file_path = args.file_path

    create_panorama(file_path)