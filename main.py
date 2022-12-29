import cv2
import numpy as np
from matplotlib import pyplot as plt

def rotacionar_imagem(img, angulo):
    # A forma de uma imagem é acessada por img.shape.
    # Ele retorna uma tupla do número de linhas, colunas e canais (se a imagem for colorida)
    # Exemplo: >>> print( img.shape )
    #          (342, 548, 3)
    height = img.shape[0] # número de linhas
    width = img.shape[1] # número de colunas

    centroY = height/2
    centroX = width/2

    matrizRotacao = cv2.getRotationMatrix2D((centroX, centroY), angulo, 1.0)

    img_rot = cv2.warpAffine(img, matrizRotacao, (width, height))

    return img_rot

def orb(query_img, train_img):
    MIN_MATCH_COUNT = 125

    # train_img = rotacionar_imagem(train_img, 95)

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    print("Rawmatches (correspodência): ", len(matches))

    if len(matches) > MIN_MATCH_COUNT:
        # Se verdade, desenhar os inliers
        src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print(f'Não foram encontradas correspondências suficientes')
        matchesMask = None



    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=None,  # draw only inliers
                       flags=2)



    final_img = cv2.drawMatches(img1, kp1,
                                img2, kp2, matches, None, **draw_params)

    cv2.imshow("Matches", final_img)
    cv2.waitKey(0)

    # https://medium.com/image-stitching-com-opencv-e-python/fazendo-uma-image-stitching-da-paisagem-da-janela-do-seu-quarto-fcf09df55c51

def sift(query_img, train_img):
    MIN_MATCH_COUNT = 30

    # train_img = rotacionar_imagem(train_img,95)

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    sift = cv2.SIFT_create()

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    print("Rawmatches  (correspodência): ", len(matches))

    # Armazena todas as boas correspondências de acordo com o teste de proporção de Lowe's
    good = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good.append(m)

    print("Good: ", len(good))

    # Deve conter no caso MIN_MATCH_COUNT para considerar que encontrou a imagem
    if len(good) > MIN_MATCH_COUNT:
        # Se verdade, desenhar os inliers
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    else:
        print(f'Não foram encontradas correspondências suficientes - '
              f'{len(good)}/{MIN_MATCH_COUNT}')
        matchesMask = None


    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1,
                           img2, kp2, good, None, **draw_params)

    cv2.imshow("Matches", img3)
    cv2.waitKey(0)

    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

def main():
    query_img = cv2.imread('antes.png')
    train_img = cv2.imread('antes-3.png')

    # query_img = cv2.imread('casa.png')
    # train_img = cv2.imread('geral.png')

    # query_img = cv2.imread('lena.png')
    # train_img = cv2.imread('lena3.jpg')

    orb(query_img, train_img)
    # sift(query_img, train_img)




if __name__ == '__main__':
    main()


# https://www.askpython.com/python-modules/feature-matching-in-images-opencv