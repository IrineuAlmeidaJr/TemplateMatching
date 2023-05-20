import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime

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


def surf_surf(query_img, train_img):
    inicio = datetime.now()

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # minHessian = 8000
    # surf = cv2.SIFT_create(minHessian)
    surf = cv2.SIFT_create()

    kp1, des1 = surf.detectAndCompute(img1, None)
    kp2, des2 = surf.detectAndCompute(img2, None)

    # --> Faz Correspondência
    # Correspondência
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)

    # Armazena todas as boas correspondências de acordo com o teste de proporção de Lowe's
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference, img_show = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, img_show, tempo_gasto

def orb_orb(query_img, train_img):
    inicio = datetime.now()

    # MIN_MATCH_COUNT = 125

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



    # if len(matches) > MIN_MATCH_COUNT:
    #     # Se verdade, desenhar os inliers
    #     src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
    #     dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
    #     M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    #     matchesMask = mask.ravel().tolist()
    #     h, w = img1.shape
    #     pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)
    #     dst = cv2.perspectiveTransform(pts, M)
    #     img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)
    # else:
    #     print(f'Não foram encontradas correspondências suficientes')
    #     matchesMask = None
    #
    #
    #
    # draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
    #                    singlePointColor=None,
    #                    matchesMask=None,  # draw only inliers
    #                    flags=2)
    #
    #
    #
    # final_img = cv2.drawMatches(img1, kp1,
    #                             img2, kp2, matches, None, **draw_params)
    #
    # cv2.imshow("Matches in ORB", final_img)
    # cv2.waitKey(0)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference, img_show = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, img_show, tempo_gasto

    # https://medium.com/image-stitching-com-opencv-e-python/fazendo-uma-image-stitching-da-paisagem-da-janela-do-seu-quarto-fcf09df55c51

def sift_sift(query_img, train_img):
    inicio = datetime.now()

    # train_img = rotacionar_imagem(train_img,15)

    # Converte para Cinza para trabalhar com um canal apenas
    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # Inicializa o SIFT
    sift = cv2.SIFT_create()

    # Extrai os pontos chaves e descritores
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # --> Desenha Ponto Chaves
    # img1_sift = cv2.drawKeypoints(img1, kp1, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('SIFT img1', img1_sift)
    # img2_sift = cv2.drawKeypoints(img2, kp2, None, flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    # cv2.imshow('SIFT img2', img2_sift)

    # --> Faz Correspondência
    # Correspondência
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    img_matches = cv2.drawMatchesKnn(img1, kp1, img2, kp2, matches, None, flags=2)

    # Armazena todas as boas correspondências de acordo com o teste de proporção de Lowe's
    good_matches  = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference, img_show = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, img_show, tempo_gasto

    # https://docs.opencv.org/3.4/d1/de0/tutorial_py_feature_homography.html

# def remove_areas_verdes(img):
#     # Converte a imagem para o espaço de cor HSV
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#
#     # Define o intervalo de cores da área verde
#     lower_green = np.array([36, 25, 25])
#     upper_green = np.array([70, 255, 255])
#
#     mask = cv2.inRange(hsv, lower_green, upper_green)
#
#     kernel = np.ones((5, 5), np.uint8)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
#     mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
#
#     mask = cv2.bitwise_not(mask)
#
#     img_sem_verde = cv2.bitwise_and(img, img, mask=mask)
#
#     # Exibe o resultado
#     cv2.imshow('img_sem_verde', img_sem_verde)

def ransac(img1, img2, kp1, kp2, good_matches):
    # Deve conter no caso MIN_MATCH_COUNT para considerar que encontrou a imagem
    inliers = 0
    outliers = 0
    img_difference = 0
    MIN_MATCH_COUNT = 20
    if len(good_matches) > MIN_MATCH_COUNT:
        # Se verdade, desenhar os inliers
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Matriz homográfica com RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matchesMask = mask.ravel().tolist()  # inliers

        # Calcula INLIERS e OUTLIERS
        # mask: é um vetor que possui 0 e 1. Sendo que 1 é inlier e 0 outliers
        for i, m in enumerate(mask):
            if m:
                inliers += 1
            else:
                outliers += 1

        # Posiciona uma imagem em relação a outra
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # ------ Transforma: coloca a imagem 1 na imagem 2
        # OBS - essa parte pega a imagem 1 e encaixa na imagem 2,
        # não necessáriamente isso é necessário fazer, pois, o que quero
        # é encaixar a imagem 2 na imagem 1, pois a imagem 2 tem uma área maior que a imagem 1,
        # ai colocando a imagem 2 na imagem 1 faz a subtração.
        # # Obter o tamanho da imagem de fundo da imagem 2
        # h, w = img2.shape
        # # Aplicar a transformação perspectiva na imagem de entrada da imagem 1
        # img_warped = cv2.warpPerspective(img1, M, (w, h))
        # # Sobrepor a imagem transformada na imagem de fundo na imagem 2
        # result = cv2.addWeighted(img_warped, 0.5, img2, 0.5, 0)
        #
        # # cv2.imshow('Transformacao_img1_img2', result)
        # -----------------------------------------------

        # --- EQUALIZAÇÃO: faz equalização de histograma para depois fazer a subtração
        # de uma imagem da outra
        #  TIROU pq estava dando erro, depois que fez a esqualização na função para
        # equalizar a cor
        # img1 = cv2.equalizeHist(img1)
        # img2 = cv2.equalizeHist(img2)

        # --- TRANSFORMAÇÃO: visando encaixar a imagem 2 na imagem 1
        # Invertendo a transformação para subtrair da imagem original
        M_inv = np.linalg.inv(M)  # calcula matriz inversa
        img2_transformed = cv2.warpPerspective(img2, M_inv, (img1.shape[1], img1.shape[0]))
        # Subtraindo a imagem transformada da imagem original
        img_diff = cv2.absdiff(img1, img2_transformed)
        # Exibi a imagem obtida por meio da subtração
        cv2.imshow('Transformacao_img2_img1', img_diff)
        img_difference = np.mean(img_diff)
    else:
        print(f'Não foram encontradas correspondências suficientes - '
              f'{len(good_matches)}/{MIN_MATCH_COUNT}')
        matchesMask = None

    # desenha os inliers
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img3 = cv2.drawMatches(img1, kp1,
                           img2, kp2, good_matches, None, **draw_params)

    return inliers, outliers, img_difference, img3

def histograma(img1, img2):
    # Converte para Cinza para trabalhar com um canal apenas
    img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    img1 = cv2.equalizeHist(img1)
    img2 = cv2.equalizeHist(img2)

    cv2.imshow("Histograma_1", img1)
    cv2.imshow("Histograma_2", img2)


def equaliza_cor(sat, uav):
    # Converter ambas as imagens para o espaço de cor Lab
    sat = cv2.cvtColor(sat, cv2.COLOR_BGR2LAB)
    uav = cv2.cvtColor(uav, cv2.COLOR_BGR2LAB)

    # Calcular a média e o desvio padrão dos canais L, a e b da imagem img1
    media_pixels_sat, desvio_padrao_sat = cv2.meanStdDev(sat)
    media_pixels_sat = media_pixels_sat.ravel()  # achatada em um array 1D, que pode ser mais facilmente manipulado ou processado
    desvio_padrao_sat = desvio_padrao_sat.ravel()  # achatada em um array 1D, que pode ser mais facilmente manipulado ou processado

    # Normalizar os canais da imagem img2 pela média e desvio padrão da imagem img1
    uav[:, :, 0] = np.clip(((uav[:, :, 0] - np.mean(uav[:, :, 0])) * (desvio_padrao_sat[0] / np.std(uav[:, :, 0]))) \
                           + media_pixels_sat[0], 0, 255)
    # esta tendo que normalizar, não entendi muito o porque seguindo o artigo eu não conseguir, perguntar
    # Teve que normalizar
    # img2[:, :, 0] = ((img2[:, :, 0] - np.mean(img2[:, :, 0])) * (desvio_padrao_img1[0] / np.std(img2[:, :, 0]))) \
    #                 + media_pixels_img1[0]
    uav[:, :, 1] = ((uav[:, :, 1] - np.mean(uav[:, :, 1])) * (desvio_padrao_sat[1] / np.std(uav[:, :, 1]))) \
                   + media_pixels_sat[1]
    uav[:, :, 2] = ((uav[:, :, 2] - np.mean(uav[:, :, 2])) * (desvio_padrao_sat[2] / np.std(uav[:, :, 2]))) \
                   + media_pixels_sat[2]

    # Converter de volta para o espaço de cor BGR
    result = cv2.cvtColor(uav, cv2.COLOR_LAB2BGR)

    cv2.imwrite('../../template-matching/app-correspondencia-imagem/public/images/equalize_color_img1.jpg', result)

    return result


def main():
    # query_img = cv2.imread('./data/outros/lena.png')
    # train_img = cv2.imread('./data/outros/lena1.jpg')
    query_img = cv2.imread('./data/curitiba_1/curitiba_1_04-20.png')
    train_img = cv2.imread('./data/curitiba_1/curitiba_1_05-21_angulo.png')
    # query_img = cv2.imread('./data/machado_1/machado_1_3-20.jpg')
    # train_img = cv2.imread('./data/machado_1/machado_1_8-22.png')

    cv2.imshow("img1", query_img)

    train_img = equaliza_cor(query_img, train_img)
    cv2.imshow("train_img equalizada", train_img)

    # histograma(query_img, train_img)

    # orb(query_img, train_img)
    # sift(query_img, train_img)
    # sif_match_ransac(query_img, train_img)

    inliers, outliers, img_difference, img_show, tempo = sift_sift(query_img, train_img)
    # inliers, outliers, img_difference, img_show, tempo = orb_orb(query_img, train_img)
    # inliers, outliers, img_difference, img_show, tempo = surf_surf(query_img, train_img)
    if inliers > 5 and img_difference < 80:
        print("Número de inliers: ", inliers)
        print("Número de outliers: ", outliers)
        print("Subtração (img2-img1): ", img_difference)
        print("Tempo: ", tempo)
    else:
        print("Imagens diferentes")

    cv2.imshow("Matches-img", img_show)

    # remove_areas_verdes(train_img)

    cv2.waitKey(0)



if __name__ == '__main__':
    main()


# https://www.askpython.com/python-modules/feature-matching-in-images-opencv