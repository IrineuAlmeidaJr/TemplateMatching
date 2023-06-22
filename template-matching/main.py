import os
import random
import cv2
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime, timedelta


def rotacionar_imagem(img, angulo):
    height = img.shape[0]  # número de linhas
    width = img.shape[1]  # número de colunas

    centroY = height / 2
    centroX = width / 2

    matrizRotacao = cv2.getRotationMatrix2D((centroX, centroY), angulo, 1.0)

    img_rot = cv2.warpAffine(img, matrizRotacao, (width, height))

    return img_rot


def fast_brief(query_img, train_img):
    inicio = datetime.now()

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()

    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)

    # Extrai os descritores usando o BRIEF
    brief = cv2.xfeatures2d.BriefDescriptorExtractor_create()
    kp1, des1 = brief.compute(img1, kp1)
    kp2, des2 = brief.compute(img2, kp2)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference = ransac(img1, img2, kp1, kp2, matches)

    return inliers, outliers, img_difference, tempo_gasto


def fast_sift(query_img, train_img):
    inicio = datetime.now()

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    fast = cv2.FastFeatureDetector_create()

    kp1 = fast.detect(img1, None)
    kp2 = fast.detect(img2, None)

    # Extrai os pontos chaves e descritores
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # --> Faz Correspondência
    # Correspondência
    FLANN_INDEX_KDTREE = 0
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)

    flann = cv2.FlannBasedMatcher(index_params, search_params)

    matches = flann.knnMatch(des1, des2, k=2)

    # Armazena todas as boas correspondências de acordo com o teste de proporção de Lowe's
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, tempo_gasto


def akaze_akaze(query_img, train_img):
    inicio = datetime.now()

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference = ransac(img1, img2, kp1, kp2, matches)

    return inliers, outliers, img_difference, tempo_gasto


def orb_orb(query_img, train_img):
    inicio = datetime.now()

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create()

    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference = ransac(img1, img2, kp1, kp2, matches)

    return inliers, outliers, img_difference, tempo_gasto

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

    # Armazena todas as boas correspondências de acordo com o teste de proporção de Lowe's
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, tempo_gasto


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

        # --- TRANSFORMAÇÃO: visando encaixar a imagem 2 na imagem 1
        # Invertendo a transformação para subtrair da imagem original
        M_inv = np.linalg.inv(M)  # calcula matriz inversa
        img2_transformed = cv2.warpPerspective(img2, M_inv, (img1.shape[1], img1.shape[0]))
        # Subtraindo a imagem transformada da imagem original
        img_diff = cv2.absdiff(img1, img2_transformed)
        # Exibi a imagem obtida por meio da subtração
        img_difference = np.mean(img_diff)
    else:
        matchesMask = None

    return inliers, outliers, img_difference


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


def calcula_exibe_histograma(img, titulo):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title(titulo)
    plt.xlabel('Intensidade de pixel')
    plt.ylabel('Contagem')
    plt.show()


# def main():
#     # query_img = cv2.imread('./data/outros/lena.png')
#     # train_img = cv2.imread('./data/outros/lena1.jpg')
#     # query_img = cv2.imread('./data/curitiba_1/curitiba_1_04-20.png')
#     # train_img = cv2.imread('./data/curitiba_1/curitiba_1_05-21_angulo.png')
#     # query_img = cv2.imread('./data/curitiba_2/curitiba_2_04-20.jpg')
#     # train_img = cv2.imread('./data/curitiba_2/curitiba_2_05-21.jpg')
#     # query_img = cv2.imread('./data/machado_1/machado_1_3-20.jpg')
#     # train_img = cv2.imread('./data/machado_1/machado_1_8-22.png')
#
#     # query_img = cv2.imread('./data/curitiba_2/curitiba_2_04-20.jpg')
#     # train_img = cv2.imread('./data/curitiba_2/curitiba_2_05-21.jpg')
#
#     # # -> Drone
#     # query_img = cv2.imread('./database/alvares_machado/06-22/alvares_machado_08.jpg')
#     # # -> Satélite
#     # train_img = cv2.imread('./database/alvares_machado/04-23/alvares_machado_08.jpg')
#
#     query_img = cv2.imread('./database/curitiba/04-20/curitiba_100.jpg')
#     train_img = cv2.imread('./database/curitiba/05-21/curitiba_100.jpg')
#
#     # ----> IMAGEM ORIGINAIS
#     # cv2.imshow("satelite", query_img)
#     # cv2.imshow("drone - original", train_img)
#     cv2.imshow("drone - original", query_img)
#     cv2.imshow("satelite", train_img)
#
#     # ----> IMAGEM EQUALIZADA
#     # train_img = equaliza_cor(query_img, train_img)
#     query_img = equaliza_cor(train_img, query_img)
#     cv2.imshow("drone - equalizada", query_img)
#
#     # histograma(query_img, train_img)
#
#     # orb(query_img, train_img)
#     # sift(query_img, train_img)
#     # sif_match_ransac(query_img, train_img)
#
#     # inliers, outliers, img_difference, img_show, tempo = sift_sift(query_img, train_img)
#     # inliers, outliers, img_difference, img_show, tempo = orb_orb(query_img, train_img)
#     inliers, outliers, img_difference, img_show, tempo = akaze_akaze(query_img, train_img)
#     # inliers, outliers, img_difference, img_show, tempo = fast_brief(query_img, train_img)
#     # inliers, outliers, img_difference, img_show, tempo = fast_sift(query_img, train_img)
#
#     print("Número de inliers: ", inliers)
#     print("Número de outliers: ", outliers)
#     print("Subtração (img2-img1): ", img_difference)
#     print("Tempo: ", tempo)
#
#     # ----> IMAGEM COM CORRESPONDÊNCIA
#     cv2.imshow("Matches-img", img_show)
#
#     # remove_areas_verdes(train_img)
#
#     cv2.waitKey(0)


def criar_arquivo():
    caminho = 'resultado/'
    arquivo = caminho + '/dados.csv'

    with open(arquivo, "w") as arquivo:
        cabecalho = "Imagem;SIFT_SIFT_Inliers;SIFT_SIFT_Outliers;SIFT_SIFT_Diferenca;SIFT_SIFT_Tempo;" \
                    "SIFT_SIFT_45_Inliers;SIFT_SIFT_45_Outliers;SIFT_SIFT_45_Diferenca;SIFT_SIFT_45_Tempo;" \
                    "ORB_ORB_Inliers;ORB_ORB_Outliers;ORB_ORB_Diferenca;ORB_ORB_Tempo;" \
                    "ORB_ORB_45_Inliers;ORB_ORB_45_Outliers;ORB_ORB_45_Diferenca;ORB_ORB_45_Tempo;" \
                    "AKAZE_AKAZE_Inliers;AKAZE_AKAZE_Outliers;AKAZE_AKAZE_Diferenca;AKAZE_AKAZE_Tempo;" \
                    "AKAZE_AKAZE_45_Inliers;AKAZE_AKAZE_45_Outliers;AKAZE_AKAZE_45_Diferenca;AKAZE_AKAZE_45_Tempo;" \
                    "FAST_BRIEF_Inliers;FAST_BRIEF_Outliers;FAST_BRIEF_Diferenca;FAST_BRIEF_Tempo;" \
                    "FAST_BRIEF_45_Inliers;FAST_BRIEF_45_Outliers;FAST_BRIEF_45_Diferenca;FAST_BRIEF_45_Tempo;" \
                    "FAST_SIFT_Inliers;FAST_SIFT_Outliers;FAST_SIFT_Diferenca;FAST_SIFT_Tempo;" \
                    "FAST_SIFT_45_Inliers;FAST_SIFT_45_Outliers;FAST_SIFT_45_Diferenca;FAST_SIFT_45_Tempo;"
        arquivo.write(cabecalho)


def escrever_linha_arquivo(linha):
    caminho = 'resultado/'
    arquivo = caminho + '/dados.csv'

    with open(arquivo, "a") as arquivo:
        arquivo.write(linha)


def converter_tempo(tempo):
    # tempo_em_segundos = tempo.total_seconds()
    return tempo


def main():
    criar_arquivo()
    num_arq = 1
    while num_arq <= 100:
        img_anterior = cv2.imread(f'./database/curitiba/04-20/curitiba_{num_arq}.jpg')
        img_atual = cv2.imread(f'./database/curitiba/05-21/curitiba_{num_arq}.jpg')
        # img_anterior = cv2.imread(f'./database/alvares_machado/06-22/alvares_machado_{num_arq}.jpg')
        # img_atual = cv2.imread(f'./database/alvares_machado/04-23/alvares_machado_{num_arq}.jpg')

        # -> EQUALIZADA IMAGEM
        # img_anterior = equaliza_cor(img_atual, img_anterior)

        # -> ROTACIONAR IMAGEM
        img_anterior_45 = rotacionar_imagem(img_anterior, 45)
        img_atual_45 = rotacionar_imagem(img_atual, 45)

        # ---------------------- ALGORITMOS ----------------------
        # -> SIFT_SIFT
        SIFT_SIFT_Inliers, SIFT_SIFT_Outliers, SIFT_SIFT_Diferenca, SIFT_SIFT_Tempo = \
            sift_sift(img_anterior, img_atual)
        # -> SIFT_SIFT_45
        SIFT_SIFT_45_Inliers, SIFT_SIFT_45_Outliers, SIFT_SIFT_45_Diferenca, SIFT_SIFT_45_Tempo = \
            sift_sift(img_anterior_45, img_atual_45)
        # ----------------------
        # -> ORB_ORB
        ORB_ORB_Inliers, ORB_ORB_Outliers, ORB_ORB_Diferenca, ORB_ORB_Tempo = \
            orb_orb(img_anterior, img_atual)
        # -> ORB_ORB_45
        ORB_ORB_45_Inliers, ORB_ORB_45_Outliers, ORB_ORB_45_Diferenca, ORB_ORB_45_Tempo = \
            orb_orb(img_anterior_45, img_atual_45)
        # ----------------------
        # -> AKAZE_AKAZE
        AKAZE_AKAZE_Inliers, AKAZE_AKAZE_Outliers, AKAZE_AKAZE_Diferenca, AKAZE_AKAZE_Tempo = \
            akaze_akaze(img_anterior, img_atual)
        # -> AKAZE_AKAZE_45
        AKAZE_AKAZE_45_Inliers, AKAZE_AKAZE_45_Outliers, AKAZE_AKAZE_45_Diferenca, AKAZE_AKAZE_45_Tempo = \
            akaze_akaze(img_anterior_45, img_atual_45)
        # ----------------------
        # -> FAST_BRIEF
        FAST_BRIEF_Inliers, FAST_BRIEF_Outliers, FAST_BRIEF_Diferenca, FAST_BRIEF_Tempo = \
            fast_brief(img_anterior, img_atual)
        # -> FAST_BRIEF_45
        FAST_BRIEF_45_Inliers, FAST_BRIEF_45_Outliers, FAST_BRIEF_45_Diferenca, FAST_BRIEF_45_Tempo = \
            fast_brief(img_anterior_45, img_atual_45)
        # ----------------------
        # -> FAST_SIFT
        FAST_SIFT_Inliers, FAST_SIFT_Outliers, FAST_SIFT_Diferenca, FAST_SIFT_Tempo = \
            fast_sift(img_anterior, img_atual)
        # -> FAST_SIFT_45
        FAST_SIFT_45_Inliers, FAST_SIFT_45_Outliers, FAST_SIFT_45_Diferenca, FAST_SIFT_45_Tempo = \
            fast_sift(img_anterior_45, img_atual_45)


        escrever_linha_arquivo(f"\n{num_arq};{SIFT_SIFT_Inliers};{SIFT_SIFT_Outliers};{SIFT_SIFT_Diferenca};{converter_tempo(SIFT_SIFT_Tempo)};"
                               f"{SIFT_SIFT_45_Inliers};{SIFT_SIFT_45_Outliers};{SIFT_SIFT_45_Diferenca};{converter_tempo(SIFT_SIFT_45_Tempo)};"
                               f"{ORB_ORB_Inliers};{ORB_ORB_Outliers};{ORB_ORB_Diferenca};{converter_tempo(ORB_ORB_Tempo)};"
                               f"{ORB_ORB_45_Inliers};{ORB_ORB_45_Outliers};{ORB_ORB_45_Diferenca};{converter_tempo(ORB_ORB_45_Tempo)};"
                               f"{AKAZE_AKAZE_Inliers};{AKAZE_AKAZE_Outliers};{AKAZE_AKAZE_Diferenca};{converter_tempo(AKAZE_AKAZE_Tempo)};"
                               f"{AKAZE_AKAZE_45_Inliers};{AKAZE_AKAZE_45_Outliers};{AKAZE_AKAZE_45_Diferenca};{converter_tempo(AKAZE_AKAZE_45_Tempo)};"
                               f"{FAST_BRIEF_Inliers};{FAST_BRIEF_Outliers};{FAST_BRIEF_Diferenca};{converter_tempo(FAST_BRIEF_Tempo)};"
                               f"{FAST_BRIEF_45_Inliers};{FAST_BRIEF_45_Outliers};{FAST_BRIEF_45_Diferenca};{converter_tempo(FAST_BRIEF_45_Tempo)};"
                               f"{FAST_SIFT_Inliers};{FAST_SIFT_Outliers};{FAST_SIFT_Diferenca};{converter_tempo(FAST_SIFT_Tempo)};"
                               f"{FAST_SIFT_45_Inliers};{FAST_SIFT_45_Outliers};{FAST_SIFT_45_Diferenca};{converter_tempo(FAST_SIFT_45_Tempo)};")

        print(f"Imagem-{num_arq}")

        num_arq += 1




if __name__ == '__main__':
    main()
