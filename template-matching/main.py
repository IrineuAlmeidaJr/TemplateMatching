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

def distancia_euclidiana(ponto1, ponto2):
    x1, y1 = ponto1
    x2, y2 = ponto2
    return np.sqrt((x2 - x1)**2 + (y2 - y1)**2)


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

    # Definir um limiar de distância para filtrar correspondências ruins
    threshold_distance = 20

    # Filtrar correspondências com base no limiar de distância
    good_matches = []
    for match in matches:
        if match.distance < threshold_distance:
            good_matches.append(match)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_difference, distancia_pontos = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, tempo_gasto, distancia_pontos


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

    inliers, outliers, img_difference, distancia_pontos = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, tempo_gasto, distancia_pontos


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

    inliers, outliers, img_difference, distancia_pontos = ransac(img1, img2, kp1, kp2, matches)

    return inliers, outliers, img_difference, tempo_gasto, distancia_pontos


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

    inliers, outliers, img_difference, distancia_pontos = ransac(img1, img2, kp1, kp2, matches)

    return inliers, outliers, img_difference, tempo_gasto, distancia_pontos

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

    inliers, outliers, img_difference, distancia_pontos = ransac(img1, img2, kp1, kp2, good_matches)

    return inliers, outliers, img_difference, tempo_gasto, distancia_pontos


def ransac(img1, img2, kp1, kp2, good_matches):
    # Deve conter no caso MIN_MATCH_COUNT para considerar que encontrou a imagem
    distancia_pontos = np.array([], float)
    inliers = 0
    outliers = 0
    img_diff = 0
    img_difference = 0
    MIN_MATCH_COUNT = 20
    if len(good_matches) > MIN_MATCH_COUNT:
        # Se verdade, desenhar os inliers
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Matriz homográfica com RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()  # inliers

        # Calcula INLIERS e OUTLIERS
        # mask: é um vetor que possui 0 e 1. Sendo que 1 é inlier e 0 outliers
        for m in matches_mask:
            if m:
                inliers += 1
            else:
                outliers += 1

        # Posiciona uma imagem em relação a outra
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # ----> Aqui faço uma verificação se o ponto esta dentro do poligono desenhado na linha de cima
        #
        # Armazena pontos para poder desenhar e calcular depois a distância entres os pontos da
        # imagem um de da imagem dois
        pontos_img1 = []
        pontos_img2 = []
        for i in range(len(matches_mask)):
            if matches_mask[i]:
                pnt_ori = np.array([src_pts[i][0][0], src_pts[i][0][1], 1.0])
                # Aplica a matriz homografia para projetar o ponto
                pnt_tr = np.dot(M, pnt_ori)
                # Normalizar a coordenada homogênea dividindo pelo terceiro componente - parte pesquisada
                dst_trs = pnt_tr[:2] / pnt_tr[2]

                # Verifica se o ponto transformado está dentro da imagem onde teve a correspondência
                ponto_transformado = tuple(map(int, dst_trs))
                distancia = cv2.pointPolygonTest(np.int32(dst), ponto_transformado, True)
                if distancia > 0:
                    pontos_img1.append((int(dst_trs[0]), int(dst_trs[1])))
                    pontos_img2.append((int(dst_pts[i][0][0]), int(dst_pts[i][0][1])))

        for ponto in pontos_img1:
            cv2.circle(img2, ponto, 20, (0, 0, 0), 4)

        for ponto in pontos_img2:
            cv2.circle(img2, ponto, 20, (255, 255, 255), 4)

        # Calcula as distância entre os pontos
        # Verifica se o número de pontos encontrados com correspondência é o mesmo tanto de inliers encontrados
        if inliers == len(pontos_img1):
            for i in range(len(pontos_img1)):
                distancia = distancia_euclidiana(pontos_img1[i], pontos_img2[i])
                distancia_pontos = np.append(distancia_pontos, distancia)

        ## ----- FIM Calculo Distância dos Pontos E se está dentro do Poligono

        # --- TRANSFORMAÇÃO: visando encaixar a imagem 2 na imagem 1
        # Invertendo a transformação para subtrair da imagem original
        M_inv = np.linalg.inv(M)  # calcula matriz inversa
        img2_transformed = cv2.warpPerspective(img2, M_inv, (img1.shape[1], img1.shape[0]))
        # Subtraindo a imagem transformada da imagem original
        img_diff = cv2.absdiff(img1, img2_transformed)
        # Exibi a imagem obtida por meio da subtração
        img_difference = np.mean(img_diff)
    else:
        matches_mask = None

    return inliers, outliers, img_diff, (-1 if distancia_pontos.size == 0 else np.mean(distancia_pontos))


def equaliza_cor(sat, uav):
    # Converter ambas as imagens para o espaço de cor Lab
    img1 = cv2.cvtColor(sat, cv2.COLOR_BGR2LAB)
    img2 = cv2.cvtColor(uav, cv2.COLOR_BGR2LAB)

    # Calcular a média e o desvio padrão dos canais L, a e b da imagem img1
    media_pixels_img1, desvio_padrao_img1 = cv2.meanStdDev(img1)
    media_pixels_img1 = media_pixels_img1.ravel()  # achatada em um array 1D, que pode ser mais facilmente manipulado ou processado
    desvio_padrao_img1 = desvio_padrao_img1.ravel()  # achatada em um array 1D, que pode ser mais facilmente manipulado ou processado

    # Normalizar os canais da imagem img2 pela média e desvio padrão da imagem img1
    img2[:, :, 0] = np.clip(
        (((img2[:, :, 0] - np.mean(img2[:, :, 0])) * (desvio_padrao_img1[0] / np.std(img2[:, :, 0]))) +
         media_pixels_img1[0]), 0, 255).astype(
        np.uint8)
    img2[:, :, 1] = ((img2[:, :, 1] - np.mean(img2[:, :, 1])) * (desvio_padrao_img1[1] / np.std(img2[:, :, 1]))) + \
                    media_pixels_img1[1]
    img2[:, :, 2] = ((img2[:, :, 2] - np.mean(img2[:, :, 2])) * (desvio_padrao_img1[2] / np.std(img2[:, :, 2]))) + \
                    media_pixels_img1[2]

    # Converter de volta para o espaço de cor BGR
    result = cv2.cvtColor(img2, cv2.COLOR_LAB2BGR)

    cv2.imwrite('../../template-matching/app-correspondencia-imagem/public/images/equalize_color_img1.jpg', result)

    return result

def calcula_exibe_histograma(img, titulo):
    hist = cv2.calcHist([img], [0], None, [256], [0, 256])
    plt.plot(hist)
    plt.title(titulo)
    plt.xlabel('Intensidade de pixel')
    plt.ylabel('Contagem')
    plt.show()

def criar_arquivo():
    caminho = 'resultado/'
    arquivo = caminho + '/dados.csv'

    with open(arquivo, "w") as arquivo:
        cabecalho = "Imagem;SIFT_SIFT_Inliers;SIFT_SIFT_Outliers;SIFT_SIFT_Distancia;SIFT_SIFT_Tempo;" \
                    "SIFT_SIFT_45_Inliers;SIFT_SIFT_45_Outliers;SIFT_SIFT_45_Distancia;SIFT_SIFT_45_Tempo;" \
                    "ORB_ORB_Inliers;ORB_ORB_Outliers;ORB_ORB_Distancia;ORB_ORB_Tempo;" \
                    "ORB_ORB_45_Inliers;ORB_ORB_45_Outliers;ORB_ORB_45_Distancia;ORB_ORB_45_Tempo;" \
                    "AKAZE_AKAZE_Inliers;AKAZE_AKAZE_Outliers;AKAZE_AKAZE_Distancia;AKAZE_AKAZE_Tempo;" \
                    "AKAZE_AKAZE_45_Inliers;AKAZE_AKAZE_45_Outliers;AKAZE_AKAZE_45_Distancia;AKAZE_AKAZE_45_Tempo;" \
                    "FAST_BRIEF_Inliers;FAST_BRIEF_Outliers;FAST_BRIEF_Distancia;FAST_BRIEF_Tempo;" \
                    "FAST_BRIEF_45_Inliers;FAST_BRIEF_45_Outliers;FAST_BRIEF_45_Distancia;FAST_BRIEF_45_Tempo;" \
                    "FAST_SIFT_Inliers;FAST_SIFT_Outliers;FAST_SIFT_Distancia;FAST_SIFT_Tempo;" \
                    "FAST_SIFT_45_Inliers;FAST_SIFT_45_Outliers;FAST_SIFT_45_Distancia;FAST_SIFT_45_Tempo;"
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
    num_arq = 101
    while num_arq <= 129:
        img_anterior = cv2.imread(f'./database/diferente/antes/diferente_{num_arq}.jpg')
        img_atual = cv2.imread(f'./database/diferente/depois/diferente_{num_arq}.jpg')
        # img_anterior = cv2.imread(f'./database/curitiba/04-20/curitiba_{num_arq}.jpg')
        # img_atual = cv2.imread(f'./database/curitiba/05-21/curitiba_{num_arq}.jpg')
        # img_anterior = cv2.imread(f'./database/alvares_machado/03-20/alvares_machado_{num_arq}.jpg')
        # img_atual = cv2.imread(f'./database/alvares_machado/06-22/alvares_machado_{num_arq}.jpg')

        # -> EQUALIZADA IMAGEM
        img_atual = equaliza_cor(img_anterior, img_atual)

        # -> ROTACIONAR IMAGEM
        img_anterior_45 = rotacionar_imagem(img_anterior, 45)

        # ---------------------- ALGORITMOS ----------------------
        # -> SIFT_SIFT
        print(f"Imagem-{num_arq} - SIFT_SIFT")
        SIFT_SIFT_Inliers, SIFT_SIFT_Outliers, SIFT_SIFT_Diferenca, SIFT_SIFT_Tempo, SIFT_SIFT_Distancia  = \
            sift_sift(img_anterior, img_atual)
        # -> SIFT_SIFT_45
        print(f"Imagem-{num_arq} - SIFT_SIFT_45")
        SIFT_SIFT_45_Inliers, SIFT_SIFT_45_Outliers, SIFT_SIFT_45_Diferenca, SIFT_SIFT_45_Tempo, SIFT_SIFT_45_Distancia = \
            sift_sift(img_anterior_45, img_atual)
        # ----------------------
        # -> ORB_ORB
        print(f"Imagem-{num_arq} - ORB_ORB")
        ORB_ORB_Inliers, ORB_ORB_Outliers, ORB_ORB_Diferenca, ORB_ORB_Tempo, ORB_ORB_Distancia = \
            orb_orb(img_anterior, img_atual)
        # -> ORB_ORB_45
        print(f"Imagem-{num_arq} - ORB_ORB_45")
        ORB_ORB_45_Inliers, ORB_ORB_45_Outliers, ORB_ORB_45_Diferenca, ORB_ORB_45_Tempo, ORB_ORB_45_Distancia = \
            orb_orb(img_anterior_45, img_atual)
        # ----------------------
        # -> AKAZE_AKAZE
        print(f"Imagem-{num_arq} - AKAZE_AKAZE")
        AKAZE_AKAZE_Inliers, AKAZE_AKAZE_Outliers, AKAZE_AKAZE_Diferenca, AKAZE_AKAZE_Tempo, AKAZE_AKAZE_Distancia = \
            akaze_akaze(img_anterior, img_atual)
        # -> AKAZE_AKAZE_45
        print(f"Imagem-{num_arq} - AKAZE_AKAZE_45")
        AKAZE_AKAZE_45_Inliers, AKAZE_AKAZE_45_Outliers, AKAZE_AKAZE_45_Diferenca, AKAZE_AKAZE_45_Tempo, AKAZE_AKAZE_45_Distancia = \
            akaze_akaze(img_anterior_45, img_atual)
        # ----------------------
        # -> FAST_BRIEF
        print(f"Imagem-{num_arq} - FAST_BRIEF")
        FAST_BRIEF_Inliers, FAST_BRIEF_Outliers, FAST_BRIEF_Diferenca, FAST_BRIEF_Tempo, FAST_BRIEF_Distancia = \
            fast_brief(img_anterior, img_atual)
        # -> FAST_BRIEF_45
        print(f"Imagem-{num_arq} - FAST_BRIEF_45")
        FAST_BRIEF_45_Inliers, FAST_BRIEF_45_Outliers, FAST_BRIEF_45_Diferenca, FAST_BRIEF_45_Tempo, FAST_BRIEF_45_Distancia = \
            fast_brief(img_anterior_45, img_atual)
        # ----------------------
        # -> FAST_SIFT
        print(f"Imagem-{num_arq} - FAST_SIFT")
        FAST_SIFT_Inliers, FAST_SIFT_Outliers, FAST_SIFT_Diferenca, FAST_SIFT_Tempo, FAST_SIFT_Distancia = \
            fast_sift(img_anterior, img_atual)
        # -> FAST_SIFT_45
        print(f"Imagem-{num_arq} - FAST_SIFT_45")
        FAST_SIFT_45_Inliers, FAST_SIFT_45_Outliers, FAST_SIFT_45_Diferenca, FAST_SIFT_45_Tempo, FAST_SIFT_45_Distancia = \
            fast_sift(img_anterior_45, img_atual)


        escrever_linha_arquivo(f"\n{num_arq};{SIFT_SIFT_Inliers};{SIFT_SIFT_Outliers};{SIFT_SIFT_Distancia};{converter_tempo(SIFT_SIFT_Tempo)};"
                               f"{SIFT_SIFT_45_Inliers};{SIFT_SIFT_45_Outliers};{SIFT_SIFT_45_Distancia};{converter_tempo(SIFT_SIFT_45_Tempo)};"
                               f"{ORB_ORB_Inliers};{ORB_ORB_Outliers};{ORB_ORB_Distancia};{converter_tempo(ORB_ORB_Tempo)};"
                               f"{ORB_ORB_45_Inliers};{ORB_ORB_45_Outliers};{ORB_ORB_45_Distancia};{converter_tempo(ORB_ORB_45_Tempo)};"
                               f"{AKAZE_AKAZE_Inliers};{AKAZE_AKAZE_Outliers};{AKAZE_AKAZE_Distancia};{converter_tempo(AKAZE_AKAZE_Tempo)};"
                               f"{AKAZE_AKAZE_45_Inliers};{AKAZE_AKAZE_45_Outliers};{AKAZE_AKAZE_45_Distancia};{converter_tempo(AKAZE_AKAZE_45_Tempo)};"
                               f"{FAST_BRIEF_Inliers};{FAST_BRIEF_Outliers};{FAST_BRIEF_Distancia};{converter_tempo(FAST_BRIEF_Tempo)};"
                               f"{FAST_BRIEF_45_Inliers};{FAST_BRIEF_45_Outliers};{FAST_BRIEF_45_Distancia};{converter_tempo(FAST_BRIEF_45_Tempo)};"
                               f"{FAST_SIFT_Inliers};{FAST_SIFT_Outliers};{FAST_SIFT_Distancia};{converter_tempo(FAST_SIFT_Tempo)};"
                               f"{FAST_SIFT_45_Inliers};{FAST_SIFT_45_Outliers};{FAST_SIFT_45_Distancia};{converter_tempo(FAST_SIFT_45_Tempo)};")



        num_arq += 1


def ransac2(img1, img2, kp1, kp2, good_matches):
    # Deve conter no caso MIN_MATCH_COUNT para considerar que encontrou a imagem
    distancia_pontos = np.array([], float)
    inliers = 0
    outliers = 0
    img_diff = 0
    img_difference = 0
    MIN_MATCH_COUNT = 20
    if len(good_matches) > MIN_MATCH_COUNT:
        # Se verdade, desenhar os inliers
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        # Matriz homográfica com RANSAC
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        matches_mask = mask.ravel().tolist()  # inliers

        # Calcula INLIERS e OUTLIERS
        # mask: é um vetor que possui 0 e 1. Sendo que 1 é inlier e 0 outliers
        for m in matches_mask:
            if m:
                inliers += 1
            else:
                outliers += 1

        # Posiciona uma imagem em relação a outra
        h, w = img1.shape
        pts = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

        dst = cv2.perspectiveTransform(pts, M)
        img2 = cv2.polylines(img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA)

        # ----> Aqui faço uma verificação se o ponto esta dentro do poligono desenhado na linha de cima
        #
        # Armazena pontos para poder desenhar e calcular depois a distância entres os pontos da
        # imagem um de da imagem dois
        pontos_img1 = []
        pontos_img2 = []
        for i in range(len(matches_mask)):
            if matches_mask[i]:
                pnt_ori = np.array([src_pts[i][0][0], src_pts[i][0][1], 1.0])
                # Aplica a matriz homografia para projetar o ponto
                pnt_tr = np.dot(M, pnt_ori)
                # Normalizar a coordenada homogênea dividindo pelo terceiro componente - parte pesquisada
                dst_trs = pnt_tr[:2] / pnt_tr[2]

                # Verifica se o ponto transformado está dentro da imagem onde teve a correspondência
                ponto_transformado = tuple(map(int, dst_trs))
                distancia = cv2.pointPolygonTest(np.int32(dst), ponto_transformado, True)
                if distancia > 0:
                    pontos_img1.append((int(dst_trs[0]), int(dst_trs[1])))
                    pontos_img2.append((int(dst_pts[i][0][0]), int(dst_pts[i][0][1])))

        for ponto in pontos_img1:
            cv2.circle(img2, ponto, 20, (0, 0, 0), 4)

        for ponto in pontos_img2:
            cv2.circle(img2, ponto, 20, (255, 255, 255), 4)

        # Calcula as distância entre os pontos
        # Verifica se o número de pontos encontrados com correspondência é o mesmo tanto de inliers encontrados
        if inliers == len(pontos_img1):
            for i in range(len(pontos_img1)):
                distancia = distancia_euclidiana(pontos_img1[i], pontos_img2[i])
                distancia_pontos = np.append(distancia_pontos, distancia)

        ## ----- FIM Calculo Distância dos Pontos E se está dentro do Poligono

        # --- TRANSFORMAÇÃO: visando encaixar a imagem 2 na imagem 1
        # Invertendo a transformação para subtrair da imagem original
        M_inv = np.linalg.inv(M)  # calcula matriz inversa
        img2_transformed = cv2.warpPerspective(img2, M_inv, (img1.shape[1], img1.shape[0]))
        # Subtraindo a imagem transformada da imagem original
        img_diff = cv2.absdiff(img1, img2_transformed)
        # Exibi a imagem obtida por meio da subtração
        img_difference = np.mean(img_diff)
    else:
        matches_mask = None

    return inliers, outliers, img_diff, img_difference, matches_mask, (-1 if distancia_pontos.size == 0 else np.mean(distancia_pontos))

def akaze_akaze_2(query_img, train_img):
    inicio = datetime.now()

    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    akaze = cv2.AKAZE_create()

    kp1, des1 = akaze.detectAndCompute(img1, None)
    kp2, des2 = akaze.detectAndCompute(img2, None)

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(des1, des2)
    matches = sorted(matches, key=lambda x: x.distance)
    good_matches = matches

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_diferenca, diferenca, matchesMask, distancia_pontos = ransac2(img1, img2, kp1, kp2,
                                                                                         good_matches)

    # desenha os inliers
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img_correspondencia = cv2.drawMatches(img1, kp1,
                                          img2, kp2, good_matches, None, **draw_params)

    return inliers, outliers, diferenca, tempo_gasto, img_correspondencia, img_diferenca, distancia_pontos


def fast_sift_2(query_img, train_img):
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

    inliers, outliers, img_diferenca, diferenca, matchesMask, distancia_pontos = ransac2(img1, img2, kp1, kp2, good_matches)

    # desenha os inliers
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img_correspondencia = cv2.drawMatches(img1, kp1,
                                          img2, kp2, good_matches, None, **draw_params)

    return inliers, outliers, diferenca, tempo_gasto, img_correspondencia, img_diferenca, distancia_pontos

def fast_brief_2(query_img, train_img):
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

    # Definir um limiar de distância para filtrar correspondências ruins
    threshold_distance = 20

    # Filtrar correspondências com base no limiar de distância
    good_matches = []
    for match in matches:
        if match.distance < threshold_distance:
            good_matches.append(match)

    fim = datetime.now()

    tempo_gasto = fim - inicio

    inliers, outliers, img_diferenca, diferenca, matchesMask, distancia_pontos = ransac2(img1, img2, kp1, kp2, good_matches)

    # desenha os inliers
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img_correspondencia = cv2.drawMatches(img1, kp1,
                                          img2, kp2, good_matches, None, **draw_params)

    return inliers, outliers, diferenca, tempo_gasto, img_correspondencia, img_diferenca, distancia_pontos

def sift_sift_2(query_img, train_img):
    inicio = datetime.now()
    # Converte para Cinza para trabalhar com um canal apenas
    img1 = cv2.cvtColor(query_img, cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(train_img, cv2.COLOR_BGR2GRAY)

    # cv2.imshow('Imagem1', img1)
    # cv2.imshow('Imagem2', img2)


    # Inicializa o SIFT
    sift = cv2.SIFT_create()

    # Extrai os pontos chaves e descritores
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

    inliers, outliers, img_diferenca, diferenca, matchesMask, distancia_pontos = ransac2(img1, img2, kp1, kp2, good_matches)

    # desenha os inliers
    draw_params = dict(matchColor=(0, 255, 0),  # draw matches in green color
                       singlePointColor=None,
                       matchesMask=matchesMask,  # draw only inliers
                       flags=2)

    img_correspondencia = cv2.drawMatches(img1, kp1,
                           img2, kp2, good_matches, None, **draw_params)

    return inliers, outliers, diferenca, tempo_gasto, img_correspondencia, img_diferenca, distancia_pontos


# def main():
#     img_anterior = cv2.imread('./database/alvares_machado/03-20/alvares_machado_0.jpg')
#     img_atual = cv2.imread('./database/alvares_machado/06-22/alvares_machado_0.jpg')
#     img_atual = equaliza_cor(img_anterior, img_atual)
#     inliers, outliers, diferenca, tempo, img_correspondencia, img_diferenca, distancia_pontos = sift_sift_2(
#         img_anterior, img_atual)
#     print(f'inliers = {inliers} \noutliers = {outliers}\ndistancia pontos = {distancia_pontos}\nTempo: {tempo}')
#     img_anterior = rotacionar_imagem(img_anterior, 45)
#
#     # inliers, outliers, diferenca, tempo, img_correspondencia, img_diferenca = sift_sift(img_anterior, img_atual)
#
#
#     inliers, outliers, diferenca, tempo, img_correspondencia, img_diferenca, distancia_pontos = sift_sift_2(img_anterior, img_atual)
#     print(f'inliers = {inliers} \noutliers = {outliers}\ndistancia pontos = {distancia_pontos}\nTempo: {tempo}')
#
#     cv2.imshow('Imagem', img_correspondencia)
#     cv2.imwrite('ImagemMathing.png', img_correspondencia)
#     # cv2.imshow('Imagem Subtração', img_diferenca)
#     cv2.waitKey(0)
#     cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
