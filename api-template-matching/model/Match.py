import random
import cv2
import numpy as np
from matplotlib import pyplot as plt


class Match:
    def __init__(self, query_image, train_image, detector, descriptor):
        self.__query_image = query_image
        self.__train_image = train_image
        self.__detector = detector
        self.__descriptor = descriptor
        self.__start_time = 0
        self.__end_time = 0

    @property
    def query_image(self):
        return self.__query_image

    @query_image.setter
    def query_image(self, query_image):
        self.__query_image = query_image
    @property
    def train_image(self):
        return self.__train_image

    @train_image.setter
    def train_image(self, train_image):
        self.__train_image = train_image

    @property
    def detector(self):
        return self.__detector

    @property
    def descriptor(self):
        return self.__descriptor

    @property
    def start_time(self):
        return self.__start_time

    @start_time.setter
    def start_time(self, start_time):
        self.__start_time = start_time

    @property
    def end_time(self):
        return self.__end_time

    @end_time.setter
    def end_time(self, end_time):
        self.__end_time = end_time


    def __repr__(self):
        return {
            "query_image": self.__query_image,
            "train_image": self.__train_image,
            "detector": self.__detector,
            "descriptor": self.__descriptor
        }

    def training(self):
        print(f'./image/{self.query_image}')
        query_image = cv2.imread(f'./image/{self.query_image}')
        train_image = cv2.imread(f'./image/{self.train_image}')

        train_image = self.equalize_color(query_image, train_image)
        inliers = 0
        outliers = 0
        subtraction = 0
        if self.detector == "SIFT":
            if self.descriptor == "SIFT":
                inliers, outliers, subtraction, img_show = self.sift_sift(query_image, train_image)
                cv2.imwrite('../../template-matching/app-correspondencia-imagem/public/images/image_match.jpg', img_show)
                if inliers > 5 and subtraction < 80:
                    print("Número de inliers: ", inliers)
                    print("Número de outliers: ", outliers)
                    print("Subtração (img2-img1): ", subtraction)
                else:
                    print("Imagens diferentes")

            elif self.detector == "FAST":
                print("Implementar...")
            elif self.detector == "AKAZE":
                print("Implementar...")

        if (inliers > 5 and subtraction < 80):
            return {
                "sucess": True,
                "inliers": inliers,
                "outliers": outliers,
                "subtraction": subtraction
            }
        else:
            return {
                "sucess": False,
                "inliers": inliers,
                "outliers": outliers,
                "subtraction": subtraction
            }


    @staticmethod
    def equalize_color(img1, img2):
        # Converter ambas as imagens para o espaço de cor Lab
        lab1 = cv2.cvtColor(img1, cv2.COLOR_BGR2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_BGR2LAB)

        # Calcular a média e o desvio padrão dos canais L, a e b da imagem img1
        mean1, std1 = cv2.meanStdDev(lab1)
        mean1 = mean1.ravel()  # achatada em um array 1D, que pode ser mais facilmente manipulado ou processado
        std1 = std1.ravel()  # achatada em um array 1D, que pode ser mais facilmente manipulado ou processado

        # Normalizar os canais da imagem img2 pela média e desvio padrão da imagem img1
        lab2[:, :, 0] = np.clip(
            (((lab2[:, :, 0] - np.mean(lab2[:, :, 0])) * (std1[0] / np.std(lab2[:, :, 0]))) + mean1[0]), 0, 255).astype(
            np.uint8)
        lab2[:, :, 1] = ((lab2[:, :, 1] - np.mean(lab2[:, :, 1])) * (std1[1] / np.std(lab2[:, :, 1]))) + mean1[1]
        lab2[:, :, 2] = ((lab2[:, :, 2] - np.mean(lab2[:, :, 2])) * (std1[2] / np.std(lab2[:, :, 2]))) + mean1[2]

        # Converter de volta para o espaço de cor BGR
        result = cv2.cvtColor(lab2, cv2.COLOR_LAB2BGR)

        cv2.imwrite('../../template-matching/app-correspondencia-imagem/public/images/equalize_color_img1.jpg', result)

        return result


    @staticmethod
    def sift_sift(query_img, train_img):
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
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)

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
            cv2.imwrite('../../template-matching/app-correspondencia-imagem/public/images/image_subtraction.jpg', img_diff)

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