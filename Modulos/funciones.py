# Numpy para calculo vectorial
import numpy as np
import math
# cv2 para trabajar con imagenes
import cv2

# matplotlib para graficar
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D

# Pandas para manipular datos
import pandas as pd

# sklearn para machine learning
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from sklearn.cluster import BisectingKMeans
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing, decomposition
# 
from IPython.display import clear_output 
from collections import Counter
from ipywidgets import interact, widgets

# Nuestra propia libreria de funciones
from Modulos.funciones import *

# Parametros por defecto de matplotlib
plt.rcParams['figure.figsize'] = [5, 5]

def gen_df(image):
    """
    Generar df de una imagen BGR o Grayscale
    """
    
    shape = image.shape
    if len(shape) == 3:
      ny, nx, _ = shape
      ncols = 5
    elif len(shape) == 2:
      ny, nx = shape
      ncols = 3

    npix = ny*nx
    data = np.zeros([npix, ncols], dtype="int")
    ind = np.indices((ny, nx))

    data[:, 0] = ind[0].flatten() #filas
    data[:, 1] = ind[1].flatten() #columnas

    data[:, 2] = image[:, :, 0].flatten() #rojo
    data[:, 3] = image[:, :, 1].flatten() # verde
    data[:, 4] = image[:, :, 2].flatten() # azul

    df = pd.DataFrame(data, columns=["fila", "columna", "b", "g", "r"])
    
    return df

def crop_limits(img):
  """Recibe una imagen, espera a la seleccion de los puntos de interes
  y extrae la capsula de petri
  """
  def click_event(event, x, y, flags, params):
     if event == cv2.EVENT_LBUTTONDOWN:
        print(f'({x},{y})')

        # put coordinates as text on the image
        cv2.putText(img_copy, f'({x},{y})',(x,y),
        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        # draw point on the image
        cv2.circle(img_copy, (x,y), 3, (0,255,255), -1)

        COORDS.append((x,y))
  
  COORDS = []

  # Copy the image to show and don't modify the original one
  img_copy = img.copy()

  # create a window
  cv2.namedWindow('Point Coordinates')

  # bind the callback function to window
  cv2.setMouseCallback('Point Coordinates', click_event)

  # display the image
  coords = []
  while True:
     cv2.imshow('Point Coordinates', img)
     k = cv2.waitKey(1) & 0xFF
     if len(COORDS) == 2:
      break

def mkcircle(image):
  """Borrar imagen fuera del circulo.

  Toma una imagen de un rectángulo y, asumiendo
  que los bordes de la imagen son tangentes al
  círculo, vuelve totalmente blancas las zonas
  externas al círculo.

  image: imagen cv2 (numpy array de tres dimensiones)
    Imagen a recortar
  """
  height = image.shape[0]
  width = image.shape[1]
  y_offset = 0
  x_offset = 0
  offset = -30
  for y in range(height):
    for x in range(width):
      if (
          (x - width/2 +x_offset) * (x - width/2 +x_offset) 
          + (y - height/2 +y_offset) * (y-height/2 +y_offset) 
          >= (height/2 + offset) * (height/2 +offset)
          ):
        
        image[y, x, 0] = 0
        image[y, x, 1] = 0
        image[y, x, 2] = 0
  return image

def mkcircle_petri(offset,background):
  """Borrar imagen fuera del circulo.

  Toma una imagen de un rectángulo y, asumiendo
  que los bordes de la imagen son tangentes al
  círculo, vuelve totalmente blancas las zonas
  externas al círculo.

  image: imagen cv2 (numpy array de tres dimensiones)
    Imagen a recortar
  """
  #image = petri
  height = petri.shape[0]
  width = petri.shape[1]
  for y in range(height):
    for x in range(width):
      if (
          (x - width/2) * (x - width/2) 
          + (y - height/2) * (y-height/2) 
          >= (height/2 - offset/10) * (height/2 -offset/10)
          ):
        
        petri[y, x, 0] = background
        petri[y, x, 1] = background
        petri[y, x, 2] = background
  plt.imshow(petri)
  
  #return petri
  return None

def blob_finder(image):
  """
  Esta funcion no se usa, la deje por las dudas. Se supone que puede
  identificar figuras genericas aisladas
  """
  # Set our filtering parameters
  # Initialize parameter setting using cv2.SimpleBlobDetector
  params = cv2.SimpleBlobDetector_Params()
  
  # Set Area filtering parameters
  params.filterByArea = False
  params.minArea = 10
  
  # Set Circularity filtering parameters
  params.filterByCircularity = True
  params.minCircularity = 0.9
  
  # Set Convexity filtering parameters
  params.filterByConvexity = False
  params.minConvexity = 0.2
  
  # Set inertia filtering parameters
  params.filterByInertia = True
  params.minInertiaRatio = 0.8
  
  # Create a detector with the parameters
  detector = cv2.SimpleBlobDetector_create(params)
  
  # Detect blobs
  keypoints = detector.detect(
    np.array(image, dtype="uint8")
  )
  
  # Draw blobs on our image as red circles
  blank = np.zeros((1, 1))
  blobs = cv2.drawKeypoints(
    image, keypoints, blank, (0, 0, 255),
    cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS
  )

  return blobs

def get_area(image, msg):
  r = cv2.selectROI(msg, image)

  # Crop image
  img = image[
      int(r[1]):int(r[1]+r[3]),
      int(r[0]):int(r[0]+r[2])
  ]

  return img, r

def set_threshold(thresh_min, thresh_max):
  global thresh_min_out 
  global thresh_max_out 
  # Binarizar colonia deseada y entrenar kmeans
  desired_gray = cv2.cvtColor(desired_colony, cv2.COLOR_BGR2GRAY)
  desired_bin = (
      (desired_gray > thresh_min) & (desired_gray < thresh_max)
  ) * 1
  thresh_min_out = thresh_min
  thresh_max_out  = thresh_max
  plt.imshow(desired_bin)
  plt.colorbar()
  plt.imshow(desired_colony, alpha=0.7)
  return thresh_min, thresh_max