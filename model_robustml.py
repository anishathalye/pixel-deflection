from keras.applications.resnet50 import ResNet50, preprocess_input, decode_predictions
import numpy as np
import robustml

from methods import pixel_deflection, denoiser

class Model(robustml.model.Model):
  def __init__(self):
      self._model = ResNet50(weights='imagenet')
      self._dataset = robustml.dataset.ImageNet(shape=(224,224,3))
      self._threat_model = robustml.threat_model.Linf(epsilon=0.05)

  @property
  def dataset(self):
      return self._dataset

  @property
  def threat_model(self):
      return self._threat_model

  def classify(self, x, deflections=200, window=10, sigma=0.04):
      x = x * 255.0
      img = pixel_deflection(x, np.zeros(x.shape[:2]),
                             deflections, window, sigma)
      img = denoiser('wavelet', img/255.0, sigma)*255.0
      img = preprocess_input(np.stack([img],axis=0))
      return np.argmax(self._model.predict(img)[0])
