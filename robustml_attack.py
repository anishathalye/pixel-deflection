import robustml
from keras.applications.resnet50 import preprocess_input, ResNet50
from methods import pixel_deflection, denoiser
from model_robustml import Model
import sys
import argparse
import tensorflow as tf
import numpy as np
import keras

class BPDA(robustml.attack.Attack):
    def __init__(self, sess, model, epsilon, max_steps=1000, learning_rate=0.1, debug=False):
        self._sess = sess

        self._x = tf.placeholder(tf.float32, (1, 224, 224, 3))
        self._out = tf.log(ResNet50(weights='imagenet')(preprocess_input(self._x)))
        self._preds = tf.argmax(self._out, 1)
        self._label = tf.placeholder(tf.int32, ())
        self._loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=self._out, labels=[self._label])
        self._grad, = tf.gradients(self._loss, self._x)

        self._epsilon = epsilon * 255.0
        self._max_steps = max_steps
        self._learning_rate = learning_rate
        self._debug = debug

    def _defend(self, x, deflections=200, window=10, sigma=0.04):
        img = pixel_deflection(x, np.zeros(x.shape[:2]), deflections, window, sigma)
        return denoiser('wavelet', img/255.0, sigma)*255.0

    def run(self, x, y, target):
        if target is not None:
            raise NotImplementedError
        x = x * 255.0
        adv = np.copy(x)
        lower = np.clip(x - self._epsilon, 0, 255)
        upper = np.clip(x + self._epsilon, 0, 255)
        for i in range(self._max_steps):
            adv_def = self._defend(np.copy(adv))
            p, l, g = self._sess.run(
                [self._preds, self._loss, self._grad],
                {self._x: [adv_def], self._label: y}
            )
            if self._debug:
                print(
                    'attack: step %d/%d, loss = %g (true %d, predicted %s)' % (
                        i+1,
                        self._max_steps,
                        l,
                        y,
                        p
                    ),
                    file=sys.stderr
                )
            if y not in p:
                # we're done
                if self._debug:
                    print('returning early', file=sys.stderr)
                break
            adv += self._learning_rate * np.sign(g[0])
            adv = np.clip(adv, lower, upper)
        return adv / 255.0

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--imagenet-path', type=str, required=True,
            help='directory containing `val.txt` and `val/` folder')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int, default=100)
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    # initialize a model
    model = Model()

    # get session
    sess = keras.backend.get_session()


    # initialize an attack (it's a white box attack, and it's allowed to look
    # at the internals of the model in any way it wants)
    attack = BPDA(sess, model, epsilon=model.threat_model.epsilon, debug=args.debug)

    # initialize a data provider for ImageNet images
    provider = robustml.provider.ImageNet(args.imagenet_path, model.dataset.shape)

    success_rate = robustml.evaluate.evaluate(
        model,
        attack,
        provider,
        start=args.start,
        end=args.end,
        deterministic=True,
        debug=args.debug,
    )

    print('attack success rate: %.2f%% (over %d data points)' % (success_rate*100, args.end-args.start))

if __name__ == '__main__':
    main()
