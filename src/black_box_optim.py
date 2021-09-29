import logging
import numpy as np
from scipy.stats import pearsonr
import time

logger = logging.getLogger(__name__)

class SPSA:

    def __init__(self, triples1, triples2, predictor, targets
                , triples1_dev=None, triples2_dev=None, targets_dev=None
                , init_lr=0.75, A=2, alpha=0.5, gamma=0.05, c=0.01
                , eval_steps=100, n_batch=4, check_every_n_batch=350):

        # predictor must have predict and set_params and get_params
        self.predictor = predictor
        self.targets = targets
        self.triples1 = triples1
        self.triples2 = triples2

        if not triples1_dev:
            self.triples1_dev = self.triples1
            self.triples2_dev = self.triples2
            self.targets_dev = self.targets
        else:
            self.triples1_dev = triples1_dev
            self.triples2_dev = triples2_dev
            self.targets_dev = targets_dev
         
        self.init_lr = init_lr
        self.A = A
        self.alpha = alpha
        self.gamma = gamma
        self.c = c
        self.eval_steps = eval_steps
        self.n_batch = n_batch
        self.check_every_n_batch = check_every_n_batch
        
        return None

    def error(self, x, y):
        """Calculates error over two arrays/lists with scalars
        
        Here, the error is defined as 1 - pearsonr, but can be set differently.
        
        Args:
            x (list or array with floats): input 1
            y (list or array with floats): input 2
            
        Returns:
            error
        """

        pr = pearsonr(x, y)[0]
        if pr >= -1 and pr <= 1:
            return 1 - pr
        return 0.0

    def pseudo_grad(self, x, ids, c, rand):
        """estimate gradient

        Args:
            x (numpy array): input parameters
            ids (list): indeces of training examples on which we perform the
                        estimation
            c: constant
            rand: random vector

        Returns:
            estimated gradient
        """

        in_1 = [self.triples1[i] for i in ids]
        in_2 = [self.triples2[i] for i in ids]
        
        self.predictor.set_params(x + c * rand)
        a = self.predictor.predict(in_1, in_2) 
        error1 = self.error(a, [self.targets[i] for i in ids])
        
        self.predictor.set_params(x - c * rand)
        b = self.predictor.predict(in_1, in_2) 
        error2 = self.error(b, [self.targets[i] for i in ids])
        num = error1 - error2
        
        return num / (2 * c * rand), error1, error2

    def clip(self, grad, x=1):
        """clip values in array"""
        
        clipped = np.clip(grad, -x, x)
        return clipped

    def fit(self):
        """fit the parameters of the underlying predictor"""

        param_shape = self.predictor.get_params().shape
        
        logger.debug("parameter shape  {}".format(param_shape))
        
        prstart = pearsonr(self.targets_dev
                , self.predictor.predict(self.triples1_dev
                , self.triples2_dev, parallel=True))[0]
        
        logger.info("start pearsonr {}".format(prstart))
        
        iters = 1
        eval_steps_done = 0
        grad_norms = []
        errors = []
        best_dev_score = prstart
        best_params = self.predictor.get_params().copy()
        while True:
            # sample mini batch ids
            i = np.random.randint(0, len(self.triples1), size=self.n_batch) 
            
            # sample from bernoulli
            rand = np.random.randint(0, 2, size=param_shape)
            rand[rand == 0] = -1
            
            # update c
            c = self.c / iters**self.gamma
            #obtain current params
            x = self.predictor.get_params().copy()
            #compute pseudo grad
            pseudo_grad, error1, error2 = self.pseudo_grad(x, i, c, rand)
            #collect some stats
            grad_norms.append(np.linalg.norm(pseudo_grad))
            errors.append(error1)
            errors.append(error2)
            
            #update learning rate
            lr = self.init_lr / (iters + self.A)**self.alpha
            
            #gradient clip
            pseudo_grad = self.clip(pseudo_grad, x=4)
            
            #SGD rule
            params = x - lr * pseudo_grad
            
            #update params
            self.predictor.set_params(params)
            iters += 1

            #maybe check results on the development set
            if iters % self.check_every_n_batch == 0:
                # some debugging
                logger.debug("mean of grad norms {}; \
                        max of grad values {}".format(np.mean(grad_norms), np.max(pseudo_grad)))
                grad_norms = []
                logger.debug("mean of errors {}; \
                        current learning rate {}; \
                        c={}".format(np.mean(errors), lr, self.c / iters**self.gamma))
                errors = []

                #compute score on dev
                logger.info("conducting evaluation step {}; \
                        processed examples={}; \
                        systime={}".format(eval_steps_done, iters * self.n_batch, time.time()))
                dev_preds = self.predictor.predict(self.triples1_dev
                                                    , self.triples2_dev
                                                    , parallel=True)
                pr = pearsonr(self.targets_dev, dev_preds)
                pr = pr[0]
                logger.info("current score {}".format(pr))
                if pr > best_dev_score:
                    logger.info("new high score on dev! Old score={}; \
                            New score={}, \
                            improvement=+{}; \
                            total improvement=+{};\
                            saving params...".format(
                                    best_dev_score, pr, pr - best_dev_score, pr - prstart))
                    best_dev_score = pr
                    best_params = params.copy()
                eval_steps_done += 1
            
            # maybe stop training
            if self.eval_steps == eval_steps_done:
                self.predictor.set_params(best_params)
                return None

        
