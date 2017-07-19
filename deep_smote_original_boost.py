from sklearn.base import BaseEstimator
import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KDTree
np.random.seed(0)

from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score


def g_score(y_true, y_score):
    return np.sqrt(precision_score(y_true, y_score) * recall_score(y_true, y_score))

class DeepSMOTEBoostClassifier(BaseEstimator):  

    def __init__(self, T=20, N=5, FLAGS_lambda = 0.000001, FLAGS_beta = 0.000001, k=10, n_smote=100, type_='smote'):
        self.T = T # number of iterations in boosting
        self.N = N # max depth of a tree
        self.random_state = 0
        
        # deep boosting parameters
        self.FLAGS_lambda = FLAGS_lambda
        self.FLAGS_beta = FLAGS_beta
        self.kTolerance = 1e-7
        self.the_normalizer = 1
        
        self.clf_s = []
        self.alpha_s = []
        self.added_or_not = np.zeros(self.T)
        self.leaves_in_clf = []
        self.summary_train = []
        self.summary_test = []
        self.num_examples = 0
        self.num_features = 0
        
        # SMOTE parameters
        self.k = k # number of neighbours
        self.n_smote = n_smote # number of synthetic objects
        
        self.type_ = type_
        
        
    def fit(self, X, y, verbose=True, X_test=None, y_test=None):
        def complexity_penalty(tree_size): 
            rademacher = np.sqrt(((2 * tree_size + 1) * np.log2(self.num_features + 2) \
                                * np.log1p(self.num_examples+self.n_smote)) * 1. / \
                                (self.num_examples + self.n_smote))
            
            if self.type_ == 'rus':
                rademacher = np.sqrt(((2 * tree_size + 1) * np.log2(self.num_features + 2) \
                                * np.log1p(self.num_examples - self.n_smote)) * 1. / \
                                (self.num_examples - self.n_smote))
            
            return ((self.FLAGS_lambda * rademacher + self.FLAGS_beta) * \
                   (self.num_examples + self.n_smote)*1.) / (2 * self.the_normalizer)
        
        
        
        def gradient(wgtd_error, tree_size, alpha, sign_edge):
            complexity_pen = complexity_penalty(tree_size)
            edge = wgtd_error - 0.5
            sign_alpha = 1 if (alpha >= 0) else -1

            if (np.abs(alpha) > self.kTolerance):
                return edge + sign_alpha * complexity_pen
            elif (np.abs(edge) <= complexity_pen):
                return 0
            else:
                return edge - sign_edge * complexity_pen
            
        
        def compute_eta(wgtd_error, tree_size, alpha):
            wgtd_error = max(wgtd_error, self.kTolerance)
            error_term = (1 - wgtd_error) * np.exp(alpha) - wgtd_error * np.exp(-alpha)
            complexity_pen = complexity_penalty(tree_size)
            ratio = complexity_pen / wgtd_error

            if np.abs(error_term) <= 2 * complexity_pen:
                return -alpha
            elif (error_term > 2 * complexity_pen):
                return np.log(-ratio + np.sqrt(ratio * ratio + (1 - wgtd_error) / wgtd_error))
            else:
                return np.log(ratio + np.sqrt(ratio * ratio +  max(1 - wgtd_error, self.kTolerance)))
            
        def smote_continuous(X, distribution, k, n_smote, random_state, t):

            # 1. generating synthetic objects
            n_minor, n_features = X.shape
            k = min([n_minor - 1, k])
            nn_tree = KDTree(X)
            
            # calculate margin and probabilitiess 
            random_state=t
            if len(self.alpha_s) >= 1:
                dists = self.get_margin(X, np.ones(len(X)))
                indices = np.where(dists < 0)[0]
                if len(indices > 0):
                    probs = dists[indices] * 1. / dists[indices].sum()
                    start_indices = np.random.RandomState(random_state).choice(list(indices), \
                        size=(n_smote,), p = probs)
                else:
                    start_indices = np.random.RandomState(random_state).choice(len(X), size=(n_smote,))
            
            if len(self.alpha_s) < 1:
                start_indices = np.random.RandomState(random_state).choice(len(X), size=(n_smote,))
            
            starts = X[start_indices, :]
            nn_dists, nn_idx = nn_tree.query(starts, k=k + 1)
            end_indices = nn_idx[np.arange(n_smote), np.random.RandomState(random_state)\
                .choice(np.arange(1, k + 1), n_smote)]
            ends = X[end_indices, :]
            shifts = np.random.RandomState(random_state).rand(n_smote)
            X_smote = np.multiply(starts, np.repeat(shifts[:, np.newaxis], n_features, axis=1)) \
                + np.multiply(ends, np.repeat((1. - shifts[:, np.newaxis]), n_features, axis=1))

            # 2. reweighting new objects  
            weight_smote = np.multiply(distribution[start_indices],shifts)+ \
                np.multiply(distribution[end_indices],(1.- shifts))
            return X_smote, weight_smote
        
        
        def rus(X, distribution, n_sample, random_state, t):
            rus_ind = np.random.RandomState(random_state).choice(len(X), len(X) - n_sample, replace=False)
            return X[rus_ind, :], distribution[rus_ind]
        
        def ros(X, distribution, n_sample, random_state, t):
            random_state=t
            ros_ind = np.random.RandomState(random_state).choice(len(X), size=(n_sample,), replace=True)
            return X[ros_ind, :], distribution[ros_ind]
            
        
        # start fitting
        if verbose:
            print 'fitting DEEP BOOST starts with params {} {} {} {}' \
                .format(self.T, self.N, self.FLAGS_lambda, self.FLAGS_beta)
        
        self.clf_s = []
        self.alpha_s = []
        self.added_or_not = np.zeros(self.T)
        self.leaves_in_clf = []
        self.the_normalizer = 1
        self.summary_train = []
        self.summary_test = []
        self.num_examples, self.num_features = X.shape
        distribution = np.ones(self.num_examples) * 1. / self.num_examples
        
        self.the_normalizer = np.exp(1) * self.num_examples
        
        for t in range(self.T):
            if verbose:
                print '------------------------------'
                print 'iteration {}: \n'.format(t),
            old_tree_is_best = False
            best_old_tree_idx = -1
            best_wgtd_error, wgtd_error, grad, best_gradient = 0, 0, 0, 0
            
            for i, old_tree in enumerate(self.clf_s):
                alpha = self.alpha_s[i]
                if (np.abs(alpha) < self.kTolerance):
                    continue
                    
                pr = old_tree.predict(X)
                wgtd_error = (1 - np.average(np.multiply(pr, y), weights=distribution, axis=0))*1. / 2
                sign_edge = 1 if (wgtd_error >= 0.5) else -1
                
                grad = gradient(wgtd_error, old_tree.tree_.node_count, alpha, sign_edge)
                
                if (np.abs(grad) >= np.abs(best_gradient)):
                    best_gradient = grad
                    best_wgtd_error = wgtd_error
                    best_old_tree_idx = i
                    old_tree_is_best = True
                    
                if verbose:
                    print '~ best tree from model, error: {}, grad: {}'.format(best_wgtd_error, best_gradient)

            eps_on_iteration = []
            d_massive = np.zeros(self.N)
            X_tr = X
            y_tr = y
            
            if self.type_ == 'smote':
                X_smote, w_smote = smote_continuous(X[y==1], distribution[y==1],self.k, 
                    self.n_smote, self.random_state, t)
                y_smote = np.ones(self.n_smote)
                X_tr = np.concatenate((X, X_smote))
                y_tr = np.concatenate((y, y_smote))
                sample_w = np.concatenate((distribution, w_smote)) / 
                    np.concatenate((distribution, w_smote)).sum()
            
            if self.type_ == 'ros':
                X_ros, w_ros = ros(X[y==1], distribution[y==1], self.n_smote, self.random_state, t)
                y_ros = np.ones(self.n_smote)
                X_tr = np.concatenate((X, X_ros))
                y_tr = np.concatenate((y, y_ros))
                sample_w = np.concatenate((distribution, w_ros)) / 
                    np.concatenate((distribution, w_ros)).sum()
            
            if self.type_ == 'rus':
                X_rus, w_rus = rus(X[y==-1], distribution[y==-1], self.n_smote, self.random_state, t)
                y_rus = np.ones(X_rus.shape[0])*(-1.)
                X_tr = np.concatenate((X[y==1], X_rus))
                y_tr = np.concatenate((y[y==1], y_rus))
                sample_w = np.concatenate((distribution[y==1], w_rus)) / 
                    np.concatenate((distribution[y==1], w_rus)).sum()
           
            for j in range(1, self.N+1):

                clf = DecisionTreeClassifier(max_depth=j, random_state=self.random_state)
                clf = clf.fit(X_tr, y_tr, sample_weight=sample_w)
                pr = clf.predict(X) 
                eps = (1 - np.average(np.multiply(pr, y), weights=distribution, axis=0)) * 1. / 2
                eps_on_iteration.append(eps)
                d_j = gradient(eps, clf.tree_.node_count, 0, -1)
                d_massive[j-1] = np.abs(d_j)
            
            k = np.argmax(d_massive)
            eps_t = eps_on_iteration[k]
            
            if (np.abs(d_massive[k]) > np.abs(best_gradient)):
                best_gradient = d_massive[k]
                best_wgtd_error = eps_t
                old_tree_is_best = False

            if verbose:
                print '~ best new tree, error: {}, grad: {}'.format(eps_t, d_massive[k])
                

            eta = 0
            if old_tree_is_best:

                if verbose:
                    print '~ old tree is chosen'

                alpha = self.alpha_s[best_old_tree_idx]
                tree_size = self.clf_s[best_old_tree_idx].tree_.node_count
                eta = compute_eta(best_wgtd_error, tree_size, alpha)
                self.alpha_s[best_old_tree_idx] += eta
                self.added_or_not[t] = -1 * self.clf_s[best_old_tree_idx].tree_.max_depth
            else:

                clf = DecisionTreeClassifier(max_depth=k+1, random_state=self.random_state)
                clf = clf.fit(X_tr, y_tr, sample_weight= sample_w)
                pred_train = clf.predict(X)
                eta = compute_eta(best_wgtd_error, clf.tree_.node_count, 0)
                
                self.clf_s.append(clf)
                self.alpha_s.append(eta)
                self.leaves_in_clf.append(clf.tree_.node_count)
                self.added_or_not[t] = k+1

            distribution=distribution * np.exp(-eta*np.multiply(pred_train, y))
    
            self.the_normalizer = distribution.sum()
            
            distribution = distribution / distribution.sum()

            # learning summary

            def get_summary(X, y):
                summary_t = {} 
                pr_train = self.predict(X)
                min_margin = np.min(self.get_margin(X, y))
                median_margin = np.median(self.get_margin(X, y))
                mean_margin = np.mean(self.get_margin(X, y))
                c = confusion_matrix(y, pr_train, labels=[-1, 1])
                summary_t['min_margin'] = min_margin
                summary_t['median_margin'] = median_margin
                summary_t['mean_margin'] = mean_margin
                summary_t['TP'] = c[1, 1]
                summary_t['FP'] = c[0, 1]
                summary_t['TN'] = c[0, 0]
                summary_t['FN'] = c[1, 0]
                summary_t['acc'] = (y == pr_train).astype(int).sum() *1. / len(y)
                summary_t['prec'] = precision_score(y, pr_train)
                summary_t['rec'] = recall_score(y, pr_train)
                summary_t['f1'] = f1_score(y, pr_train)
                summary_t['g'] = g_score(y, pr_train)
                return summary_t
            

            self.summary_train.append(get_summary(X, y))
            
            if X_test is not None:
                self.summary_test.append(get_summary(X_test, y_test))
                
                if verbose:
                    print 'ON TRAIN iteration is finished with {} '.format(summary_t)
                    print 'ON TEST iteration is finished with {} '.format(summary_te)
        
        return self
    
    
    def get_margin(self, X, y):
        y_answers = np.zeros(shape=(len(self.clf_s), X.shape[0]))

        for i, clf in enumerate(self.clf_s):
            pred = clf.predict(X)
            y_answers[i, :] = pred * self.alpha_s[i]
            
        t = y_answers.sum(axis=0)
        return (t * y) *1. / np.sum(self.alpha_s)


    def predict(self, X, with_margin=False):
        y_answers = np.zeros(shape=(len(self.clf_s), X.shape[0]))

        for i, clf in enumerate(self.clf_s):
            pred = clf.predict(X)
            y_answers[i, :] = pred * self.alpha_s[i]
            
        t = y_answers.sum(axis=0)
        assert not np.any(np.sign(t)==0)
        t = np.sign(t)
        return t

