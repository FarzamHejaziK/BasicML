import numpy as np
from DecisionTree import DT
from collections import Counter

class RF:
    def __init__(self, Ntree = 10, min_Node_split = 2,max_depth = 100) -> None:
        self.Ntree = Ntree
        self.min_Node_split = min_Node_split
        self.max_depth = max_depth
    
    def fit(self,X,y):
        self.Trees = []
        for i in range(self.Ntree):
            tree = DT(min_Node_split = self.min_Node_split , max_depth = self.max_depth) 
            Xtmp,ytmp  = self._random_sampling(X,y)
            tree.fit(Xtmp,ytmp)
            self.Trees.append(tree)

    def _random_sampling(self,X,y):
        idx = np.random.choice(len(X), int(len(X)/min(self.Ntree,10)), replace = False)
        return X[idx],y[idx]
    
    def predict(self,X):
        out = []
        for x in X:
            out.append(self._find_cat(x))
        return out
    
    def _find_cat(self,x):
        predictions = [t.predict(np.array([x])) for t in self.Trees]
        return self._maj_c(np.array(predictions).flatten())
    
    def _maj_c(self,y):
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]

        

