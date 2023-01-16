from collections import Counter
import numpy as np

class Node:
    def __init__(self,left = None,right = None,feature = None, thr = None) -> None:
        self.left, self.right, self.feature, self.thr = left, right, feature, thr
        self.c = None
        self.is_leaf_node = False

    def class_value(self,c):
        self.c = c
        self.is_leaf_node = True


class DT:

    def __init__(self, min_Node_split = 2,max_depth = 100) -> None:
        self.min_Node_split = min_Node_split
        self.max_depth = max_depth
        self.root = Node()
    
    def fit(self, X, y):
        self._maketree(X,y,self.root,0)
    
    def _maketree(self,X,y,node,depth):
        if len(X) < self.min_Node_split or depth > self.max_depth or len(np.unique(y)) == 1:
            node.class_value(self._maj_c(y))
            return 
        feaure_idx, best_thr, leftind, rightind = self._find_best_feature_and_thr(X,y)
        node.feature = feaure_idx
        node.thr = best_thr  
        if len(leftind) > 0:
            node.left = Node()
            self._maketree(X[leftind],y[leftind],node.left,depth+1)
        if len(rightind) > 0:
            node.right = Node()
            self._maketree(X[rightind],y[rightind],node.right,depth+1)
    
    def _find_best_feature_and_thr(self,X,y):
        IG = -1 
        for i in range(len(X[0])):
            for j in range(len(y)):
                leftind_temp,rightind_temp,tempIG = self._information_gain(X[:,i],y,j)
                if tempIG > IG:
                    IG = tempIG
                    best_feature = i
                    best_thr = X[j,i]
                    leftind = leftind_temp
                    rightind =rightind_temp

        return best_feature, best_thr, leftind, rightind 

    def _information_gain(self,X,y,j):
        parent_ent = self._entropy(y)
        leftind = np.array(np.where(X <= X[j])).flatten()
        rightind = np.array(np.where(X > X[j])).flatten()
        left_ent = self._entropy(y[leftind]) 
        right_ent = self._entropy(y[rightind])
        IG = parent_ent - left_ent * (len(leftind)/len(y)) - right_ent * (len(rightind)/len(y))  
        return leftind,rightind,IG   

    def _maj_c(self,y):
        occurence_count = Counter(y)
        return occurence_count.most_common(1)[0][0]

    def _entropy(self,y):
        probs = np.bincount(y)/len(y)
        return np.sum([-1*p*np.log(p) for p in probs if p > 0])

    
    def predict(self,X):
        out = []
        for x in X:
            out.append(self._traverse_tree(x,self.root))
        return out

    def _traverse_tree(self,X,node):
        if node.is_leaf_node: return node.c
        if X[node.feature] > node.thr: 
            return self._traverse_tree(X,node.right)
        else:
            return self._traverse_tree(X,node.left)


    
        

    




    



