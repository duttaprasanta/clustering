"""
Contains : 
    1. KMeans clustering algorithm implemention
    2. Agglomerative clustering algorithm implementation
"""

import numpy as np

## Note: From here onwards, distance => euclidean distance

class KMeans:
    '''
    KMeans Clustering Algorithm
    
    * Note : distance => Euclidean distance
    
    * Parameters : 
        1. n_clusters : int, default = 8 
            [Number of clusters] 
        2. max_iter : int, default = 300 
            [Max. number of iterations, the algorithm runs]
        3. random_state : int, default = None
            [Seed for the random function]
    * Attributes :
        1. labels_ : numpy array (int)
            [Output labels after clustering]
        2. cluster_centers_ : numpy array (float)
            [Centroids after clustering]
    '''
    def __init__(self,n_clusters=8,max_iter=300,random_state=None):
        '''
        This function initializes the parameters
        * Parameters : 
            1. n_clusters : int, default = 8
                    [Number of clusters]
            2. max_iter : int, default = 300
                    [Max. number of iterations, the algorithm runs]
            3. random_state : int, default = None
                    [Seed for the random function]
        '''
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self,X):
        '''
        This function fits X and predict clustering labels
        * Parameters : 
            1. X : numpy array (float)
                [Dataset, on which clustering has to be done] 
        * Returns : 
            1. labels_ : numpy array (int)
                [Computed labels after clustering]
        '''
        self.data = X
        np.random.seed(self.random_state) # seed for the random values
        index = np.random.choice(len(self.data), self.n_clusters, replace=False) # Getting random indices
        centroids=self.data[index,:]    # Initial centroids
        for _ in range(self.max_iter):  # Runs upto max_iter
            distances=self.all_distances(self.data,centroids) # calculating all distances
            # Assigning pts to the nearest cluster
            points = np.array([np.argmin(i) for i in distances]) 
            centroids = [] 
            for ind in range(self.n_clusters):
                centroid = self.data[points==ind].mean(axis=0) # Recomputing centroids
                centroids.append(centroid)
            centroids = np.vstack(centroids) # Arranging centroids as an array (row-major)
        self.cluster_centers_ = centroids
        self.labels_ = points
        return self.labels_ # Returning labels

    def all_distances(self,data,centroids):
        '''
        This function calculates distance between all data points and centroids
        * Parameters : 
            1. data : numpy_array (float)
            2. centroids : numpy array (float)
                [cluster_centers_]
        * Returns : 
            1. [all distances as numpy array]

        '''
        outer_list=[]
        for i in data: # Iterating over all the data points
            inner_list=[]
            for j in centroids: # Iterating over all the centroids
                inner_list.append(distance(i,j))
            outer_list.append(inner_list)
        return np.array(outer_list) # Sending all distances as numpy array






class AgglomerativeClustering:
    '''
    Agglomerative hierarchical clustering

    * Note : distance => Euclidean distance

    * Parameters : 
        1. n_clusters : int, default = 2
            [Number of clusters]
        2. linkage : str, default = 'single'
            [linkage -> {'single', 'complete','average'}]
    * Attributes : 
        1. labels_ : numpy array (int)
            [Computed labels after clustering]

    '''
    def __init__(self,n_clusters=2,linkage="single"):
        '''
        This function initializes the parameters
        * Parameters : 
            1. n_clusters : int, default = 2
                [Number of clusters]
            2. linkage : str, default = 'single'
                [linkage -> {'single', 'complete','average'}]
        '''
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self,X):
        '''
        This function fits X and predict clustering labels
        * Parameters : 
            1. X : numpy array (float)
                [Dataset, on which clustering has to be done] 
        * Returns : 
            1. labels_ : numpy array (int)
                [Computed labels after clustering]
        '''
        n=X.shape[0]     # No.of rows
        d=self.d_matrix(X)    # Proximity matrix
        cluster=self.get_initial_cluster(n)  # Defining initial singleton clusters
        # This set is helpful for keeping track of active rows (or columns) in d
        s=set(range(n))     
        for _ in range(n-self.n_clusters): # Because we need k clusters
            p,q=np.unravel_index(np.argmin(d, axis=None), d.shape) # Getting min. value in d
            t_set=s-{p,q} 
            d=self.update_d(d,p,q,t_set,self.linkage) # Updating d
            cluster=self.update_cluster(cluster,p,q) # Updating clusters
            s=s-{max(p,q)} # Deleting used row (or column)
        decor_l=[]
        for v in cluster.values():
            decor_l.append(v)
        # Converting cluster format -> Label format
        self.labels_= self.clustertolabels(decor_l)
        return self.labels_

    def clustertolabels(self,clusters):
        '''
        This function converts cluster format -> Label format
        * Parameters :
            1. clusters : numpy array (float)
                [data in cluster format]
        * Returns :
            1. [data in label format in numpy array (int)]
        '''
        ln = sum([len(c) for c in clusters])
        labels = np.zeros(ln,dtype = np.int)
        ind = -1
        for c in clusters:
            ind+=1
            for i in c:
                labels[i] = ind
        return labels


    def d_matrix(self,data):
        '''
        This function computes proximity upper triangular matrix
        * Parameters : 
            1. data : numpy array(float)
        * Returns : 
            1. [Proximity matrix as numpy array (float)]
        '''
        n=data.shape[0] # No. of rows in proximity matrix 
        d=np.empty(shape=[n,n]) # Initializing the matrix
        d.fill(np.inf)  # Defining the matrix with infinity, since we need to calculate min.
        # Iterating over upper triangle
        for i in range(n-1):
            for j in range(i+1,n):
                d[i,j]=distance(data[i],data[j]) # Storing distances
        return d

    
    def get_initial_cluster(self,n):
        '''
        This function defines initial singleton clusters
        * Parameters : 
            1. n : int 
                [No. of clusters]
        * Returns :
            1. [singleton clusters as dictionary]
        '''
        c={}
        for i in range(n):
            c[i]={i}   # Initializing singleton clusters 
        return c

   
    def update_d(self,d,p,q,t_set,linkage):
        '''
        This function updates the proximity matrix
        * Parameters : 
            1. d : numpy array (float)
                [proximity_matrix]
            2. p : int 
                [index]
            3. q : int
                [index]
            4. t_set : set (int)
                [set containing candidate indices]
            5. linkage : str
                [linkage -> {'single', 'complete','average'}]
        * Returns : 
            1. [returns the proximity matrix as numpy array (float)]
        '''
        for i in t_set: # current set containing all values except p and q
            # Since only upper triangle contains values
            u,v=min(i,p),max(i,p) 
            w,x=min(i,q),max(i,q)
            if(linkage=="complete"):
                t=max(d[u,v],d[w,x])
            elif(linkage=="average"):
                t=(d[u,v]+d[w,x])/2
            else:     # single linkage
                t=min(d[u,v],d[w,x])
            # Updating the values according to the linkage criteria
            d[u,v]=t
            d[w,x]=t
        # Setting max(p,q) rows and cols to infinity    
        m_pq=max(p,q)
        d[m_pq,:]=np.inf
        d[:,m_pq]=np.inf
        return d


    def update_cluster(self,c,p,q):
        '''
        This function updates (merges) the centroids
        * Parameters : 
            1. c : numpy array (float)
                [centroid]
            2. p : int 
                [index]
            3. q : int 
                [index]
        * Returns : 
            1. [updated centroids as numpy array (float)]
        '''
        i=c.pop(max(p,q)) # deleting centroid : max(p,q)
        m=min(p,q)
        c[m]=c[m].union(i) # combining centroids 
        return c


def distance(pt1,pt2):
    '''
    This function calculates distance between two multi-dimentional points
    * Parameters : 
        1. pt1 [multi-dimentional points]
        2. pt2 [multi-dimentional points]
    * Returns :
        1. [Distance between them (float)] 
    '''
    # Checking the dimention of the points are equal or not
    # If not equal, then returing
    if(len(pt1)!=len(pt2)):
        print("Error distance(): The dimensions of two points are not equal")
        return  
    dim=len(pt1)  # Dimention of a point
    s=0
    for i in range(dim):
        s+=(pt1[i]-pt2[i])**2 # sum((pt1-pt2)^2)
    dist=np.sqrt(s)  # sqrt(sum((pt1-pt2)^2))
    return dist