
import numpy as np
## KMeans Clustering Algorithm
### Note: From here onwards, distance => euclidean distance
class KMeans:
    # KMeans clustering class
    # Input
    # -----
    # Data : Dataset 
    # n_clusters : Number of centroids
    # max_iter : No. of iterations, the algo runs 
    # rand_seed : seed for random function
    # Output : centroids, labels_of_the_pts
    
    def __init__(self,n_clusters=8,max_iter=300,random_state=None):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.random_state = random_state

    def fit_predict(self,X):
        self.data = X
        np.random.seed(self.random_state) # seed for the random values
        index = np.random.choice(len(self.data), self.n_clusters, replace=False) # Getting k random indices
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
        return self.labels_ # labels

    # This function calculates distance between all data points and centroids
    # Input : data_array, centroids
    # Output : All distances
    def all_distances(self,data,centroids):
        outer_list=[]
        for i in data: # Iterating over all the data points
            inner_list=[]
            for j in centroids: # Iterating over all the centroids
                inner_list.append(distance(i,j))
            outer_list.append(inner_list)
        return np.array(outer_list) # Sending all distances as numpy array






## Hierarchical Clustering
class AgglomerativeClustering:
# Hierarchical Clustering (Main function)
# Strategy : Agglomerative
# Input : data, linkage
# Output : clusters

    def __init__(self,n_clusters=2,linkage="single"):
        self.n_clusters = n_clusters
        self.linkage = linkage

    def fit_predict(self,X):
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
        # This part producing the labels in a particular format
        # You can change this to produce the output similar to Kmeans's output
        # I am using this format because it is useful for calculating Jaccard index
        decor_l=[]
        for v in cluster.values():
            decor_l.append(v)
        # Converting them to sklearn's format
        self.labels_= self.clustertolabels(decor_l)
        return self.labels_

    def clustertolabels(self,clusters):
        ln = sum([len(c) for c in clusters])
        labels = np.zeros(ln,dtype = np.int)
        ind = -1
        for c in clusters:
            ind+=1
            for i in c:
                labels[i] = ind
        return labels


    # This function computers proximity upper triangular matrix
    # Input : data
    # Output : Proximity matrix
    def d_matrix(self,data):
        n=data.shape[0] # No. of rows in proximity matrix 
        d=np.empty(shape=[n,n]) # Initializing the matrix
        d.fill(np.inf)          # Defining the matrix with infinity, since we need to calculate min.
        # Iterating over upper triangle
        for i in range(n-1):
            for j in range(i+1,n):
                d[i,j]=distance(data[i],data[j]) # Storing distances
        return d

    # This function defining initial singleton clusters
    # Input : No_of_clusters
    # Output : singleton clusters
    def get_initial_cluster(self,n):
        c={}
        for i in range(n):
            c[i]={i}   # Initializing singleton clusters 
        return c

    # This function updates the proximity matrix
    # linkage => 
    # 'single' (based on nearest pt)
    # 'complete' (based on farthest pt)
    # 'average' (based on average of pts)
    # Inputs : proximity_matrix, index, index, set containing candidate indices, linkage
    # Output : returns the proximity matrix
    def update_d(self,d,p,q,t_set,linkage):
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

    # This function updates (merges) the centroids
    # Input : centroid, indices,indices
    # Output : updated centroids
    def update_cluster(self,c,p,q):
        i=c.pop(max(p,q)) # deleting centroid : max(p,q)
        m=min(p,q)
        c[m]=c[m].union(i) # combining centroids 
        return c






# This function calculates distance between two multi-dimentional points
# Input : Two multi-dimentional points
# Output : Distance between them 
def distance(pt1,pt2):
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