## Silhouette Index computation
import numpy as np

# This function returns indices of all pts in a cluster
# Input : labels, clusters
# Output : Indices
def indices_at_cluster(labels,cluster):
    return np.array([i for i,j in enumerate(labels) if j==cluster])


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



# This function calculates distances between all pts
# Input : all points
# Output : distance matrix
def all_pair_distances(data):
    n_points=data.shape[0] # No. of points in data
    d=np.zeros(shape=[n_points,n_points]) # Initializing an array with 0
    # Iterating over only one triangle
    for p in range(n_points):
        for q in range(p+1,n_points):
            d[p,q]=distance(data[p],data[q]) # Storing distances
            d[q,p]=d[p,q] # Making symmetry
    return d # Returning the distance matrix

# This function returns all "a"(intracluster distances)
# Input : labels, distance_matrix
# Output : All a values
def find_all_a(labels,d):
    n_centroids=len(np.unique(labels)) # Getting no. of centroids
    a=np.empty(shape=len(labels)) # Initializing array "a" with 0
    for c in range(n_centroids): # Iterating over all the centroids
        idx=indices_at_cluster(labels,c) # Getting indices of pts at cluster c
        for p in idx: # For every index
            sum=0
            for q in idx: # For every index
                sum+=d[p,q] # Sum of all intracluster distances
            a[p]=sum/(len(idx)-1) # Average of all intracluster distances for a pt 
    return a # Returning all "a"

# This function calculates distances between a point and all points in a cluster
# Input : A point's index, A cluster index, distance matrix, labels
# Output : Distance a point and a cluster
def pt_cluster_distance(pt,oth,d,labels):
    idx=indices_at_cluster(labels,oth) # Getting indices at cluster oth
    sum=0
    for i in idx: # Iterating over all indices
        sum+=d[pt,i] # Sum(distances)
    return sum/(len(idx)) # Avg(distances)

# This function finds all "b" (intercluster distances)
# Input : labels, distance_matrix
# Ouput : An array containing all "b" values
def find_all_b(labels,d):
    b=np.empty(shape=len(labels)) # Initializing array "b"
    n_centroids=len(np.unique(labels)) # Getting no. of centroids
    for c in range(n_centroids): # Iterating over all centroids
        other_centroids=set(range(n_centroids))-{c} # all_centroids - current_centroid
        idx=indices_at_cluster(labels,c) # Getting indices of all pts in cluster c
        for p in idx:  # Iterating over all indices
            t=[]
            for o in other_centroids: # Iterating over other centroids
                t.append(pt_cluster_distance(p,o,d,labels)) # Getting all intercluster distances
                b[p]=min(t) # Taking the min. of all intercluster distances
    return b # Returning b array
    
# This function calculates silhouette values of all points
# Input : intracluster distances, intercluster distances, labels
# Output : an array containing all silhouette values
def find_all_s(a,b,labels):
    if(len(a)!=len(b)): # Checking whether a and b of same size
        print("Error find_all_s() : length of a and b are not same")
        return  # Otherwise returning from the function
    s=np.empty(shape=len(a)) # Initializing array s
    n_centroids=len(np.unique(labels)) # Getting no. of centroids
    for c in range(n_centroids): # Iterating over all centroids 
        idx=indices_at_cluster(labels,c) # Indices of pts at cluster c
        for p in idx: # Iterating over all indices
            if(len(idx)==1): # s=0 when |C|=1
                s[p]=0
            else:
                s[p]=(b[p]-a[p])/max(a[p],b[p]) # s=(b-a)/max(a,b)
    
    return s # Returning silhoutte values
        
# Silhouette index computation function (Main function) 
# Input : data,labels
# Output : silhouette score over all points
def silhouette_score(X,labels):
    d=all_pair_distances(X) # all pair distances
    a=find_all_a(labels,d)     # all intracluster distances
    b=find_all_b(labels,d)     # all intercluster distances
    s=find_all_s(a,b,labels)   # all silhouette indices
    SC=np.mean(s)              # silhouette score over all points
    return SC

# -------------------------------------------Silhouette End --------------------------------------




# -------------------------------------------Jaccard Start ---------------------------------------
def jaccard_index(list1,list2):
    uni_lab1 = list(set(list1))
    uni_lab2 = list(set(list2))
    f = []
    for i,u1 in enumerate(uni_lab1):
        l = []
        for j,u2 in enumerate(uni_lab2):
            s1 = set(np.where(list1==u1)[0])
            s2 = set(np.where(list2==u2)[0])
            iou = inter_over_union(s1,s2)
            l.append((iou,i,j))
        m = max(l)
        f.append(m)
    f.sort(reverse=True)
    return f
                   
def inter_over_union(s1,s2):
    u=s1.union(s2) # Finding union
    i=s1.intersection(s2) # Finding intersection
    iou=(len(i)/len(u)) # (|intersection|/|union|)*100
    return iou