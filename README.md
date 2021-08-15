# About the repository
In this repository, the following classes and functions are implemented
- [**KMeans** clustering](https://en.wikipedia.org/wiki/K-means_clustering) class
- [**Agglomerative** clustering](https://en.wikipedia.org/wiki/Hierarchical_clustering) class
- [**Silhouette score**](https://en.wikipedia.org/wiki/Silhouette_(clustering)) function
- [**Jaccard Index**](https://en.wikipedia.org/wiki/Jaccard_index) function

The classes and functions are distributed into the following modules -
- **cluster** : It contains KMeans and Agglomerative classes
- **metrics** : It contains Silhouette score and Jaccard Index computation functions 
- **utils** : It contains a function named **export** which is used to export the results of clustering into a file

# Usage
To know how to use them, go through [**driver.ipynb**](https://github.com/duttaprasanta/clustering/blob/main/driver.ipynb)

# Running the codes
### Step 1
Clone this repository with `git clone https://github.com/duttaprasanta/clustering.git`. You may also download this repository as zip file and extract it.

### Step 2
Run the python files by installing required packages. For running the ipynb files you need to install jupter notebook or jupyter lab or visual studio code (an extension is needed). You may also run it by uploading into [Google Colab](https://colab.research.google.com/).  

If you stuck, with dependency conflicts, you can create a virtual environment and activate it by doing the following -
```
pip install virtualenv
virtualenv env
source env/bin/activate
pip install -r requirements.txt
```
Deactivate all the activated environment before doing the previous step. In case of conda, deactivate it using `conda deactivate`. In case of pip, do `deactivate`.

# LICENSE
GNU Affero General Public License v3.0. See [LICENSE](https://github.com/duttaprasanta/clustering/blob/main/LICENSE) for more information

# Useful links
- Email : prasanta7dutta@gmail.com
- Project link : https://github.com/duttaprasanta/clustering
