import matplotlib,sklearn,numpy,os,imageio
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_mldata
mnist = fetch_mldata('MNIST original')

Matrix=mnist.data
Labels=mnist.target

def decomposition_demo(input_matrix,input_labels,function="pca"):
    #randomly choose 2000 samples from the mnist dataset
    matrix,dummy_matrix,labels,dummy_labels=train_test_split(input_matrix,
                                                             input_labels,
                                                             train_size=2000)
    del dummy_matrix,dummy_labels
    if function!="nmf":
        #change image format into float, then reshpae every image into 1D array
        matrix=matrix.reshape((len(matrix),-1)).astype(numpy.float64)
        #re-center 1D array
        sc=StandardScaler(with_mean=True,with_std=False)
        matrix_std=sc.fit_transform(matrix)

        if function=="pca":
            from sklearn.decomposition import PCA
            pca=PCA(n_components=2,whiten=True)
            projection=pca.fit_transform(matrix_std)
        elif function=="ica":
            from sklearn.decomposition import FastICA
            ica=FastICA(n_components=2)
            projection=ica.fit_transform(matrix_std)
        elif function=="t-sne":
            from sklearn.manifold import TSNE
            tsne=TSNE(n_components=2,n_iter=5000)
            projection=tsne.fit_transform(matrix_std)
        elif function=="pcoa":
            from sklearn.metrics.pairwise import pairwise_distances
            from sklearn.manifold import MDS
            matrix_std_dis=pairwise_distances(matrix_std,metric="braycurtis")
            pcoa=MDS(max_iter=5000,dissimilarity="precomputed")
            projection=pcoa.fit_transform(matrix_std_dis)
        elif function=="k-pca":
            from sklearn.decomposition import KernelPCA
            kpca=KernelPCA(n_components=2,kernel="cosine")
            projection=kpca.fit_transform(matrix_std)
        plt.scatter(projection[:,0],projection[:,1],c=labels,s=5)
        return(plt.show())

    #No need re-center data before perform NMF
    elif function=="nmf":
        from sklearn.decomposition import NMF
        nmf=NMF(max_iter=400)
        nmf_fit=nmf.fit(matrix)
        components=nmf_fit.components_
        if os.path.exists("NMF_test/")==False:
            os.mkdir("NMF_test")
        for i in range(len(components)):
            imageio.imsave("NMF_test/"+str(i)+".png",components[i].reshape((28,28,1)).astype(numpy.uint8))
    else:
        print("not a valid method")
            
#choose one the following methods: "pca", "ica", "t-sne", "pcoa", "k-pca" or "nmf",
#put it in the third arguement.
        
decomposition_demo(Matrix,Labels,"nmf")

