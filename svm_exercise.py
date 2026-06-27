"""
In this document, we will learn about classification tasks and Support Vector Machines
(called SVMs from now on, both in text and code). Please read carefully the doctexts and
comments intertwined with the code and do not execute the file until explicitly told so
in a doctext. Also do not modify the code until explicitly told so in a doctext or
once you have finished reading through it once.

Consider a dataset containing only numerical, continuous variables. As an example,
think of the famous Fisher's "iris" dataset, that contains numerical measurements in
centimiters of the lenght and width of the petals and sepals of a set of 150 flowers.
The flowers all belong to the genus Iris, but they're members of three different species
of that genus: Virginica, Setosa and Versicolor (the dataset contains 50 samples of
each). A classification task is one that answers the question: ¿Can one say to which
species a given flower belongs to only by looking at its measurements?. Mathematically,
a classifier is a function that takes a numerical vector (in R4, in this case) and
outputs a label belonging to some discrete set ({'Virginica', 'Setosa', 'Versicolor'} or
{'0', '1', '2'}, in this case) whose elements are called labels. The classification problem
for a given dataset consists in finding a classifier that minimizes some quantification of
the error (e.g. the number of mislabeled samples).

SVMs are a family of algorithms used extensively in classification and even regression 
(numerical prediction) tasks, but here we will focus on binary classification (that is,
a classification problem where there are only two labels) since it's the simplest to explain
and visualize.

The simplest SVM is the linear one, and it is also the basis on top of which all others
are built. The main idea behind linear SVMs is to try to find a hyperplane that separates
the sets of points belonging to the different classes. That is, such that all samples with a
label lay on one side of the hyperplane, and all samples having the other label lay on the
other side of the hyperplane. Of course, for many datasets such a hyperplane does not exist
(we'll talk about these later), but nevertheless it is possible to find one that
minimizes the error in some sense. But for those datasets in which a plane exists, it is often
possible to find infinitely many hyperplanes. In this case, it is clearly interesting to
try to maximize the distance between the plane and the closest points to it, since this will make
our classifier more robust to noise or errors in the measurements (cf. 'margins.png').

We will not cover the details, but it is interesting to know that an optimization problem
can be posed for this, and it can be naturally reduced to a quadratic minimization problem with
affine constraints. The theory of Karush-Kuhn-Tucker can then be invoked to solve it explicitly.
There is a natural correspondece between each vector in the training dataset and each affine
constraint of the problem. When solving the system of equations of the KKT conditions, there will
be some active constraints. The vectors associated to those active constraints are called
support vectors, and hence the name of the algorithm. Geometrically, support vectors are
the points which are closest (minimizers of the distance) to the separating hyperplane.

Once a hyperplane has been found, a given sample can be classified just by computing its inner
product with the normal vector of the hyperplane and checking whether it's larger-than or
smaller-than a specific threshold; then assigning the label based on that.

Now we'll see a linear SVM in action.
"""
import numpy as np
from matplotlib import pyplot
from sklearn import datasets, svm

SEED = 41
N_SAMPLES = 100
TRAINING_SAMPLES = 80

"""
We generate the dataset at random and split it between training and testing samples.
This splitting is important because it mimicks a real-world scenario in which we train
our model and then use it to classify previously unseen samples. In our case, we will
use only the training points to find our hyperplane, and hope that it also classifies well
the testing (unseen) samples.

Also, this dataset will contain two variables, so that the resulting points are 2D and hence
easily visualizable.
"""
points, labels = datasets.make_blobs(n_samples=N_SAMPLES, centers=2, cluster_std=1, random_state=SEED)


training_points = points[:TRAINING_SAMPLES]
training_labels = labels[:TRAINING_SAMPLES]

testing_points = points[TRAINING_SAMPLES:]
testing_labels = labels[TRAINING_SAMPLES:]

"""
This is scikit-learn's implementation of a SVM. Notice how we pass it the 'kernel=linear'
parameter to specify we want the linear classifier (more on why it's a 'kernel' and which others
exist later!)
"""
classifier = svm.SVC(kernel='linear')

"""
Now we can compute the solution of the optimization problem in order to find our hyperplane.
scikit-learn implements this in a simple-to-use '.fit' function.
"""
classifier.fit(points[:TRAINING_SAMPLES, :], labels[:TRAINING_SAMPLES])

# Plot everything (skip this code)
# ----- (Start of unimportant code)
pyplot.scatter(training_points[training_labels == 0][:, 0], training_points[training_labels == 0][:, 1], 
               c='pink', edgecolors='k', alpha=0.8, label='Class 0 training points')
pyplot.scatter(testing_points[testing_labels == 0][:, 0], testing_points[testing_labels == 0][:, 1], 
               c='red', edgecolors='k', alpha=0.8, label='Class 0 testing points')

pyplot.scatter(training_points[training_labels == 1][:, 0], training_points[training_labels == 1][:, 1], 
               c='lightblue', edgecolors='k', alpha=0.8, label='Class 1 training points')
pyplot.scatter(testing_points[testing_labels == 1][:, 0], testing_points[testing_labels == 1][:, 1], 
               c='darkblue', edgecolors='k', alpha=0.8, label='Class 1 testing points')


ax = pyplot.gca()
xlim = ax.get_xlim()
ylim = ax.get_ylim()

YY, XX = np.meshgrid(
    np.linspace(ylim[0], ylim[1], 100),
    np.linspace(xlim[0], xlim[1], 100)
)
xy = np.vstack([XX.ravel(), YY.ravel()]).T
distance_to_bdry = classifier.decision_function(xy).reshape(XX.shape)

ax.contour(XX, YY, distance_to_bdry, colors='k', levels=[-1, 0, 1], alpha=0.8,
           linestyles=['--', '-', '--'], linewidths=[1, 2, 1])


support_vectors = classifier.support_vectors_
pyplot.scatter(support_vectors[:, 0], support_vectors[:, 1], s=100, linewidth=1.5, 
               facecolors='none', edgecolors='black', label='Support Vectors')

pyplot.legend(loc='best')
pyplot.title("SVM Decision Boundary, Margins, and Support Vectors")
# ----- (End of unimportant code)

"""
Finally, we plot our dataset and classifier.

Now you should execute the code. Use your IDE for this or run "python svm_exercise.py" in
a terminal. In both cases it is very important that you have the virtual environment of
the project properly set up and that it is currently active (you should see an indication
of this in the PS1 of your terminal, or somewhere in your IDE).

Spend a bit of time examining the figure and its legend while you read this. Notice how
there is a black line in the middle of the figure: this is the separating hyperplane (of
course, a hyperplane is just a line in R2!). This should perfectly separate the two classes
of points since the dataset was generated to be linearly separable.
There are two dotted lines which are parallel to the separating hyperplane. Notice how
all testing samples are outside of the region enclosed by the dotted lines, except the
support vectors, which are exactly at its boundary. The distance between those is defined
as the *margin* of the classifier. It is the largest possible among all hyperplanes separating
the tranining set. There are testing points inside of the margin region, but since the margin is
so large, the classifier still properly labels all of them. If the margin were sufficiently small
(in a suboptimal separating hyperplane), the SVM would misclassify some of the testing points.
"""
pyplot.show()

"""
This was an ideal case in which the dataset was linearly separable, but it is often the case
that it isn't! What do we do then? There are two lines of possible developments here.

1. Any linear classifier will necessarily misclassify some sample (by definition of linear
separability!), but it is possible to modify the minimization problem in a way that penalizes
having those misclassified points too far away from the hyperplane. It also reduces to the
first case if the dataset is, in fact, linearly separable. This approach is called the
'soft-margin' formulation, in contraposition to the 'hard-margin' previous one. In fact, the
default implementation of scikit-learn already uses this since it is a necessity in almost all
real-world applications.

To see it in action, go to line 66 of this file and increase the standard deviation of
the points ('cluster_std') to 3. Then write the file, re-execute the code and read the
following paragraph.

Notice how the dataset isn't linearly separable anymore, but the classifier still works
quite well: only one testing point is mislabeled. Support vectors now correspond to those
inside the closure of the margin region, although the interpretation of what the margin is
and how it is computed is a bit more abstract now, depending on details of the resulting KKT
conditions which we will not discuss.

2. A really clever idea is to try to find some function that maps R2 to some higher-dimensional
vector space in such a way that the images of the points (not the points themselves!) are 
linearly separable or almost linearly separable, and then solve the classification problem for the
images. Provided we are able to find such a function (this is done via trial-and-error for each
problem specifically), this works quite well for small datasets. But for larger problems or
computationally-constrained environments (think of a system with a camera that has to classify
fish passing through a conveyor belt and only has a milliseconds to produce an output for each fish)
it can become too computationally expensive to compute the image by that function of each point we
want to classify, and then the associated inner product. Mathematics provides a very beautiful
and sophisticated way to mitigate this, using the theory of Reproducing Kernel Hilbert Spaces, which
is worth commenting but sadly cannot be covered in due detail here. The main element here is a
'kernel' function that allows efficient computation of the quantities necessary for classification.
Not all mapping functions admit a kernel, nor are all kernels useful for computation; but they allow
for a great degree of flexibility while keeping computational cost admissible. In fact, some kernels
(specifically the Gaussian ones) correspond to mapping the original points to infinite-dimensional
(Hilbert) spaces! A general heuristic here is that the higher the dimension, the easier it is to linearly
separate points, since there's more hyperplanes to choose from; so Gaussian kernels tend to work quite
well.

To see this in action, go to line 66 of this file and replace it by the following:
'points, labels = datasets.make_circles(noise=0.1, factor=0.5, random_state=SEED)'
This generates a dataset whose classes are laid out in annuli, and hence will never be
remotely separable by a hyperplane. Now go to line 79 and change the 'kernel' parameter to the string
"rbf" (the name "rbf" means Radial Basis Function and in this case it corresponds to the
Gaussian kernel). Now re-execute the code and observe the figure.

Notice how the separating line has become a closed curve resembling a circle. This is the preimage
of the separating hyperplane under the infinite-dimensional mapping. The margin region has also become
"round". And the classifier works stupendously.
"""

"""
To recapitulate, we have covered classification tasks and the basics of support vector machines.

But we missed a lot of details about the theory which you can seek out and study if you wish. The
mathematics of the optimization problem associated are very nice; and those behind the "kernel trick"
(that is, the way of mapping points to higher-dimensional spaces while keeping computation bounded via
a kernel) are particularly beautiful and uselful.

You may also browse the documentation of scikit-learn (or ask an LLM) and play around with different
datasets and kernels for a bit!
"""
