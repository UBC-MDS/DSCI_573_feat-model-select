from utils import *
import matplotlib.pyplot as plt
import mglearn
from imageio import imread
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.model_selection import cross_validate, train_test_split
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, RidgeCV
from sklearn.preprocessing import (
    OneHotEncoder,
    OrdinalEncoder,
    PolynomialFeatures,
    StandardScaler,
)

from mpl_toolkits.mplot3d import Axes3D, axes3d


def gen_outlier_data(n=20, Noutliers=2, rand_seed=0):
    # generate random data
    np.random.seed(rand_seed)
    x = np.random.randn(n)
    y = 10 * x
    # add random outliers
    y[:Noutliers] = -100 * (x[:Noutliers] + np.random.randn(Noutliers))

    X = x[:, None]  # reshape for sklearn
    return X, y

def plot_reg(X_toy, y_toy, xlabel='feat1', y_label='y', preds=None, line=False):
    plt.figure(figsize=(6, 4), dpi=80)
    plt.xlabel("feat1")
    plt.ylabel("y")
    plt.scatter(X_toy, y_toy, s=40, edgecolors=(0, 0, 0));
    if line:
        plt.plot(X_toy, preds, color='red', linewidth=1);
    plt.show()

def compare_regression_lines(X_toy, y_toy, sklearn_preds, our_preds):
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))

    ax[0].set_xlabel("feat1")
    ax[0].set_ylabel("y")
    ax[0].scatter(X_toy, y_toy, s=40, edgecolors=(0, 0, 0));
    ax[0].plot(X_toy, sklearn_preds, color='red')
    ax[0].set_title('sklearn OLS line')

    ax[1].set_xlabel("feat1")
    ax[1].scatter(X_toy, y_toy, s=40, edgecolors=(0, 0, 0));
    ax[1].plot(X_toy, our_preds, color='red')
    ax[1].set_title('Our line');    
    
def plot_tree_decision_boundary(
    model, X, y, x_label="x-axis", y_label="y-axis", eps=None, ax=None, title=None
):
    if ax is None:
        ax = plt.gca()

    if title is None:
        title = "max_depth=%d" % (model.tree_.max_depth)

    mglearn.plots.plot_2d_separator(
        model, X.to_numpy(), eps=eps, fill=True, alpha=0.5, ax=ax
    )
    mglearn.discrete_scatter(X.iloc[:, 0], X.iloc[:, 1], y, ax=ax)
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


def plot_tree_decision_boundary_and_tree(
    model, X, y, height=6, width=16, x_label="x-axis", y_label="y-axis", eps=None
):
    fig, ax = plt.subplots(
        1,
        2,
        figsize=(width, height),
        subplot_kw={"xticks": (), "yticks": ()},
        gridspec_kw={"width_ratios": [1.5, 2]},
    )
    plot_tree_decision_boundary(model, X, y, x_label, y_label, eps, ax=ax[0])
    ax[1].imshow(tree_image(X.columns, model))
    ax[1].set_axis_off()
    plt.show()
    
def plot_fruit_tree(ax=None):
    import graphviz

    if ax is None:
        ax = plt.gca()
    mygraph = graphviz.Digraph(
        node_attr={"shape": "box"}, edge_attr={"labeldistance": "10.5"}, format="png"
    )
    mygraph.node("0", "Is tropical?")
    mygraph.node("1", "Has pit?")
    mygraph.node("2", "Is red?")
    mygraph.node("3", "Mango")
    mygraph.node("4", "Banana")
    mygraph.node("5", "Cherry")
    mygraph.node("6", "Kiwi")
    mygraph.edge("0", "1", label="True")
    mygraph.edge("0", "2", label="False")
    mygraph.edge("1", "3", label="True")
    mygraph.edge("1", "4", label="False")
    mygraph.edge("2", "5", label="True")
    mygraph.edge("2", "6", label="False")
    mygraph.render("tmp")
    ax.imshow(imread("tmp.png"))
    ax.set_axis_off()    
    

def plot_knn_clf(X_train, y_train, X_test, n_neighbors=1, class_names=['class 0','class 1'], test_format='star'):
    # credit: This function is based on: https://github.com/amueller/mglearn/blob/master/mglearn/plot_knn_classification.py
    plt.clf()    
    print('n_neighbors', n_neighbors)
    dist = euclidean_distances(X_train, X_test)
    closest = np.argsort(dist, axis=0)
    clf = KNeighborsClassifier(n_neighbors=n_neighbors).fit(X_train, y_train)
    plot_train_test_points(X_train, y_train, X_test, class_names, test_format)
    for x, neighbors in zip(X_test, closest.T):
        for neighbor in neighbors[:n_neighbors]:
            plt.arrow(
                x[0],
                x[1],
                X_train[neighbor, 0] - x[0],
                X_train[neighbor, 1] - x[1],
                head_width=0,
                fc="k",
                ec="k",
            )    
    plt.show()

def plot_knn_decision_boundaries(X_train, y_train, k_values = [1,11,100]):
    fig, axes = plt.subplots(1, len(k_values), figsize=(15, 4))

    for n_neighbors, ax in zip(k_values, axes):
        clf = KNeighborsClassifier(n_neighbors=n_neighbors)
        scores = cross_validate(clf, X_train, y_train, return_train_score=True)
        mean_valid_score = scores["test_score"].mean()
        mean_train_score = scores["train_score"].mean()
        clf.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(
            clf, X_train.to_numpy(), fill=True, eps=0.5, ax=ax, alpha=0.4
        )
        mglearn.discrete_scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], y_train, ax=ax)
        title = "n_neighbors={}\n train score={}, valid score={}".format(
            n_neighbors, round(mean_train_score, 2), round(mean_valid_score, 2)
        )
        ax.set_title(title)
        ax.set_xlabel("longitude")
        ax.set_ylabel("latitude")
    axes[0].legend(loc=1);    

def plot_train_test_points(X_train, y_train, X_test, class_names=['class 0','class 1'], test_format='star'):
    training_points = mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train)
    if test_format == "circle": 
        test_points = mglearn.discrete_scatter(
                X_test[:, 0], X_test[:, 1], markers="o", c='k', s=18
            );
    else: 
        test_points = mglearn.discrete_scatter(
                X_test[:, 0], X_test[:, 1], markers="*", c='g', s=16
            );        
    plt.legend(
        training_points + test_points,
        [class_names[0], class_names[1], "test point(s)"],
    )  
    

def plot_support_vectors(svm, X, y):
    mglearn.plots.plot_2d_separator(svm, X, eps=.5)
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    sv = svm.support_vectors_ # plot support vectors
    # class labels of support vectors are given by the sign of the dual coefficients
    sv_labels = svm.dual_coef_.ravel() > 0
    mglearn.discrete_scatter(sv[:, 0], sv[:, 1], sv_labels, s=15, markeredgewidth=3)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1");
    


def plot_svc_gamma(param_grid, X_train, y_train, x_label="longitude", y_label='latitude'): 
    fig, axes = plt.subplots(1, len(param_grid), figsize=(len(param_grid)*5, 4))
    for gamma, ax in zip(param_grid, axes):
        clf = SVC(gamma=gamma)
        scores = cross_validate(clf, X_train, y_train, return_train_score=True)
        mean_valid_score = scores["test_score"].mean()
        mean_train_score = scores["train_score"].mean()
        clf.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(
            clf, X_train, fill=True, eps=0.5, ax=ax, alpha=0.4
        )
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        title = "gamma={}\n train score={}, valid score={}".format(
            gamma, round(mean_train_score, 2), round(mean_valid_score, 2)
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    axes[0].legend(loc=1);    
    
def plot_svc_C(param_grid, X_train, y_train, x_label="longitude", y_label='latitude'): 
    fig, axes = plt.subplots(1, len(param_grid), figsize=(len(param_grid)*5, 4))
    for C, ax in zip(param_grid, axes):
        clf = SVC(C=C, gamma=0.01)
        scores = cross_validate(clf, X_train, y_train, return_train_score=True)
        mean_valid_score = scores["test_score"].mean()
        mean_train_score = scores["train_score"].mean()
        clf.fit(X_train, y_train)
        mglearn.plots.plot_2d_separator(
            clf, X_train, fill=True, eps=0.5, ax=ax, alpha=0.4
        )
        mglearn.discrete_scatter(X_train[:, 0], X_train[:, 1], y_train, ax=ax)
        title = "C={}\n train score={}, valid score={}".format(
            C, round(mean_train_score, 2), round(mean_valid_score, 2)
        )
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
    axes[0].legend(loc=1);    
    



def make_bracket(s, xy, textxy, width, ax):
    annotation = ax.annotate(
        s, xy, textxy, ha="center", va="center", size=20,
        arrowprops=dict(arrowstyle="-[", fc="w", ec="k",
                        lw=2,), bbox=dict(boxstyle="square", fc="w"))
    annotation.arrow_patch.get_arrowstyle().widthB = width

    
def plot_improper_processing(estimator_name):
    # Adapted from https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/plot_improper_preprocessing.py#L12
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    for axis in axes:
        bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9], left=[0, 12, 15],
                         color=['white', 'grey', 'grey'], hatch="//",
                         align='edge', edgecolor='k')
        bars[2].set_hatch(r"")
        axis.set_yticks(())
        axis.set_frame_on(False)
        axis.set_ylim(-.1, 6)
        axis.set_xlim(-0.1, 20.1)
        axis.set_xticks(())
        axis.tick_params(length=0, labeltop=True, labelbottom=False)
        axis.text(6, -.3, "training folds",
                  fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(13.5, -.3, "validation fold",
                  fontdict={'fontsize': 14}, horizontalalignment="center")
        axis.text(17.5, -.3, "test set",
                  fontdict={'fontsize': 14}, horizontalalignment="center")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[0])
    make_bracket(estimator_name + " fit", (6, 3), (6, 4), 12, axes[0])
    make_bracket(estimator_name + "predict", (13.4, 3), (13.4, 4), 2.5, axes[0])

    axes[0].set_title("Cross validation")
    axes[1].set_title("Test set prediction")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[1])
    make_bracket(estimator_name + " fit", (7.5, 3), (7.5, 4), 15, axes[1])
    make_bracket(estimator_name + " predict", (17.5, 3), (17.5, 4), 4.8, axes[1])    
    
    
def plot_proper_processing(estimator_name):
    # Adapted from https://github.com/amueller/mglearn/blob/106cf48ef03710ef1402813997746741aa6467da/mglearn/plot_improper_preprocessing.py#L12
    
    fig, axes = plt.subplots(2, 1, figsize=(15, 8))

    for axis in axes:
        bars = axis.barh([0, 0, 0], [11.9, 2.9, 4.9],
                         left=[0, 12, 15], color=['white', 'grey', 'grey'],
                         hatch="//", align='edge', edgecolor='k')
        bars[2].set_hatch(r"")
        axis.set_yticks(())
        axis.set_frame_on(False)
        axis.set_ylim(-.1, 4.5)
        axis.set_xlim(-0.1, 20.1)
        axis.set_xticks(())
        axis.tick_params(length=0, labeltop=True, labelbottom=False)
        axis.text(6, -.3, "training folds", fontdict={'fontsize': 14},
                  horizontalalignment="center")
        axis.text(13.5, -.3, "validation fold", fontdict={'fontsize': 14},
                  horizontalalignment="center")
        axis.text(17.5, -.3, "test set", fontdict={'fontsize': 14},
                  horizontalalignment="center")

    make_bracket("scaler fit", (6, 1.3), (6, 2.), 12, axes[0])
    make_bracket(estimator_name + " fit", (6, 3), (6, 4), 12, axes[0])
    make_bracket(estimator_name + " predict", (13.4, 3), (13.4, 4), 2.5, axes[0])

    axes[0].set_title("Cross validation")
    axes[1].set_title("Test set prediction")

    make_bracket("scaler fit", (7.5, 1.3), (7.5, 2.), 15, axes[1])
    make_bracket(estimator_name + " fit", (7.5, 3), (7.5, 4), 15, axes[1])
    make_bracket(estimator_name + " predict", (17.5, 3), (17.5, 4), 4.8, axes[1])
    fig.subplots_adjust(hspace=.3)
    
def plot_original_scaled(
    X_train,
    X_test,
    train_transformed,
    test_transformed,
    title_transformed="Properly transformed",
):
    fig, axes = plt.subplots(1, 2, figsize=(15, 4))
    axes[0].scatter(X_train[:, 0], X_train[:, 1], label="Training set", s=60)
    axes[0].scatter(
        X_test[:, 0],
        X_test[:, 1],
        marker="^",
        color=mglearn.cm2(1),
        label="Test set",
        s=60,
    )
    axes[0].legend(loc="upper right")

    axes[0].set_title("Original Data")

    axes[1].scatter(
        train_transformed[:, 0], train_transformed[:, 1], label="Training set", s=60
    )
    axes[1].scatter(
        test_transformed[:, 0],
        test_transformed[:, 1],
        marker="^",
        color=mglearn.cm2(1),
        label="Test set",
        s=60,
    )
    axes[1].legend(loc="upper right")
    axes[1].set_title(title_transformed);    
    
    
def plot_logistic_regression(x, w):
    import graphviz
    sentiment = 'pos' if sum(w) > 0 else 'neg'    
    lr_graph = graphviz.Digraph(node_attr={'shape': 'circle', 'fixedsize': 'False'},
                                graph_attr={'rankdir': 'LR', 'splines': 'line'})
    inputs = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_0")
    output = graphviz.Digraph(node_attr={'shape': 'circle'}, name="cluster_2")

    for i in range(len(x)):
        inputs.node(x[i], labelloc="c")
    inputs.body.append('label = "inputs"')
    inputs.body.append('color = "white"')

    lr_graph.subgraph(inputs)

    output.body.append('label = "output"')
    output.body.append('color = "white"')
    output.node("y_hat=%s" %sentiment)

    lr_graph.subgraph(output)
    print('Weighted sum of the input features = %0.3f y_hat = %s' %(sum(w), sentiment))
    for i in range(len(w)):
        lr_graph.edge(x[i], "y_hat=%s" %sentiment, label=str(w[i]))
    return lr_graph    
    
def plot_confusion_matrix_ex(tn, fp, fn, tp, target='Fraud'):
    plt.figure(figsize=(7, 7))
    confusion = np.array([[tn, fp], [fn, tp]])
    plt.text(0.40, .7, confusion[0, 0], size=45, horizontalalignment='right')
    plt.text(0.40, .2, confusion[1, 0], size=45, horizontalalignment='right')
    plt.text(.90, .7, confusion[0, 1], size=45, horizontalalignment='right')
    plt.text(.90, 0.2, confusion[1, 1], size=45, horizontalalignment='right')
    plt.xticks([.25, .75], ["predicted not " + target, "predicted " + target], size=20, rotation=25)
    plt.yticks([.25, .75], ["true " + target, "true not " + target ], size=20)
    plt.plot([.5, .5], [0, 1], '--', c='k')
    plt.plot([0, 1], [.5, .5], '--', c='k')

    plt.xlim(0, 1)
    plt.ylim(0, 1)
    

def plot_confusion_matrix_example(tn, fp, fn, tp, target='Fraud'):
    fig, ax = plt.subplots(1, 2, figsize=(20, 6), subplot_kw={'xticks': (), 'yticks': ()})

    plt.setp(ax, xticks=[.25, .75], xticklabels=["predicted not " + target, "predicted " + target],
       yticks=[.25, .75], yticklabels=["true " + target, "true not " + target ])    
    confusion = np.array([[tn, fp], [fn, tp]])
    ax[0].text(0.40, .7, confusion[0, 0], size=45, horizontalalignment='right')
    ax[0].text(0.40, .2, confusion[1, 0], size=45, horizontalalignment='right')
    ax[0].text(.90, .7, confusion[0, 1], size=45, horizontalalignment='right')
    ax[0].text(.90, 0.2, confusion[1, 1], size=45, horizontalalignment='right')
    ax[0].plot([.5, .5], [0, 1], '--', c='k')
    ax[0].plot([0, 1], [.5, .5], '--', c='k')

    ax[0].set_xlim(0, 1)
    ax[0].set_ylim(0, 1)    
    
    ax[1].text(0.45, .6, "TN", size=100, horizontalalignment='right')
    ax[1].text(0.45, .1, "FN", size=100, horizontalalignment='right')
    ax[1].text(.95, .6, "FP", size=100, horizontalalignment='right')
    ax[1].text(.95, 0.1, "TP", size=100, horizontalalignment='right')
    ax[1].plot([.5, .5], [0, 1], '--', c='k')
    ax[1].plot([0, 1], [.5, .5], '--', c='k')
    ax[1].yaxis.set_tick_params(labelsize=12)
    ax[1].set_xlim(0, 1)
    ax[1].set_ylim(0, 1)  
    
def make_num_tree_plot(preprocessor, X_train, y_train, X_test, y_test, num_trees, scoring_metric='accuracy'):
    """
    Make number of trees vs error rate plot for RandomForestClassifier

    Parameters
    ----------
    model: sklearn classifier model
        The sklearn model
    X_train: numpy.ndarray
        The X part of the train set
    y_train: numpy.ndarray
        The y part of the train set
    X_test: numpy.ndarray
        The X part of the test/validation set
    y_test: numpy.ndarray
        The y part of the test/validation set
    num_trees: int
        The value for `n_estimators` argument of RandomForestClassifier
    Returns
    -------
        None
        Shows the number of trees vs error rate plot

    """
    train_scores = []
    test_scores = []
    for ntree in num_trees:
        model = make_pipeline(preprocessor, RandomForestClassifier(n_estimators=ntree))
        scores = cross_validate(
            model, X_train, y_train, return_train_score=True, scoring=scoring_metric
        )
        train_scores.append(np.mean(scores["train_score"]))
        test_scores.append(np.mean(scores["test_score"]))

    plt.semilogx(num_trees, train_scores, label="train")
    plt.semilogx(num_trees, test_scores, label="cv")
    plt.legend()
    plt.xlabel("number of trees")
    plt.ylabel("scores") 
    
    
def plot_svc_3d_decision_boundary(X_new, y, linear_svm_3d):
    coef, intercept = linear_svm_3d.coef_.ravel(), linear_svm_3d.intercept_

    # show linear decision boundary
    figure = plt.figure()
    ax = figure.add_subplot(projection='3d')
    #ax = Axes3D(figure, elev=-152, azim=-26)
    xx = np.linspace(X_new[:, 0].min() - 2, X_new[:, 0].max() + 2, 50)
    yy = np.linspace(X_new[:, 1].min() - 2, X_new[:, 1].max() + 2, 50)

    XX, YY = np.meshgrid(xx, yy)
    ZZ = (coef[0] * XX + coef[1] * YY + intercept) / -coef[2]
    ax.plot_surface(XX, YY, ZZ, rstride=8, cstride=8, alpha=0.3)
    mask = y == 0
    ax.scatter(
        X_new[mask, 0],
        X_new[mask, 1],
        X_new[mask, 2],
        c="b",
        s=60,
        edgecolor="k",
    )
    ax.scatter(
        X_new[~mask, 0],
        X_new[~mask, 1],
        X_new[~mask, 2],
        c="r",
        marker="^",
        s=60,
        edgecolor="k",
    )

    ax.set_xlabel("feat0")
    ax.set_ylabel("feat1")
    ax.set_zlabel("feat1^2")
    return XX, YY    

def plot_3d_reg(X_sq, y, surface=False):
    figure = plt.figure()
    # visualize in 3D
    ax = figure.add_subplot(projection='3d')
    #ax = Axes3D(figure, elev=-152, azim=-26)
    ax.scatter(
        X_sq[:, 0],
        X_sq[:, 1], 
        y, 
        s=60,
        edgecolor="k",
    )

    # plot the plane
    if surface: 
        lr = LinearRegression()
        lr.fit(X_sq, y)        
        x_surf, y_surf = np.meshgrid(np.linspace(X_sq[:, 0].min(), X_sq[:, 0].max()),np.linspace(X_sq[:, 1].min(), X_sq[:, 1].max()))
        z_surf=lr.coef_[0]*x_surf+lr.coef_[1]*y_surf+lr.intercept_        
        ax.plot_surface(x_surf,y_surf,z_surf,rstride=1, cstride=1, color='b', alpha=0.3)
    
    ax.set_xlabel("feat1")
    ax.set_ylabel("feat1^2")
    ax.set_zlabel("y");
    
def plot_mglearn_3d(X_new, y):
    figure = plt.figure()
    # visualize in 3D    
    #ax = Axes3D(figure, elev=-152, azim=-26)
    ax = figure.add_subplot(projection='3d')
    # plot first all the points with y == 0, then all with y == 1
    mask = y == 0
    ax.scatter(
        X_new[mask, 0],
        X_new[mask, 1],
        X_new[mask, 2],
        c="b",
        s=60,
        edgecolor="k",
    )
    ax.scatter(
        X_new[~mask, 0],
        X_new[~mask, 1],
        X_new[~mask, 2],
        c="r",
        marker="^",
        s=60,
        edgecolor="k",
    )
    ax.set_xlabel("feat0")
    ax.set_ylabel("feat1")
    ax.set_zlabel("feat1^2");
    
def plot_Z_space_boundary_in_X_space(linear_svm_3d, X, y, XX, YY):
    ZZ = YY ** 2
    dec = linear_svm_3d.decision_function(np.c_[XX.ravel(), YY.ravel(), ZZ.ravel()])
    plt.contourf(
        XX,
        YY,
        dec.reshape(XX.shape),
        levels=[dec.min(), 0, dec.max()],
        cmap=mglearn.cm2,
        alpha=0.5,
    )
    mglearn.discrete_scatter(X[:, 0], X[:, 1], y)
    plt.xlabel("Feature 0")
    plt.ylabel("Feature 1");    
    
    
def plot_orig_transformed_svc(linear_svm, X, X_transformed, y): 
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    mglearn.discrete_scatter(X[:, 0], np.zeros_like(X), y, ax=axes[0])
    axes[0].set_title('Original 1-d data')
    mglearn.discrete_scatter(X_transformed[:, 0], X_transformed[:, 1], y, ax=axes[1])
    axes[1].set_xlabel("feat0")
    axes[1].set_ylabel("feat0^2")
    axes[1].set_title('Transformed data')    
    mglearn.discrete_scatter(X_transformed[:, 0], X_transformed[:, 1], y, ax=axes[2])
    axes[2].set_title('SVC decision boundary')        
    mglearn.plots.plot_2d_separator(linear_svm, X_transformed, ax=axes[2])
    axes[0].legend();    
    
    
def plot_loss_diagram(labels_inside=False): # From Mike's notebook: https://github.com/UBC-CS/cpsc340-2020w2/blob/main/lectures/19_linear-classifiers-fit.ipynb        
    grid = np.linspace(-2,2,1000)
    plt.figure(figsize=(6, 4), dpi=80)
    plt.xlabel('$y_iw^T x_i$', fontsize=18)
    plt.ylabel('$f_i(w)$', fontsize=18)
    plt.xlim(-2,2)
    plt.ylim(-0.025,3)
    plt.fill_between([0, 2], -1, 3, facecolor='blue', alpha=0.2);
    plt.fill_between([-2, 0], -1, 3, facecolor='red', alpha=0.2);
    plt.yticks([0,1,2,3]);

    if labels_inside:
        plt.text(-1.95, 2.73, "incorrect prediction", fontsize=15) # 2.68
        plt.text(0.15, 2.73, "correct prediction", fontsize=15)
    else:
        plt.text(-1.95, 3.1, "incorrect prediction", fontsize=15) # 2.68
        plt.text(0.15, 3.1, "correct prediction", fontsize=15)


    plt.tight_layout()

def plot_train_valid_poly_deg(X_train, y_train, X_valid = None, y_valid = None, valid=True):
    plt.figure(figsize=(16, 14))
    count = 1
    degrees = [0, 1, 2, 3, 6, 10, 12, 16, 20]
    for deg in degrees:
        pipe_poly_lr = make_pipeline(PolynomialFeatures(degree=deg), LinearRegression())
        plt.subplot(3, 3, count)
        pipe_poly_lr.fit(X_train, y_train)
        mglearn.discrete_scatter(X_train, y_train, s=8)
        plt.plot(X_train, pipe_poly_lr.predict(X_train), color="green", linewidth=2)
        if valid:
            plt.title(
                "p = %s, tr=%0.2f, val = %0.2f"
                % (
                    str(deg),
                    pipe_poly_lr.score(X_train, y_train),
                    pipe_poly_lr.score(X_valid, y_valid),
                )
            )
        else:
            plt.title(
                "p = %s, train = %0.2f"
                % (str(deg), pipe_poly_lr.score(X_train, y_train))
            )
        count += 1    


def plot_coefficient_magnitudes(ridge1, ridge2, lr, alpha1 = 100, alpha2 = 0.01):
    plt.figure(figsize=(8, 6))
    plt.plot(
        ridge1.coef_, "^", markersize=8, markeredgecolor="black", label="Ridge alpha=" + str(alpha1)
    )
    plt.plot(
        ridge2.coef_, "^", markersize=8, markeredgecolor="black", label="Ridge alpha=" + str(alpha2)
    )
    plt.plot(
        lr.coef_, "o", markersize=8, markeredgecolor="black", label="LinearRegression"
    )
    plt.xlabel("Coefficient index")
    plt.ylabel("Coefficient magnitude")
    plt.hlines(0, 0, len(lr.coef_))
    plt.ylim(-25, 25)
    plt.legend();
    
def plot_poly_deg(X_train, y_train, X_valid=None, y_valid=None, degree=2, ax=None, valid=False):
    if ax is None:
        ax = plt.gca()
    mglearn.discrete_scatter(X_train, y_train, s=8)
    pipe_poly_lr = make_pipeline(PolynomialFeatures(degree=degree), LinearRegression())
    pipe_poly_lr.fit(X_train, y_train)
    ax.plot(X_train, pipe_poly_lr.predict(X_train), color="green", linewidth=2)

    if valid:
        ax.set_title(
            "p = %s, tr=%0.3f, val = %0.3f"
            % (
                str(degree),
                pipe_poly_lr.score(X_train, y_train),
                pipe_poly_lr.score(X_valid, y_valid),
            )
        )
    else:
        ax.set_title(
            "p = %s, train = %0.2f" % (str(degree), pipe_poly_lr.score(X_train, y_train))
        )        
        
def compare_poly_degrees(deg1, deg2, X_train, y_train, X_valid=None, y_valid=None): 
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))    
    pt = PolynomialFeatures(degree=deg1)
    X_train_deg1 = pt.fit_transform(X_train)
    lr_deg1 = LinearRegression().fit(X_train_deg1, y_train)    
    plot_poly_deg(X_train, y_train, X_valid, y_valid, degree=deg1, ax=ax[0], valid=True)

    pt = PolynomialFeatures(degree=deg2)
    X_train_deg2 = pt.fit_transform(X_train)
    lr_deg2 = LinearRegression().fit(X_train_deg2, y_train)    
    plot_poly_deg(X_train, y_train, X_valid, y_valid, degree=deg2, ax=ax[1], valid=True)
        

def plot_cross_validated_rfe(pipe_rfecv):
    # yerr = pipe_rfecv.named_steps['rfecv'].cv_results_['std_test_score']
    print("Optimal number of features : %d" % pipe_rfecv.named_steps['rfecv'].n_features_)
    min_features_to_select = pipe_rfecv.named_steps['rfecv'].get_params()['min_features_to_select']
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel("Number of features selected")
    plt.ylabel("Cross validation score (R^2)")
    plt.plot(
        range(min_features_to_select, len(pipe_rfecv.named_steps['rfecv'].cv_results_['mean_test_score']) + min_features_to_select),
        pipe_rfecv.named_steps['rfecv'].cv_results_['mean_test_score']
    )
    plt.show()        
        
        
from sklearn.model_selection import learning_curve, KFold
def plot_learning_curve(est, X, y):    
    training_set_size, train_scores, test_scores = learning_curve(
        est, X, y, train_sizes=np.linspace(.1, 1, 20), cv=KFold(20, shuffle=True, random_state=1))
    estimator_name = est.__class__.__name__
    parameter = est.get_params()['alpha']
    line = plt.plot(training_set_size, train_scores.mean(axis=1), '--',
                    label="train " + estimator_name + ', alpha =' + str(parameter) )
    plt.plot(training_set_size, test_scores.mean(axis=1), '-',
             label="test " + estimator_name + ', alpha =' + str(parameter), c=line[0].get_color())
    plt.legend(loc="best", fontsize=8)
    plt.xlabel('Training set size')
    plt.ylabel('Score (R^2)')
    plt.ylim(0.0, 0.7)

    