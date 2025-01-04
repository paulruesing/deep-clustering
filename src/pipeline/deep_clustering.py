import os
import csv
import pandas as pd
import numpy as np
from numpy.typing import ArrayLike

import matplotlib.pyplot as plt
import seaborn as sns

from scipy.cluster.vq import kmeans, whiten, vq

from sklearn.manifold import Isomap
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import euclidean_distances

from tqdm.notebook import tqdm
import imageio

import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras import Layer, InputSpec, losses, regularizers
import tensorflow.keras.backend as K

import src.utils.statistical_analysis as analysis
import src.utils.plotting as plotting


class Preprocessor:
    """
    A class to
    - remove missing values and outliers
    - analyse feature significance
    - standardize features
    - plot feature space overviews
    for optimal clustering input data.
    """
    def __init__(
            self,
            data: pd.DataFrame,
            plot_path=None,
            save_fig=False,
            plot_size=(16, 10)
    ) -> None:
        """
        Initiliase the pre-processor instance.

        Parameters
        ----------
        data: pd.DataFrame
            input data
        plot_path: str, optional
            path to save plot if save_fig is True
        save_fig: bool, default is False
            whether to save created plots
        plot_size: tuple, default is (16, 10)
            size (width, height) of output plots
        """
        self.data = data.copy()
        self.plot_path = plot_path
        self.save_fig = save_fig
        self.plot_size = plot_size
        self.standardized_data = None
        self.pca_loadings = None

    #############  information functions #############
    @property
    def feature_list(self) -> list:
        """
        Returns the list of features in the dataset.

        Returns
        -------
        list
            List of feature names.
        """
        return list(self.data.columns)

    def describe(self) -> str:
        """
        Provides an overview of the dataset including its size and features.

        Returns
        -------
        str
            A formatted string with dataset information.
        """
        length_str = "Amount of objects: " + str(len(self.data))
        feature_str = "Included features: " + str(self.feature_list)
        return "-- Preprocessor Object --\n\n" + length_str + "\n\n" + feature_str

    def __repr__(self):
        """
        Returns a string representation of the object.

        Returns
        -------
        str
            Description of the object.
        """
        return self.describe()

    def __str__(self):
        """
        Returns a printable representation of the object.

        Returns
        -------
        str
            Description of the object.
        """
        return self.describe()

    #############  outlier removal methods #############
    def mask_outlier(self, ser: ArrayLike, quantile=(0.01, 0.99), iqr_factor=2):
        """
        Identify outliers using the interquartile range (IQR) method.

        Parameters
        ----------
        ser : ArrayLike
            An array-like object containing numerical values for outlier detection.
        quantile : tuple
            A tuple containing two float values representing the quantiles for IQR calculation.
        iqr_factor : float
            Factor to multiply the IQR with when defining the lower and upper fences.

        Returns
        -------
        np.ndarray
            A boolean array where `True` indicates an outlier and `False` indicates a non-outlier.
        """
        q_a, q_b = np.quantile(ser, quantile)
        iqr = q_b - q_a
        lower_fence = q_a - iqr_factor * iqr
        higher_fence = q_b + iqr_factor * iqr
        return (ser < lower_fence) | (ser > higher_fence)

    def remove_outliers_by_iqr(self, cols=None, **iqr_mask_kwargs):
        """
        Remove rows containing outliers in the specified columns.

        Parameters
        ----------
        cols : list, optional
            List of columns to search for outliers. Defaults to all columns.
        iqr_mask_kwargs : dict
            Additional arguments passed to the `mask_outlier` method.
        """
        outlier_mask = [False] * len(self.data)

        if cols is None:
            cols = list(self.data.columns)

        for col in cols:
            # Iterate over each specified column and mark each outlier row in one mask
            outlier_mask = np.logical_or(
                outlier_mask, self.mask_outlier(self.data[col], **iqr_mask_kwargs)
            )
        self.data = self.data.loc[~outlier_mask]

    def data_cleansing(self, **outlier_kwargs) -> None:
        """
        Removes missing values and outliers based on the IQR method.

        Parameters
        ----------
        outlier_kwargs : dict
            Additional arguments passed to the `remove_outliers_by_iqr` method.
        """
        self.data.dropna(inplace=True)
        self.remove_outliers_by_iqr(**outlier_kwargs)

    #############  standardization methods #############
    def standardize(self, remove_nans=False) -> None:
        """
        Standardize the dataset by normalizing features to unit variance.

        Parameters
        ----------
        remove_nans : bool, default is False
            Whether to remove rows with NaN values before standardizing.

        Raises
        ------
        BaseException
            If NaN values are present and `remove_nans` is False.
        """
        if (self.data.isnull().any().any()):
            if remove_nans:
                # Convert to numpy array without NaNs:
                X = self.data.dropna().to_numpy()
                indices = self.data.dropna().index
                print("Removed NaN-values in standardized frame!")
            else:
                raise BaseException("Data contains NaN-values. Call standardize(remove_nans=True) to handle these!")
        else:
            # Convert to numpy array:
            X = self.data.to_numpy()
            indices = self.data.index

        # Normalize the observations on a per-feature basis
        # Each feature is divided by its standard deviation across all observations to give it unit variance.
        X_scaled = whiten(X)

        # Finally, the numpy array is reconverted to a pandas df and saved:
        self.standardized_data = pd.DataFrame(X_scaled, index=indices,
                                              columns=self.data.columns)

    #############  feature selection methods #############
    def calculate_pca(self, feature_cols=None, target_col=None,
                      no_of_components=None, treat_nans="delete rows") -> None:
        """
        Calculates the PCA representation of the dataset.

        Parameters
        ----------
        feature_cols : list, optional
            Subset of columns to consider.
        target_col : str, optional
            Column to retain (unchanged) in the resulting PCA dataframe.
        no_of_components : int, optional
            Number of principal components to compute.
        treat_nans : str, default "delete rows"
            How to handle NaN values. Options: "delete rows", "delete columns", "impute".
        """
        if self.standardized_data is None:
            self.standardize()
        temp_frame = self.standardized_data.copy()

        # (optionally) select subset of columns:
        if feature_cols is not None:
            features = temp_frame.loc[:, feature_cols]
        else:
            features = temp_frame

        # Treat NaNs:
        if treat_nans == "delete rows":
            features.dropna(axis=0, inplace=True)
            print("Deleted each row where at least one feature contains NaNs.")
        elif treat_nans == "delete columns":
            features.dropna(axis=1, inplace=True)
            print("Deleted each column which contains NaNs.")
        elif treat_nans == "impute":
            features.fillna(features.mean(), inplace=True)
            features.dropna(axis=1, inplace=True)
            print("Filled NaNs with the respective column's mean.")

        # Define no. of components:
        if no_of_components is None:
            no_of_components = min(features.shape[0],
                                   features.shape[1]) - 1  # initial no. of components = smaller dimension - 1
        print(f"Calculating {no_of_components} principal components...")

        # Calculate principal components:
        pca = PCA(n_components=no_of_components)
        pca_features = pca.fit_transform(features)

        # pca_df currently unused
        pca_df = pd.DataFrame(data=pca_features, columns=[f'PC{i}' for i in range(1, no_of_components + 1)])
        if target_col is not None:  # If target variable column given, retain values in PCA dataframe
            pca_df[target_col] = temp_frame.reset_index().loc[:, target_col]
        pca_df.set_index(temp_frame.index, inplace=True)  # retain indices

        # save component loadings:
        loadings = pca.components_
        self.pca_loadings = pd.DataFrame(data=loadings.T, index=features.columns,
                                         columns=[f'PC{i}' for i in range(1, no_of_components + 1)])

    def plot_pca_loadings_heatmap(self, threshold=0, show_annotations=False,
                                  save_title=None) -> None:
        """
        Plots a heatmap containing the loadings of each principal component (PC).

        Parameters
        ----------
        threshold : float, default is 0
            Minimum value for loadings to be displayed.
        show_annotations : bool, default is False
            Whether to show precise values as annotations.
        save_title : str, optional
            Specific name for saving the heatmap if `save_fig` is True.
        """
        # calculate PCA:
        if self.pca_loadings is None:
            self.calculate_pca()

        # adjust plot size:
        plot_size = self.plot_size
        if len(self.pca_loadings.columns) >= 8:
            plot_size = (self.plot_size[0], len(self.pca_loadings.columns) * 1.2)

        # plot heatmap using seaborn package:
        fig, ax1 = plt.subplots(figsize=plot_size)
        ax = sns.heatmap(self.pca_loadings.T, cmap="mako_r",
                         yticklabels=list(self.pca_loadings.columns),
                         xticklabels=list(self.pca_loadings.index),
                         cbar_kws={'orientation': 'horizontal', 'location': 'top'},
                         mask=abs(self.pca_loadings.T) < threshold,  # define mask based on threshold
                         annot=show_annotations,
                         fmt='.1f',
                         ax=ax1)

        # formating:
        ax.set_aspect("equal")
        ax.set_title("Principal Component Loadings", loc='left')
        if threshold != 0:
            ax.set_title(f"Principal Component Loadings (absolutely above {threshold})", loc='left')
        plt.yticks(rotation=0)
        fig.tight_layout()

        # saving:
        if self.save_fig:
            if save_title is None:
                if self.plot_path is None:
                    raise Exception("Plot path needs to be defined!")
                save_title = self.plot_path / plotting.file_title("Loadings Heatmap")
            plt.savefig(save_title)

    def remove_features(self, features_to_remove: list) -> None:
        """
        Removes a list of features from the current dataframe.

        Parameters
        ----------
        features_to_remove : list
            List of features to remove from the dataset.

        Raises
        ------
        ValueError
            If any feature in `features_to_remove` is not in the dataset.
        """
        current_features = self.feature_list
        for feature in features_to_remove:
            if feature not in current_features:
                raise ValueError(f"Feature {feature} not in feature list.")

        self.data.drop(columns=features_to_remove, inplace=True)
        if self.standardized_data is not None:
            self.standardized_data.drop(columns=features_to_remove, inplace=True)
        print(f"Removed features {features_to_remove} from dataframe.")

    #############  analysis plot methods #############
    def plot_isomap_overview(self, title='Isomap', save_title=None, **embedding_kwargs) -> None:
        """
        Creates a lower-dimensional visualization of a high-dimensional feature space using Isomap.

        Parameters
        ----------
        title : str, default 'Isomap'
            Display title of the plot.
        save_title : str, optional
            Filename to save the plot if `save_fig` is True.
        embedding_kwargs : dict
            Additional arguments passed to the Isomap embedding method.
        """
        # we require standardized data:
        if self.standardized_data is None:
            self.standardize()
        embedding = Isomap(n_components=2, **embedding_kwargs).fit_transform(self.standardized_data)

        # Creating the scatterplot, two alternatives based on target_column:
        fig = plt.figure(figsize=self.plot_size)
        axt = fig.add_subplot(111)
        chart1 = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], ax=axt)

        # formating:
        plt.title(title, fontsize=24)
        axt.set_xticklabels([])
        axt.set_yticklabels([])
        fig.tight_layout()

        # saving:
        if self.save_fig:
            if save_title is None:
                if self.plot_path is None:
                    raise Exception("Plot path needs to be defined!")
                save_title = self.plot_path / plotting.file_title("Isomap Overview")
            plt.savefig(save_title)


class AutoEncoderModel:
    """
    Class to create and manage an autoencoder model.

    An autoencoder compresses the input into a lower dimension and reconstructs the output from this representation.
    This implementation uses TensorFlow/Keras Sequential models for building the encoder and decoder.
    """
    def __init__(self, keras_file=None, n_input_features: int = 200, n_latent_features: int = 10,
                 n_ae_layers: int = 5, encoder_activation: str = 'relu', decoder_activation: str = 'relu',
                 model_name: str = 'AutoEncoder', sparsity_constraint='L1', **regularizer_kwargs):
        """
        Initialize the autoencoder model from a Keras file or provided parameters.

        Parameters
        ----------
        keras_file : str, optional
            Path to a pre-trained Keras model file. If provided, other parameters are ignored.
        n_input_features : int, default=200
            Number of input features.
        n_latent_features : int, default=10
            Number of features in the latent representation.
        n_ae_layers : int, default=5
            Number of layers in the encoder/decoder.
        encoder_activation : str, default='relu'
            Activation function for encoder layers.
        decoder_activation : str, default='relu'
            Activation function for decoder layers.
        model_name : str, default='AutoEncoder'
            Name of the autoencoder model.
        sparsity_constraint : str, default='L1'
            Type of sparsity constraint ('L1', 'L2', 'L1+L2', or None).
        regularizer_kwargs : dict
            Additional keyword arguments for the regularizer.
        """
        self.encoder = self.decoder = self._autoencoder = None

        # define variables either from model file:
        if keras_file is not None:
            print(f'Loading AE model from {keras_file}.')
            # this calls a setter that derives other necessary properties:
            self.autoencoder = tf.keras.models.load_model(keras_file)

        # or from provided parameters:
        else:
            # define dimensionality of the three layers:
            layer_dims = np.linspace(n_input_features, n_latent_features, 1 + n_ae_layers, dtype="int")

            # save as internal variable:
            self.n_input_features = n_input_features
            self.n_latent_features = n_latent_features

            # define regularizer:
            #   regularizers add penalties to the model loss during training
            #   activity_regularizers are regularizers to be applied on the layer's output
            #   we utilize them for the latent_output_layer to achieve sparse, significant latent representations
            if sparsity_constraint is None:
                regularizer = None
            elif sparsity_constraint == 'L1':
                # L1 regularizer is calculated as:     l1_weight * reduce_sum(abs(layer_output))
                regularizer = regularizers.L1(**regularizer_kwargs)
            elif sparsity_constraint == 'L2':
                # L2 regularizer is calculated as:     l2_weight * reduce_sum(square(layer_output))
                regularizer = regularizers.L2(**regularizer_kwargs)
            elif sparsity_constraint == 'L1+L2':
                # these option sums both the above and takes two weight arguments
                regularizer = regularizers.L1L2(**regularizer_kwargs)
            else:
                raise ValueError(
                    f"'{sparsity_constraint}' sparsity_constraint is not defined. Options are: None, 'L1', 'L2' or 'L1+L2'.")

            # specify encoder
            self.encoder = Sequential([
                                          Input((layer_dims[0],), name='input_layer'), ] +
                                      [Dense(layer_dims[x], activation=encoder_activation) for x in
                                       range(1, n_ae_layers)] +  # 1st argument specifies dim. of output
                                      [Dense(layer_dims[-1], activation=encoder_activation,
                                             activity_regularizer=regularizer, name='latent_output_layer')
                                       ], name="encoder_layers")

            # specify decoder:
            self.decoder = Sequential([
                                          Input((layer_dims[-1],), name='latent_input_layer')] +
                                      [Dense(layer_dims[x], activation=decoder_activation) for x in
                                       range(1, n_ae_layers)[::-1]] +
                                      [Dense(layer_dims[0], name='output_layer'),
                                       # last layer requires linear activation for unrestrained output value range
                                       ], name='decoder_layers')

            # define input and output layers:
            input_layer = Input((n_input_features,), name="input_layer")
            latent_vector = self.encoder(input_layer)
            decoder_output = self.decoder(latent_vector)

            # consolidate model:
            self._autoencoder = Model(inputs=input_layer, outputs=decoder_output, name=model_name)
            self.pretrained = False

    def get_models(self):
        """
        Returns the encoder, decoder, and autoencoder models.

        Returns
        -------
        tuple
            (autoencoder model, encoder model, decoder model)
        """
        return self.autoencoder, self.encoder, self.decoder

    def train_autoencoder(self, input_frame, save_dir, optimizer='adam', epochs=10, verbose=0, batch_size=64):
        """
        Train the autoencoder using MSE reconstruction loss.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data for training.
        save_dir : str or Path
            Directory to save the trained model.
        optimizer : str, default='adam'
            Optimizer for training.
        epochs : int, default=10
            Number of training epochs.
        verbose : int, default=0
            Verbosity level.
        batch_size : int, default=64
            Size of training batches.
        """
        print('Training autoencoder...')
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        self.autoencoder.fit(x=input_frame, y=input_frame,
                             epochs=epochs, verbose=verbose,
                             shuffle=True, batch_size=batch_size,
                             validation_split=0.05)

        print(f'Saving AE model to {save_dir}...')

        # calculate final loss for file-name:
        final_loss = round(self.autoencoder.evaluate(input_frame, input_frame, verbose=False), 2)

        self.autoencoder.save(
            save_dir / plotting.file_title(f'AE Model features{self.n_latent_features} loss{final_loss}', '.keras',
                                           short=True))

        self.pretrained = True
        print(f'Done! Final loss: {final_loss}')

    def encode(self, input_frame):
        """
        Generate latent representation using the encoder model.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data to encode.

        Returns
        -------
        pd.DataFrame
            Latent representations.
        """
        assert (self.pretrained is True), 'Autoencoder needs to be pre-trained first.'

        # predict latent code:
        return pd.DataFrame(
            self.encoder.predict(input_frame, verbose=0),
            index=input_frame.index,
            columns=[f"Latent feature {i + 1}" for i in range(self.n_latent_features)])

    def decode(self, latent_input, feature_columns):
        """
        Reconstruct features from latent representation using the decoder model.

        Parameters
        ----------
        latent_input : pd.DataFrame
            Latent representations to decode.
        feature_columns : list
            Names of the reconstructed features.

        Returns
        -------
        pd.DataFrame
            Reconstructed features.
        """
        assert (self.pretrained is True), 'Autoencoder needs to be pre-trained first.'

        # predict features from latent code:
        return pd.DataFrame(
            self.decoder.predict(latent_input, verbose=0),
            index=latent_input.index,
            columns=feature_columns)

    def load_weights(self, filepath):
        """
        Load pre-trained weights into the autoencoder.

        Parameters
        ----------
        filepath : str
            Path to the weights file.
        """
        self.autoencoder.load_weights(filepath)
        self.pretrained = True
        print(f'Imported AE weights from {filepath}!')

    @property
    def autoencoder(self):
        """ Autoencoder Property with input handling """
        return self._autoencoder

    @autoencoder.setter
    def autoencoder(self, model):
        """
        Set the autoencoder model and update encoder/decoder references.

        Parameters
        ----------
        model : tf.keras.Model
            Autoencoder model.
        """
        self._autoencoder = model
        self.encoder = model.get_layer('encoder_layers')
        self.decoder = model.get_layer('decoder_layers')
        self.n_input_features = self.encoder.get_config()['build_input_shape'][1]
        self.n_latent_features = self.decoder.get_config()['build_input_shape'][1]
        self.pretrained = True


class ClusteringLayer(Layer):
    """
    Clustering layer that converts input sample to soft cluster assignments (calculated with Student's t-distribution).

    Inspired by https://github.com/FlorentF9/DeepTemporalClustering/blob/master/TSClusteringLayer.py while adjusting
    for our purpose, which is clustering a feature space not a time-series.

    # Arguments:
        n_clusters: number of clusters
        weights: represent initial cluster centers
    # Input shape:
        2D array with shape: (n_samples, n_features) containing feature space
    # Output shape:
        2D array with shape: (n_samples, n_clusters) containing soft cluster assignments
    """

    def __init__(self, n_clusters, weights=None, **kwargs):
        # call baseclass __init__:
        super(ClusteringLayer, self).__init__(**kwargs)

        # initialise:
        self.n_clusters = n_clusters
        self.initial_weights = weights

        # From documentation: These objects enable the layer to run input compatibility checks for input structure, input rank, input shape, and input dtype for the first argument of Layer.call.
        self.input_spec = InputSpec(ndim=2)

        self.centroids = None
        self.built = False  # control boolean to check whether build() was called before call()

    def build(self, input_shape):
        assert len(input_shape) == 2  # compatibility check
        input_dim = input_shape[1]  # no of features

        # definition see __init__ above, extended by shape and dtype
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))

        # add weights (cluster assignments) with glorot_uniform initialiser (see documentation for more info).
        self.centroids = self.add_weight(shape=(self.n_clusters, input_dim),
                                         initializer='glorot_uniform', name='cluster_centers')

        # adjust such weights if initial_weights are defined:
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights

        self.built = True

    def call(self, inputs, **kwargs):
        '''
        Calculate soft cluster assignments with Student t-distribution.

        Arguments:
            inputs: input sequence with shape (n_samples, n_features), as defined above
        Return:
            q: soft cluster assignments (probability of assigning point to respective cluster)
        '''
        assert self.built == True  # (happens automatically)

        # utilize the Keras backend (K) for calculation
        # axis defines which input dimension to apply the calculation on and follows python indexing rules
        distance = K.sqrt(
            K.sum(
                K.square(
                    K.expand_dims(inputs, axis=1) - self.centroids),
                # expand dimension of input (n_samples, n_features) to fit n_clusters dimension of self.clusters (n_samples, n_clusters, n_features)
                axis=2))
        # sqrt of
        #    sum over all features (axis=2 after dim. expansion) of
        #       squared distance of
        #          feature_values - cluster_centroid_values (self.clusters)

        # formula for q based on Student's t-distribution as described in "Unsupervised Deep Embedding for Clustering Analysis" paper from Xie et al.
        q = 1.0 / (1.0 + K.square(distance))

        # divide q by sum over all clusters' (axis=1) q
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1))
        # transpose q so that shapes are (n_clusters, n_samples) / (n_clusters); then transpose again to return shape (n_samples, n_clusters)

        return q

    def get_config(self):
        config = {'n_clusters': self.n_clusters}  # this class's config
        base_config = super(ClusteringLayer, self).get_config()  # fetch config from baseclass
        return dict(list(base_config.items()) + list(config.items()))


class DECModel:
    """
    Deep Embedded Clustering Model.

    This model integrates an autoencoder for feature extraction and a clustering layer for unsupervised clustering.
    """
    def __init__(self, keras_file=None,
                 n_clusters=3, n_input_features=200, n_latent_features=10, **ae_kwargs):
        """
        Initializes the DEC Model either from a saved keras file or by constructing a new model.

        Parameters
        ----------
        keras_file : str or None
            Path to a pre-trained keras model file. If None, a new model will be constructed.
        n_clusters : int
            Number of clusters for the clustering layer (default is 3).
        n_input_features : int
            Number of input features for the autoencoder (default is 200).
        n_latent_features : int
            Number of latent features for the autoencoder (default is 10).
        ae_kwargs : dict
            Additional arguments to pass to the autoencoder model.
        """
        if keras_file is not None:  # construct model from keras file
            self.model = tf.keras.models.load_model(keras_file)

        else:  # construct model from parameters
            self.n_clusters = n_clusters
            self.n_input_features = n_input_features
            self.n_latent_features = n_latent_features
            self.pretrained = False

            self._autoencoder, self.encoder, self.decoder = AutoEncoderModel(n_input_features=self.n_input_features,
                                                                             n_latent_features=self.n_latent_features,
                                                                             **ae_kwargs).get_models()
            clustering_layer = ClusteringLayer(n_clusters=self.n_clusters, name='cluster_layer')(self.encoder.output)

            # Consolidate model. Cluster_layer is called with latent code.
            self._model = Model(inputs=self.autoencoder.input,
                                outputs=[self.autoencoder.output, clustering_layer],
                                name="DECModel")

    def __repr__(self):
        """ String representation (for e.g. Jupyter console output) """
        self.model.summary(expand_nested=False)
        return f"DEC Model with {self.n_clusters} clusters, {self.n_input_features} input- and {self.n_latent_features} latent features."

    @property
    def autoencoder(self):
        """
        Retrieves the autoencoder model.

        Returns
        -------
        keras.Model
            The autoencoder model.
        """
        return self._autoencoder

    @autoencoder.setter
    def autoencoder(self, model):
        """
        Sets the autoencoder model and updates dependent properties.

        Parameters
        ----------
        model : keras.Model
            The autoencoder model.
        """
        self._autoencoder = model
        # also initialise callable submodels:
        self.encoder = model.get_layer('encoder_layers')
        self.decoder = model.get_layer('decoder_layers')

        # set other variables:
        self.n_input_features = self.encoder.get_config()['build_input_shape'][1]
        self.n_latent_features = self.decoder.get_config()['build_input_shape'][1]
        self.pretrained = True

    @property
    def model(self):
        """
        Retrieves the DEC model, which includes the autoencoder and clustering layer.

        Returns
        -------
        keras.Model
            The DEC model.
        """
        return self._model

    @model.setter
    def model(self, model):
        """
        Sets the DEC model and updates dependent properties.

        Parameters
        ----------
        model : keras.Model
            The DEC model.
        """
        self._model = model
        # also initialise callable submodels:
        self.encoder = model.get_layer('encoder_layers')
        self.decoder = model.get_layer('decoder_layers')
        self._autoencoder = Model(inputs=self.encoder.input,
                                  outputs=self.decoder.output, name='AutoEncoder')

        # set other variables:
        self.n_clusters = self.model.get_layer('cluster_layer').get_config()['n_clusters']
        self.n_input_features = self.encoder.get_config()['build_input_shape'][1]
        self.n_latent_features = self.decoder.get_config()['build_input_shape'][1]
        self.pretrained = True

    def pretrain_autoencoder(self, input_frame, save_dir, optimizer='adam', epochs=10, verbose=0, batch_size=64):
        """
        Pre-trains the autoencoder using MSE reconstruction loss.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data for training.
        save_dir : Path
            Directory to save the trained autoencoder.
        optimizer : str
            Optimizer for training (default is 'adam').
        epochs : int
            Number of training epochs (default is 10).
        verbose : int
            Verbosity level (default is 0).
        batch_size : int
            Batch size for training (default is 64).
        """
        print('Pretraining autoencoder...')
        self.autoencoder.compile(optimizer=optimizer, loss="mse")
        self.autoencoder.fit(x=input_frame, y=input_frame,
                             epochs=epochs, verbose=verbose,
                             shuffle=True, batch_size=batch_size,
                             validation_split=0.05)

        print(f'Saving AE model to {save_dir}...')

        # calculate final loss for file-name:
        final_loss = round(self.autoencoder.evaluate(input_frame, input_frame, verbose=False), 2)

        self.autoencoder.save(
            save_dir / plotting.file_title(f'AE Model features{self.n_latent_features} loss{final_loss}', '.keras',
                                           short=True))

        self.pretrained = True
        print(f'Done! Final loss: {final_loss}')

    @property
    def centroids(self):
        """
        Retrieves the cluster centroids (weights of the clustering layer).

        Returns
        -------
        np.ndarray
            Cluster centroids.
        """
        return self.model.get_layer(name='cluster_layer').get_weights()

    def compile(self, cluster_loss_weight=1.0, optimizer='adam'):
        """
        Compiles the DEC model with specified loss weights and optimizer.

        Parameters
        ----------
        cluster_loss_weight : float
            Weight for the clustering loss (default is 1.0).
        optimizer : str
            Optimizer for training (default is 'adam').
        """
        self.model.compile(optimizer=optimizer,
                           loss=[losses.MeanSquaredError(), losses.KLDivergence()],
                           loss_weights=[1.0, cluster_loss_weight])

    def encode(self, input_frame):
        """
        Encodes input data to its latent representation using the encoder.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data to encode.

        Returns
        -------
        pd.DataFrame
            Latent representations of the input data.
        """
        assert (self.pretrained is True), 'Autoencoder needs to be pre-trained first.'

        # predict latent code:
        return pd.DataFrame(
            self.encoder.predict(input_frame, verbose=False),
            index=input_frame.index,
            columns=[f"Latent feature {i + 1}" for i in range(self.n_latent_features)])

    def decode(self, latent_input, feature_columns):
        """
        Decodes latent representations back to feature space using the decoder.

        Parameters
        ----------
        latent_input : pd.DataFrame
            Latent representations to decode.
        feature_columns : list
            Names of the feature columns for the decoded output.

        Returns
        -------
        pd.DataFrame
            Decoded feature data.
        """
        assert (self.pretrained is True), 'Autoencoder needs to be pre-trained first.'

        # predict features from latent code:
        return pd.DataFrame(
            self.decoder.predict(latent_input, verbose=False),
            index=latent_input.index,
            columns=feature_columns)

    def init_cluster_weights(self, input_frame):
        """
        Initializes the cluster weights in the clustering layer using K-means.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data to calculate initial cluster centers.
        """
        print('Initialising cluster weights...')
        latent_code = self.encode(input_frame)
        # Calculate K-Means clustering:
        centroids, distortion = kmeans(latent_code, self.n_clusters)
        # centroids are the coordinates of the cluster centers for each cluster in latent variable units

        # cluster configuration (centroids) is saved as the cluster_layer's weights
        self.model.get_layer(name='cluster_layer').set_weights([centroids])
        print('Done!')

    def predict(self, input_frame):
        """
        Predicts cluster assignments for input data.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data to assign to clusters.

        Returns
        -------
        np.ndarray
            Predicted cluster assignments.
        """
        # selecting second model output (first is decoded representation)
        soft_assignments = self.model.predict(input_frame, verbose=0)[1]
        # return highest confidence assigment for each sample:
        return soft_assignments.argmax(axis=1)

    def load_ae_weights(self, filepath):
        """
        Loads pre-trained autoencoder weights.

        Parameters
        ----------
        filepath : str
            Path to the weights file.
        """
        self.autoencoder.load_weights(filepath)
        self.pretrained = True
        print(f'Imported AE weights from {filepath}!')

    def load_weights(self, filepath):
        """
        Loads the DEC model weights, including clustering and autoencoder weights.

        Parameters
        ----------
        filepath : str
            Path to the weights file.
        """
        self.model.load_weights(filepath)
        self.pretrained = True
        print(f'Imported DEC weights from {filepath}!')

    @staticmethod
    def target_distribution(soft_assignments):
        """
        Calculates the target distribution for clustering optimization.

        Parameters
        ----------
        soft_assignments : np.ndarray
            Soft cluster assignments.

        Returns
        -------
        np.ndarray
            Adjusted target distribution.
        """
        soft_cluster_frequency = soft_assignments.sum(axis=0)  # calculate f
        point_weight_per_cluster = soft_assignments ** 2 / soft_cluster_frequency  # numerator of p
        # this weight is based on high-confidence assignments (squared q)
        # and normalized by the frequency of the respective cluster (to equalize the impact of small and large clusters)

        cumulated_point_weight = point_weight_per_cluster.sum(axis=1)  # denominator of p
        # axis=1 sums each point's weigth over all clusters

        return (point_weight_per_cluster.T / cumulated_point_weight).T  # calculate p
        # why transpose is used here see https://numpy.org/doc/stable/user/basics.broadcasting.html

    @staticmethod
    def plot_log_evaluation(log_file_path, smoothing_window=50, log_save_step=10):
        """
        Plots training log for evaluation metrics over epochs.

        Parameters
        ----------
        log_file_path : str
            Path to the log file.
        smoothing_window : int
            Window size for smoothing (default is 50).
        log_save_step : int
            Step size for saving logs (default is 10).
        """
        log = pd.read_csv(log_file_path)
        if smoothing_window != 1:
            print(f'Log-values are smoothed utilizing a rolling average over {smoothing_window} epochs.')

        # metrics to analyse, tuples will be analysed in one plot:
        metrics = [('L', 'MSE_L', 'KLD_L'), 'Label_Changes', 'Silh_Coeff', 'Dunn_Index']
        optim_directions = [('min', 'min', 'min'), 'min', 'max', 'max']

        # prepare plots:
        fig, axs = plt.subplots(len(metrics), 1, figsize=(2.5 * len(metrics), len(metrics) * 2.5))
        for ind, (metric, direction) in enumerate(zip(metrics, optim_directions)):
            # calculate rolling average:
            smoothed = log.loc[:, metric].rolling(window=smoothing_window).mean()

            # print optimal value and plot lineplot:
            analysis.print_optimal_index(smoothed, metric, direction)
            sns.lineplot(ax=axs[ind], data=smoothed)

            # remove ax ticks for all but lowest plot:
            if ind != len(metrics) - 1:
                axs[ind].set_xticklabels([])
            else:
                axs[ind].set_xlabel(f'Training epoch [x{log_save_step}]')
        fig.tight_layout()

    def isomap_plot(self, input_data: pd.DataFrame,
                    title='Isomap Clustering Overview', save_title=None,
                    seaborn_palette='bright', plot_size=(16, 10),
                    legend_cols=1, legend_pos=(1.2, 0.78),
                    hidden=False) -> None:
        """
        Generates a 2D visualization of the high-dimensional feature space using Isomap.

        Parameters
        ----------
        input_data : pd.DataFrame
            High-dimensional input data.
        title : str
            Title of the plot (default is 'Isomap Clustering Overview').
        save_title : str or None
            Path to save the plot (default is None).
        seaborn_palette : str
            Seaborn palette for coloring clusters (default is 'bright').
        plot_size : tuple
            Size of the plot (default is (16, 10)).
        legend_cols : int
            Number of columns for the legend (default is 1).
        legend_pos : tuple
            Position of the legend (default is (1.2, 0.78)).
        hidden : bool
            Whether to close the plot after saving (default is False).
        """
        # calculate cluster assignments:
        hard_assignments = self.predict(input_data)

        # calculate isomap representation:
        try:
            embedding = Isomap(n_components=2).fit_transform(self.encode(input_data))
        except UserWarning as err:
            print('Warning when creating isomap representation:', err)
            print('Plotting again with an increased number of neighbors (now 15 instead of 5).')
            embedding = Isomap(n_components=2, n_neighbors=15).fit_transform(self.encode(input_data))

        # Creating the scatterplot, two alternatives based on target_column:
        fig = plt.figure(figsize=plot_size)
        axt = fig.add_subplot(111)
        chart1 = sns.scatterplot(x=embedding[:, 0], y=embedding[:, 1], hue=hard_assignments,
                                 palette=seaborn_palette, ax=axt)

        # formating of legend:
        sns.move_legend(
            chart1, 'lower right',
            ncol=legend_cols, title='Cluster', frameon=True,
            bbox_to_anchor=legend_pos
        )

        # formating:
        plt.title(title, fontsize=24)
        axt.set_xticklabels([])
        axt.set_yticklabels([])
        fig.tight_layout()

        # saving:
        if save_title is not None:
            plt.savefig(save_title)
        if hidden:
            plt.close()

    def fit(self, X_train, log_dir, save_dir, epochs=500, eval_epochs=10, save_epochs=100,
            verbose=0, shuffle=True, batch_size=64, tol=0.001, patience=10, visualize_dir=None,
            freeze_autoencoder=False):
        """
        Trains the DEC model.

        Parameters
        ----------
        X_train : pd.DataFrame
            Training data.
        log_dir : Path
            Directory to save logs.
        save_dir : Path
            Directory to save model checkpoints.
        epochs : int
            Total number of epochs for training (default is 500).
        eval_epochs : int
            Interval for evaluation (default is 10).
        save_epochs : int
            Interval for saving model checkpoints (default is 100).
        verbose : int
            Verbosity level (default is 0).
        shuffle : bool
            Whether to shuffle data each epoch (default is True).
        batch_size : int
            Batch size for training (default is 64).
        tol : float
            Tolerance for stopping criterion (default is 0.001).
        patience : int
            Number of epochs to wait before stopping (default is 10).
        visualize_dir : Path or None
            Directory for saving visualizations (default is None).
        freeze_autoencoder : bool
            Whether to freeze the autoencoder during training (default is False).
        """
        assert (self.pretrained is True), 'Autoencoder needs to be pre-trained first.'

        # specify which layers to train:
        if freeze_autoencoder:
            self.encoder.trainable = False
            self.decoder.trainable = False
            self.model.get_layer("cluster_layer").trainable = True
            print("Only training cluster layer.")
        else:
            self.encoder.trainable = True
            self.decoder.trainable = True
            self.model.get_layer("cluster_layer").trainable = True

        print(f'Training model for {epochs} epochs...')

        logfile = open(log_dir / plotting.file_title("DEC LOG", ".csv"), 'w')
        fieldnames = ['epoch', 'L', 'MSE_L', 'KLD_L', 'Label_Changes', 'Silh_Coeff', 'Dunn_Index', 'N_Clusters']
        logwriter = csv.DictWriter(logfile, fieldnames)
        logwriter.writeheader()

        prev_hard_assignments = None
        patience_count = 0

        for epoch in tqdm(range(epochs)):
            soft_assignments = self.model.predict(X_train, verbose=0)[1]
            target_dist = self.target_distribution(soft_assignments)

            # to be implemented:
            if epoch % eval_epochs == 0:
                # Initialise log dictionary
                logdict = dict(epoch=epoch)

                # calculate current assignments
                hard_assignments = soft_assignments.argmax(axis=1)

                # calculate loss metrics:
                logdict['L'], logdict['MSE_L'], logdict['KLD_L'] = self.model.evaluate(x=X_train,
                                                                                       y=[X_train, target_dist],
                                                                                       batch_size=batch_size,
                                                                                       verbose=False)

                # effective number of clusters:
                logdict['N_Clusters'] = len(np.unique(hard_assignments))

                # calculate other metrics:
                logdict['Silh_Coeff'], logdict['Dunn_Index'] = self.calculate_cluster_metrics(X_train, hard_assignments,
                                                                                              batch_size)

                # calculate assignment changes:
                assignment_changes = 1  # first iteration assumed with 100% new assignments
                if prev_hard_assignments is not None:
                    # calculates percentage of assignments that have changed from last evaluation
                    assignment_changes = np.sum(hard_assignments != prev_hard_assignments).astype(np.float32) / \
                                         hard_assignments.shape[0]
                logdict['Label_Changes'] = assignment_changes
                prev_hard_assignments = hard_assignments

                # log loss metrics:
                logwriter.writerow(logdict)

                # check termination criterion
                if epoch > 0 and assignment_changes < tol:
                    patience_count += 1  # increase patience count
                    print(
                        f'Assignment changes {assignment_changes} below {tol}. Patience: {patience_count} of {patience}.')
                    if patience_count >= patience:
                        print('Reached max. patience. Training finished!')
                        break
                else:
                    patience_count = 0  # reset patience when tol-criterion met again

            if epoch % save_epochs == 0:
                # isomap plot option to visualise training procedure:
                if visualize_dir is not None:
                    title = f'Isomap Cluster Overview Iteration {epoch}'
                    self.isomap_plot(input_data=X_train, title=title,
                                     save_title=visualize_dir / plotting.file_title(title, ".png"),
                                     hidden=True)

                print(f'Saving interim DEC model to {save_dir}...')
                self.model.save(
                    save_dir / plotting.file_title(f'Epoch{epoch} DEC Model', '.keras', short=False))

            # Train for one epoch:
            self.model.fit(x=X_train,
                           y=[X_train, target_dist],
                           epochs=1, verbose=verbose,
                           shuffle=shuffle,
                           batch_size=batch_size)

        logfile.close()
        print('Reached maximum number of epochs. Training finished!')

        # calculate final stats for filename:
        soft_assignments = self.model.predict(X_train, verbose=0)[1]
        target_dist = self.target_distribution(soft_assignments)
        hard_assignments = soft_assignments.argmax(axis=1)
        final_loss = round(self.model.evaluate(x=X_train,
                                               y=[X_train, target_dist],
                                               batch_size=batch_size, verbose=False)[0],
                           2)
        final_silh, final_di = self.calculate_cluster_metrics(X_train, hard_assignments)

        print(f'Saving final DEC model to {save_dir}...')
        self.model.save(save_dir / plotting.file_title(
            f'DEC Model clusters{self.n_clusters} loss{final_loss} silh{final_silh} dunn{final_di}',
            '.keras', short=True))

        # save visualization to gif:
        if visualize_dir is not None:
            # select saved png files:
            today = plotting.file_title("", short=True)[:8]
            file_titles = [visualize_dir / f for f in os.listdir(visualize_dir) if f[:8] == today and f[-4:] == ".png"]
            file_titles.sort()
            # save such as gif:
            gif_path = visualize_dir / plotting.file_title("Animated Isomap Training Progress", ".gif", short=True)
            ims = [imageio.v2.imread(f) for f in file_titles]
            imageio.mimwrite(gif_path, ims)
            print(f'Saved training visualization-pngs and -gif to {str(gif_path)}!')

    @staticmethod
    def calculate_cluster_metrics(input_frame, hard_assignments, sample_size=None):
        """
        Calculates clustering evaluation metrics.

        Parameters
        ----------
        input_frame : pd.DataFrame
            Input data.
        hard_assignments : np.ndarray
            Cluster assignments.
        sample_size : int or None
            Number of samples for silhouette calculation (default is None).

        Returns
        -------
        tuple
            Silhouette score and Dunn index.
        """
        try:
            silh_score = silhouette_score(input_frame, hard_assignments, sample_size=sample_size)
            dunn_index = dunn_fast(input_frame.to_numpy(), hard_assignments)
        except ValueError as err:
            # handling if cluster number to low:
            if len(np.unique(hard_assignments)) == 1:
                print('All points assigned to only 1 cluster currently. Metrics therefore cannot be calculated (NaN).')
                silh_score = dunn_index = np.nan
            # otherwise raise error:
            else:
                raise ValueError(err)
        return silh_score, dunn_index


# auxiliary function:
def dunn_fast(points, labels):
    """
    Dunn index - FAST (using sklearn pairwise euclidean_distance function)
    From Joaquim Viegas (JQM_CV)

    Parameters
    ----------
    points : np.ndarray
        Array of points in feature space.
    labels : np.ndarray
        Cluster labels for each point.

    Returns
    -------
    float
        Dunn index value.
    """

    def __delta_fast(ck, cl, distances):
        values = distances[np.where(ck)][:, np.where(cl)]
        values = values[np.nonzero(values)]

        return np.min(values)

    def __big_delta_fast(ci, distances):
        values = distances[np.where(ci)][:, np.where(ci)]

        return np.max(values)

    distances = euclidean_distances(points)
    ks = np.sort(np.unique(labels))

    deltas = np.ones([len(ks), len(ks)]) * 1000000
    big_deltas = np.zeros([len(ks), 1])

    l_range = list(range(0, len(ks)))

    for k in l_range:
        for l in (l_range[0:k] + l_range[k + 1:]):
            deltas[k, l] = __delta_fast((labels == ks[k]), (labels == ks[l]), distances)

        big_deltas[k] = __big_delta_fast((labels == ks[k]), distances)

    di = np.min(deltas) / np.max(big_deltas)
    return di