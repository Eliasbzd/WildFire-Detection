import torch


class PCA(torch.nn.Module):
    def __init__(self, PCA_limit=None, center=True):
        """
        PCA (Principal Component Analysis) class constructor.
        
        .. todo :: Add the Import method to build the torch.Tensor with the features and their values
                   using the class in the other files.

        Parameters
        ----------
        PCA_limit : scalar
            -- number of principal component kept at the end
        center : booleen
            -- Whether to center the feature_value. Default is True.
        """
        self.center = center
        self.PCA_limit = PCA_limit

    def SVD(self, feature_value):
        """
        Fit the PCA model to the input feature_value.
        Computes and stores the principal components and explained variance.
        
        Parameters
        ----------
        x : torch.Tensor
            -- Input feature_value tensor

        """
        if self.center:
            feature_value = feature_value - torch.mean(feature_value, dim=0)

        n_samples, n_features = feature_value.shape
        self.n_samples = n_samples

        # self.PCA_limit = n_features
        if self.PCA_limit == None:
            self.PCA_limit = min(n_samples, n_features)
        U, S, V = torch.linalg.svd(feature_value)
        self.components = V.T[:, :self.PCA_limit]
        self.explained_variance = torch.mul(
            S[0:self.PCA_limit], S[0:self.PCA_limit]) / (n_samples - 1)

    def extract(self, feature_value):
        """
        Reduce the dimensionality of the input feature_value by projecting it onto the principal components.
        
        Parameters
        ----------
        feature_value : torch.Tensor
            -- Input feature_value tensor.
        
        Returns
        -------
         not : torch.Tensor
            -- Transformed feature_value using the principal components.

        """
        if self.center:
            feature_value = feature_value - torch.mean(feature_value, dim=0)
        return torch.matmul(feature_value, self.components)