function [eigenvectors,scores,mu,pca_mse,pca_psnr] = PCA_data(trainingImages, components)
%PCA_data Calculate all data related to PCA (eigenvectors, projected matrix, mean,
% mse error and psnr error

[eigenvectors,scores, latent, tsq, exp, mu] = pca(trainingImages', 'NumComponents', components);
Xhat = scores * eigenvectors' + repmat(mu, size(trainingImages,2), 1); 
pca_mse = immse(trainingImages',Xhat);
pca_psnr = psnr(trainingImages',Xhat);
end

