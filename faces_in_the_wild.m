%Run this script to take the Faces in the Wild dataset and then get the MSE and PSNR Errors for the images
%reconstructed by both PCA and the Autoencoder

data = importdata('faceData/FacesInTheWild.mat'); 
metaData = data.metaData;
final_matrix = readFacesInTheWild(metaData);
components = 10;

[eigenvectors,scores,mu,pca_mse,pca_psnr] = PCA_data(final_matrix, components);

[autoenc, auto_mse, auto_psnr] = train_autoencoder(final_matrix', components);