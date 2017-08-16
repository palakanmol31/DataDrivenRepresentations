%This script gets the MNSIT Dataset as input and calculates the MSE and
%PSNR Errors for the training and testing images regenarated by PCA and
%Autoencoders.
%This script also trains the KNN Classifier by training it with
%trainingImages and then classifying the given testingImages. We calculate
%the accuracy of the classification and save it in the variables.


training_label_vector = loadMNISTLabels('train-labels-idx1-ubyte');
testing_label_vector = loadMNISTLabels('t10k-labels-idx1-ubyte');
trainingImages = loadMNISTImages('train-images-idx3-ubyte');
testingImages = loadMNISTImages('t10k-images-idx3-ubyte');

components = 50;
knn_neighbors = 5;

[eigenvectors,scores,mu,pca_mse,pca_psnr] = PCA_data(trainingImages, components);   %PCA data for training set
[test_coeff, test_score, test_mu,pca_test_mse,pca_test_psnr] = PCA_data(testingImages, components); %%PCA data for training set

[pca_knn, test_class] = knn_classify(scores, training_label_vector, knn_neighbors, test_score); %Classify PCA Training data
pca_acc = calc_accuracy(test_class,testing_label_vector);   %Calcuate PCA Accuracy

[autoenc, auto_mse, auto_psnr] = train_autoencoder(trainingImages, components); %Autoencoder Training Data

x_test_Reconstructed = predict(autoenc, testingImages);
encoded = encode(autoenc, trainingImages);
test_encode = encode(autoenc, testingImages);
[auto_knn, auto_class] = knn_classify(encoded', training_label_vector, knn_neighbors, test_encode');    %Train KNN Classifier


auto_test_mse = immse(testingImages, x_test_Reconstructed);  %Autoencoder testing data
auto_test_psnr = psnr(testingImages, x_test_Reconstructed);

auto_acc = calc_accuracy(auto_class, testing_label_vector); %Autoencoder Accuracy

