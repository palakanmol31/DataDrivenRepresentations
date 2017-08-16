function [knn, predict_class] = knn_classify(trainingData, training_label_vector, knn_neighbors, predictData)
%knn_classify Classifies the given training data using knn classifier with
%the number of k as knn_neighbors and returns the predicted class of
%predictData

knn = fitcknn(trainingData,training_label_vector,'NumNeighbors',knn_neighbors);
predict_class = predict(knn, predictData);

end

