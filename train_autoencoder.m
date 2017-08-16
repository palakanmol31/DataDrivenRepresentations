function [autoenc, auto_mse, auto_psnr] = train_autoencoder(trainingImages, components)
%train_autoencoder Train autoencoder and return the reconstructed mse error
%and psnr error

autoenc = trainAutoencoder(trainingImages,components,'MaxEpoch',400,...
           'L2WeightRegularization',0.004,...
           'SparsityRegularization',4,...
           'SparsityProportion',0.15);
      
  xReconstructed = predict(autoenc,trainingImages);
  
  auto_mse = immse(trainingImages, xReconstructed);
  auto_psnr = psnr(trainingImages,xReconstructed);

end

