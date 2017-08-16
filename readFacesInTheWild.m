function [final_matrix] = readFacesInTheWild(metaData)
%readFacesInTheWild Read the Faces in the Wild Dataset

a = 86 * 86; % Dimensions of Image
 for i = 1 : numel(metaData)
im = rgb2gray(imread(strcat('faceData/',metaData{i}.fileName)));
 im = reshape(im,1,size(im,1)*size(im,2)) ;
 im = im2double(im);
 diff = a - size(im,2); %To handle missing data
 size(im, 2)
 im(1,size(im,2)+1:a) = 0;  %%To handle missing data (setting it to 0)
 final_matrix(i,:) = im; 
end

end

