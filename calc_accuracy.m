function [accuracy] = calc_accuracy(new_class, orig_class)
%calc_accuracy Calculate the accuracy of the given data

accuracy = sum(new_class == orig_class)/numel(new_class);

end

