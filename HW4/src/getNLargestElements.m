function [largestNElements LargestNIdx] = getNLargestElements(A, n)
     [DSorted DIdx] = sort(A, 'descend');
     largestNElements = DSorted(1:n);
     LargestNIdx = DIdx(1:n);
end