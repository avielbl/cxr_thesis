function I = loadNPrepImg(im_path)
% loadNPrepImg performs loading for CXR
% This includes:
%   1. Verify input is a valid DICOM file
%   2. Flip intensity of negative images
% Usage:
%       out = loadNPrepImg(im_path);
% Inputs:
%       im_path: full path to the dicom file
% Outputs:
%       out: single type image

if (~isdicom(im_path))
    I = [];
    return;
end
I = single(dicomread(im_path));
[~, dinfo] = evalc('dicominfo(im_path)');

% MONOCHROME1 indicate 0=white so in such cases take a negative image
if strcmp(dinfo.PhotometricInterpretation, 'MONOCHROME1')
    maxI = max(I(:));
    I = (2^ceil(log2(maxI-1)))-1-I;
end
end