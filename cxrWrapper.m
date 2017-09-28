function cxrWrapper(varargin)
% cxrWrapper is the matlab script that runs the processing of a
% single chest x-ray series
% Usage:
%       cxrWrapper [-v] in_folder out_folder view_position
% Inputs:
%       in_folder:      path to input folder containing the dicom image/s for the scan to be processed
%       out_folder:     path to output folder where the results of the processing will
%                       be saved as an xml file : output.xml
%       view_position:  string indicating series view position (either ‘AP’ or ‘PA’)

ALG_VER = '0.3 beta7';
BUILD_DATE = '20/11/2016';
if ~isdeployed
    addpath LungsSegmentation
    addpath ptxClassifier
    addpath AbnormalitiesClassifier
    addpath AbnormalitiesClassifier\Common\libsvm-mat-3.0-1
    addpath export_fig
    addpath ../HelpFunctions
    addpath ../matconvnet1.0beta20/matlab
    addpath ../xml_toolbox
    vl_setupnn; % Initialize MatConvNet
else
    %#function dicominfo dicomread showImageWithContour
end

if nargin() == 1 && strcmp(varargin(1),'-v')
    disp(['Algorithm Version: ' ALG_VER]);
    disp(['Build date: ' BUILD_DATE]);
    disp('Compiled with: Version 8.6.0.267246, Release 2015b');
    return;
else
    in_folder = varargin{1};
    out_folder = varargin{2};
    if nargin == 3
        view_position = varargin{3};
    else
        view_position = 'NA';
    end
end

% Create log file
slashInd = strfind(in_folder,'\');
switch numel(slashInd)
    case 0
        folderName = in_folder;
    case 1
        folderName = in_folder(slashInd + 1 : end);
    otherwise
        folderName = in_folder(slashInd(end-1)+1 : slashInd(end)-1);
end
if ~exist(out_folder,'file')
    mkdir(out_folder)
end
logFullPath = fullfile(out_folder,['Log_' folderName '.txt']);
fid_log = fopen(logFullPath,'w+');

% Write version data to log
add_line_to_log(fid_log,'INFO',['Algorithm Version: ' ALG_VER]) % Write to log
add_line_to_log(fid_log,'INFO',['Build date: ' BUILD_DATE]) % Write to log
if isdeployed
    [vmj, vmn] = mcrversion;
    add_line_to_log(fid_log,'INFO',['MCR version: ' num2str(vmj), '.', num2str(vmn)]) % Write to log
end
add_line_to_log(fid_log,'INFO',['View Position: ' view_position]) % Write to log


% Generate name-sorted list of dicom files in input folder
[files_list, num_files] = getFilesList(in_folder);
if num_files == 0
    add_line_to_log(fid_log, 'ERROR', ['No files found in input folder ', in_folder]);
    return;
end
add_line_to_log(fid_log, 'INFO', ['Found ', num2str(num_files), ' files in the input folder']);
if ~isdeployed
    instNum = zeros(size(files_list));
    for i=1:num_files
        file_path = files_list(i).fullPath;
        if ~isdicom(file_path) %skip files other then DICOM (commonly desktop.ini is found when not deployed)
            files_list(i) = [];
            num_files = num_files - 1;
            instNum(i) = [];
            continue;
        end
        [~, info] = evalc('dicominfo(file_path)');
        instNum(i) = info.InstanceNumber;
    end
    
    if ~issorted(instNum)
        [~, sortedInd] = sort(instNum);
        files_list = files_list(sortedInd);
    end
end

% cxrAlgUnit options
options = [];
try
    % Initialize variables.
    filename = 'cxr.ini';
    delimiter = '=';
    formatSpec = '%s%s%[^\n]';
    fileID = fopen(filename,'r');
    if fileID ~= -1
        scan = textscan(fileID, formatSpec, 'Delimiter', delimiter,  'ReturnOnError', false);
        fclose(fileID);
        options = cell2struct(scan{:,2}, scan{:,1}, 1);
        fields = fieldnames (options);
        for i=1 : length(fields)
            options.(fields{i}) = str2num(options.(fields{i}));
        end
    end
catch
    add_line_to_log(fid_log,'WARNING','Failed reading cxr.ini file, using default values instead') % Write to log
    options.runPtx = 1;
    options.runAbnormalities = 1;
    options.runTubes = 0;
    options.ptxKeyImageTypeScores = 1;
    if fileID ~= -1
        fclose(fileID);
    end
end

options.view_position = view_position;

try
    % Run the Algorthimic module
    resultStruct = cxrAlgUnit(files_list, out_folder, fid_log, options);
    cxrOutSruct = cxrFillOutStruct(resultStruct, options, fid_log);
    cxrOutSruct.ALGORITHM_NAME = 'Chest XRAY';
    cxrOutSruct.ALGORITHM_VERSION = ALG_VER;
    
    % Generate and save the output.xml file
    Pref.StructItem = false;
    Pref.CellItem  = false;
    xml_write(strcat(out_folder,filesep, 'output.xml'), cxrOutSruct, 'root', Pref);
    add_line_to_log(fid_log, 'NOTICE', 'cxrWrapper finished');
catch err
    add_line_to_log(fid_log, 'ERROR', err.message, err) % Write to log
    fclose(fid_log);
    if isdeployed
        exit(1);
    else
        return;
    end
end

fclose(fid_log);