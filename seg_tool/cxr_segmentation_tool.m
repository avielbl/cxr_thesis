
% --------------------------------------------------------------------
function varargout = cxr_segmentation_tool(varargin)
% CXR_SEGMENTATION_TOOL MATLAB code for cxr_segmentation_tool.fig

% Begin initialization code - DO NOT EDIT
gui_Singleton = 0;
gui_State = struct('gui_Name',       mfilename, ...
                   'gui_Singleton',  gui_Singleton, ...
                   'gui_OpeningFcn', @cxr_segmentation_tool_OpeningFcn, ...
                   'gui_OutputFcn',  @cxr_segmentation_tool_OutputFcn, ...
                   'gui_LayoutFcn',  [] , ...
                   'gui_Callback',   []);
if nargin && ischar(varargin{1})
    gui_State.gui_Callback = str2func(varargin{1});
end

if nargout
    [varargout{1:nargout}] = gui_mainfcn(gui_State, varargin{:});
else
    gui_mainfcn(gui_State, varargin{:});
end
% End initialization code - DO NOT EDIT
end

% --------------------------------------------------------------------
% --- Executes just before cxr_segmentation_tool is made visible.
function cxr_segmentation_tool_OpeningFcn(hObject, ~, handles, varargin)
% This function has no output args, see OutputFcn.
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
% varargin   command line arguments to cxr_segmentation_tool (see VARARGIN)
global curr_im_ind images lung_masks ptx_masks other_masks currView displayProp
%Handeling inputs

if numel(varargin) < 1
    allDataList = getAllDataList();
    preProcessedImagesFile = 'images.mat';
elseif numel(varargin) < 2
    allDataList = varargin{1};
    preProcessedImagesFile = 'images.mat';
else
    preProcessedImagesFile = varargin{2};
end


panPointer = [
            NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
            NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
            NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN,NaN
            NaN,NaN,NaN,NaN,2  ,2  ,NaN,2  ,2  ,NaN,2  ,2  ,NaN,NaN,NaN,NaN
            NaN,NaN,NaN,2  ,1  ,1  ,2  ,1  ,1  ,2  ,1  ,1  ,2  ,2  ,NaN,NaN
            NaN,NaN,2  ,1  ,2  ,2  ,1  ,2  ,2  ,1  ,2  ,2  ,1  ,1  ,2  ,NaN
            NaN,NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2  ,1  ,2
            NaN,NaN,NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2
            NaN,NaN,2  ,1  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2
            NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2
            NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2
            NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2  ,NaN
            NaN,NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2  ,NaN
            NaN,NaN,NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2  ,NaN,NaN
            NaN,NaN,NaN,NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2  ,NaN,NaN
            NaN,NaN,NaN,NaN,2  ,1  ,2  ,2  ,2  ,2  ,2  ,2  ,1  ,2  ,NaN,NaN
            ];


N = numel(allDataList);

%Setting slider step based on images amount
handles.slider1.SliderStep = [1/N, 1/N];

%Generating images and masks
images = struct('image', {}, 'notBlackInd', {});
lung_masks = struct('l_lungs_mask', {}, 'r_lungs_mask', {});
ptx_masks = struct('ptx_mask', {});
other_masks = struct('other_mask', {});
curr_im_ind = 1;
%Saving display properties (w/l and zoom)
emptyCell = cell(N,1);
displayProp = struct('xLim', emptyCell, 'yLim', emptyCell, 'cLim', emptyCell);
imSize = 1024;

loadImages(allDataList, imSize);

%Setting application variabels
setappdata(gcf, 'N', N);
setappdata(gcf, 'imSize', imSize);
setappdata(gcf, 'preProcessedImagesFile', preProcessedImagesFile);
setappdata(gcf, 'allDataList', allDataList);
setappdata(gcf, 'panPointer', panPointer);

currView = display_image(handles);
hZoom = zoom;
hZoom.ActionPostCallback = @postZoomCallback;
hPan = pan;
hPan.ActionPostCallback = @postZoomCallback;
% Update handles structure
guidata(hObject, handles);
end


% --------------------------------------------------------------------
function allDataList = getAllDataList(input_folder)
if nargin < 1
    input_folder = uigetdir;
end
files_list = getFilesList(input_folder);
N = numel(files_list);
allDataList = struct('im_path',{},'im_ID',{});
for i=1:N
    tmpStr = files_list(i).name;
    allDataList(i).im_path = fullfile(input_folder, tmpStr);
    allDataList(i).im_ID = tmpStr;
end
end
        
% --------------------------------------------------------------------
function loadImages(allDataList, imSize)
global images lung_masks ptx_masks other_masks displayProp
N = numel(allDataList);
h = waitbar(0, 'Loading images...');
for i = 1:N
    I = loadNPrepImg(allDataList(i).im_path);
    I = resizeWAspect(I, imSize);
    sz = size(I);
    images(i).image = I;
    lung_masks(i).l_lungs_mask = false(size(images(i).image));
    lung_masks(i).r_lungs_mask = false(size(images(i).image));
    ptx_masks(i).ptx_mask = false(size(images(i).image));
    other_masks(i).other_mask = false(size(images(i).image));
    displayProp(i).cLim = [min(I(:)), max(I(:))];
    displayProp(i).xLim = 0.5 + [0 sz(2)];
    displayProp(i).yLim = 0.5 + [0 sz(1)];
    waitbar(i/N);
end
close(h);
end

% --------------------------------------------------------------------
function postZoomCallback(~, eventdata)

global curr_im_ind displayProp
xLim = eventdata.Axes.XLim;
yLim = eventdata.Axes.YLim;
displayProp(curr_im_ind).xLim = xLim;
displayProp(curr_im_ind).yLim = yLim;
end

% --------------------------------------------------------------------
% --- Outputs from this function are returned to the command line.
function varargout = cxr_segmentation_tool_OutputFcn(hObject, eventdata, handles) 
% varargout  cell array for returning output args (see VARARGOUT);
% hObject    handle to figure
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

% Get default command line output from handles structure
out.images = getappdata(gcf,'images');
out.lung_masks = getappdata(gcf,'lung_masks');
out.ptx_masks = getappdata(gcf,'ptx_masks');
out.other_masks = getappdata(gcf,'other_masks');
varargout{1} = out;
end

% --------------------------------------------------------------------
function mask_out = updateMask(mask_in, handles)
global currView
if handles.zoom.Value == 1
    handles.zoom.Value = 0;
    handles.zoom.BackgroundColor = [0.0, 0.0, 0.0];
end
if handles.pan.Value == 1
    handles.pan.Value = 0;
    handles.pan.BackgroundColor = [0.0, 0.0, 0.0];
end

set(gcf, 'WindowButtonDownFcn',[], 'WindowButtonUpFcn', []);
mask_out = false(size(mask_in));
if handles.free_hand_radio.Value == 1
    h = imfreehand;
else
    h = impoly;
end
set(gcf, 'WindowButtonDownFcn', @figure1_WindowButtonDownFcn,... 
         'WindowButtonUpFcn', @figure1_WindowButtonUpFcn);
if ~isempty(h)
    mask = createMask(h,currView);
    if handles.add_radio.Value == 1
        mask_out = logical(mask + mask_in);
    else
        remove_mask = mask .* mask_in;
        mask_out = logical(mask_in - remove_mask);
    end
end
end
    
% --------------------------------------------------------------------
% --- Executes on button press in l_lung_button.
function l_lung_button_Callback(~, ~, handles)
% hObject    handle to l_lung_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global curr_im_ind lung_masks currView

mask = updateMask(lung_masks(curr_im_ind).l_lungs_mask, handles);
lung_masks(curr_im_ind).l_lungs_mask = mask;
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes on button press in r_lung_button.
function r_lung_button_Callback(~, ~, handles)
% hObject    handle to r_lung_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global curr_im_ind lung_masks currView

mask = updateMask(lung_masks(curr_im_ind).r_lungs_mask, handles);
lung_masks(curr_im_ind).r_lungs_mask = mask;
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes on slider movement.
function slider1_Callback(hObject, ~, handles)
% hObject    handle to slider1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global curr_im_ind currView
slider_value = get(hObject,'Value');
step_size = get(hObject,'SliderStep'); step_size = step_size(1);

if slider_value > (1 - step_size)
    curr_im_ind = getappdata(gcf, 'N');
    set(hObject,'Value',1 - step_size);
elseif slider_value < 0
    curr_im_ind = 1;
    set(hObject,'Value',0);
else
    curr_im_ind = uint16(slider_value * getappdata(gcf, 'N') + 1);
end
currView = display_image(handles);
end

% --------------------------------------------------------------------
function currView = display_image(handles)
global images curr_im_ind lung_masks ptx_masks other_masks displayProp

I = images(curr_im_ind).image;

imshow(I, 'Parent', handles.image_axis); 
cLim = displayProp(curr_im_ind).cLim;
handles.image_axis.CLim = cLim;

setappdata(gcf,'window',cLim(2) - cLim(1));
setappdata(gcf,'level',(cLim(1) + cLim(2)) * 0.5);

if handles.l_lung_visible.Value == 1
    red_mask = lung_masks(curr_im_ind).l_lungs_mask;
else
    red_mask = false(size(I));
end

if handles.r_lung_visible.Value == 1
    red_mask = red_mask + lung_masks(curr_im_ind).r_lungs_mask;
end

if handles.ptx_visible.Value == 1
    green_mask = ptx_masks(curr_im_ind).ptx_mask ;
else
    green_mask = false(size(I));
end

if handles.other_visible.Value == 1
    blue_mask = imresize(other_masks(curr_im_ind).other_mask,size(I));
else
    blue_mask = false(size(I));
end

mask_rgb = false([size(I), 3]);
mask_rgb(:,:,1) = red_mask;
mask_rgb(:,:,2) = green_mask;
mask_rgb(:,:,3) = blue_mask;
mask_rgb = 255 * uint8(mask_rgb);
hold on;
% r_contour = bwboundaries(red_mask); r_contour= r_contour{1};
% plot(r_contour(:,2), r_contour(:,1), 'r', 'Linewidth', 1)

% g_contour = bwboundaries(green_mask);
% if ~isempty(g_contour)
%     for i = 1:numel(g_contour)
%         plot(g_contour{i}(:,2), g_contour{i}(:,1), 'g', 'Linewidth', 1);
%     end
% end

% b_contour = bwboundaries(blue_mask); b_contour= b_contour{1};
% plot(b_contour(:,2), b_contour(:,1), 'b', 'Linewidth', 1);

currView = imshow(mask_rgb, 'Parent', handles.image_axis); 
handles.image_axis.XLim = displayProp(curr_im_ind).xLim;
handles.image_axis.YLim = displayProp(curr_im_ind).yLim;
hold off;
alpha_map = rgb2gray(single(mask_rgb));
alpha_map(alpha_map > 0) = 0.1;
set(currView, 'AlphaData', alpha_map);
allDataList = getappdata(gcf, 'allDataList');
title = [allDataList(curr_im_ind).im_ID,' ',num2str(curr_im_ind) ,'/',num2str(getappdata(gcf, 'N'))];
handles.image_title.String = title;
c = uicontextmenu;
uimenu(c,'Label','Reset View','Callback',@reset_view_contmenue_Callback);
currView.UIContextMenu = c;

end

% --------------------------------------------------------------------
% --- Executes on button press in l_lung_visible.
function l_lung_visible_Callback(~, ~, handles)
% hObject    handle to l_lung_visible (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currView
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes on button press in r_lung_visible.
function r_lung_visible_Callback(~, ~, handles)
% hObject    handle to r_lung_visible (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currView
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes when user attempts to close figure1.
function figure1_CloseRequestFcn(hObject, ~, ~)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if isequal(get(hObject, 'waitstatus'), 'waiting')
    % The GUI is waiting
    uiresume(hObject);
else
    % The GUI is no longer waiting
    delete(hObject);
end
end

% --------------------------------------------------------------------
function load_lungs_segmentation_menu_Callback(~, ~, handles)
% hObject    handle to load_lungs_segmentation_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global lung_masks
[FileName,PathName] = uigetfile('*.mat','Select lungs segmentation file');
if exist(fullfile(PathName, FileName), 'file')
    h = msgbox('Loading...');
    loaded_lung_masks = load(fullfile(PathName, FileName)); loaded_lung_masks = loaded_lung_masks.lung_masks;
    close(h);
    N = getappdata(gcf, 'N');
    if ~isstruct(loaded_lung_masks)
        msgbox('Wrong file, struct is expected');
    elseif numel(loaded_lung_masks) ~= N
        str=sprintf('Wrong file, file has %d masks while %d are expected\n', numel(loaded_lung_masks), N);
        msgbox(str, 'Error', 'error');
    else
        imSize = getappdata(gcf,'imSize');
        for i = 1:N
            if ~isempty(loaded_lung_masks(i).l_lungs_mask)
                loaded_lung_masks(i).l_lungs_mask =  resizeWAspect(loaded_lung_masks(i).l_lungs_mask, imSize);
            else
                loaded_lung_masks(i).l_lungs_mask = [];
            end
            if ~isempty(loaded_lung_masks(i).r_lungs_mask)
                loaded_lung_masks(i).r_lungs_mask =  resizeWAspect(loaded_lung_masks(i).r_lungs_mask, imSize);
            else
                loaded_lung_masks(i).r_lungs_mask = [];
            end
        end
        lung_masks = loaded_lung_masks;
        display_image(handles);
        msgbox('Lungs segmentation loaded');
    end
else
    msgbox('No valid file was selected', 'Error', 'error');
end
end

% --------------------------------------------------------------------
function save_lungs_segmentation_menu_Callback(varargin)
% hObject    handle to save_lungs_segmentation_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global lung_masks
file_name = getappdata(gcf, 'lungMasksFile');
h = msgbox('Saving...');
save(file_name, 'lung_masks');
close(h);
msgbox('Lung Masks Saved');
end

% --------------------------------------------------------------------
function save_as_lungs_segmentation_menu_Callback(varargin)
% hObject    handle to save_as_lungs_segmentation_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global lung_masks

[FileName,PathName] = uiputfile('*.mat', 'Save lungs segmentations as', 'lung_masks_gt.mat');
h = msgbox('Saving...');
save(fullfile(PathName, FileName) ,'lung_masks');
close(h);
msgbox('Lung Masks Saved');
end

% --------------------------------------------------------------------
function save_images_menu_Callback(varargin)
% hObject    handle to save_images_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global images
file_name = getappdata(gcf, 'preProcessedImagesFile');
h = msgbox('Saving...');
save(file_name, 'images', '-v7.3');
close(h);
msgbox('Images Saved');
end

% --------------------------------------------------------------------
function save_lungs_toolbar_button_ClickedCallback(hObject, eventdata, handles)
% hObject    handle to save_lungs_toolbar_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
save_lungs_segmentation_menu_Callback(hObject, eventdata, handles);
end

% --------------------------------------------------------------------
function w_l_tool_OffCallback(varargin)
% hObject    handle to w_l_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currView
set(currView,'ButtonDownFcn','');
end

% --------------------------------------------------------------------
function w_l_tool_OnCallback(varargin)
% hObject    handle to w_l_tool (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currView
set(currView,'ButtonDownFcn',@startWindowing);
end

% --------------------------------------------------------------------
function reset_view_contmenue_Callback(varargin)
% hObject    handle to reset_view_contmenue (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global images curr_im_ind displayProp

I = images(curr_im_ind).image;
window=[min(I(:)), max(I(:))];
set(gca,'CLim',window);
XLim = 0.5 + [0, size(I,1)];
YLim = 0.5 + [0, size(I,2)];
set(gca, 'XLim', XLim);
set(gca, 'YLim', YLim);
zoom(gcf,'reset');
setappdata(gcf,'window',window(2)-window(1));
setappdata(gcf,'level',(window(1)+window(2))*0.5);

displayProp(curr_im_ind).cLim = window;
displayProp(curr_im_ind).xLim = XLim;
displayProp(curr_im_ind).yLim = YLim;
end

% --------------------------------------------------------------------
% --- Executes on mouse press over figure background, over a disabled or
% --- inactive control, or over an axes background.
function figure1_WindowButtonUpFcn(varargin)
% hObject    handle to figure1 (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
set(gcf,'WindowButtonMotionFcn','', 'pointer', 'arrow');
end

% --------------------------------------------------------------------
function save_as_images_segmentation_menu_Callback(varargin)
% hObject    handle to save_as_images_segmentation_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global images
[FileName,PathName] = uiputfile('*.mat', 'Save images as', 'images.mat');
h = msgbox('Saving...');
save(fullfile(PathName, FileName) ,'images', '-v7.3');
close(h);
fprintf('Images Saved\n');
end

% --------------------------------------------------------------------
% --- Executes on button press in ptx_button.
function ptx_button_Callback(~, ~, handles)
% hObject    handle to ptx_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global ptx_masks curr_im_ind currView

mask = updateMask(ptx_masks(curr_im_ind).ptx_mask, handles);
ptx_masks(curr_im_ind).ptx_mask = mask;
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes on button press in ptx_visible.
function ptx_visible_Callback(~, ~, handles)
% hObject    handle to ptx_visible (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currView
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes on button press in other_button.
function other_button_Callback(~, ~, handles)
% hObject    handle to other_button (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global other_masks curr_im_ind currView

mask = updateMask(other_masks(curr_im_ind).other_mask, handles);
other_masks(curr_im_ind).other_mask = mask;
currView = display_image(handles);
end

% --------------------------------------------------------------------
% --- Executes on button press in other_visible.
function other_visible_Callback(~, ~, handles)
% hObject    handle to other_visible (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global currView
currView = display_image(handles);
end

% --------------------------------------------------------------------
function load_ptx_menu_Callback(hObject, ~, handles)
% hObject    handle to load_ptx_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global ptx_masks
[FileName,PathName] = uigetfile('*.mat','Select ptx segmentation file');
if exist(fullfile(PathName, FileName), 'file')
    h = msgbox('Loading...');
    loaded_ptx_masks = load(fullfile(PathName, FileName)); loaded_ptx_masks = loaded_ptx_masks.ptx_masks;
    close(h);
    N = getappdata(gcf, 'N');
    if ~isstruct(loaded_ptx_masks)
        msgbox('Wrong file, struct is expected\n');
    elseif numel(loaded_ptx_masks) ~= N
        str = sprintf('Wrong file, file has %d masks while %d are expected\n', numel(loaded_ptx_masks), N);
        msgbox(str, 'Error', 'error');
    else
        for i = 1:N
            loaded_ptx_masks(i).ptx_mask = resizeWAspect(loaded_ptx_masks(i).ptx_mask, getappdata(gcf,'imSize'));
        end
        ptx_masks = loaded_ptx_masks;
        guidata(hObject,handles);
        display_image(handles);
        msgbox('PTX segmentation loaded');
    end
else
    msgbox('No valid file was selected','Error','error');
end
end

% --------------------------------------------------------------------
function save_as_ptx_menu_Callback(varargin)
% hObject    handle to save_as_ptx_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global ptx_masks

[FileName,PathName] = uiputfile('*.mat', 'Save ptx segmentations as', 'ptx_masks_gt.mat');
h = msgbox('Saving...');
save(fullfile(PathName, FileName) ,'ptx_masks');
close(h);
msgbox('PTX Masks Saved');
end

% --------------------------------------------------------------------
% --- Executes on button press in zoom.
function zoom_Callback(hObject, ~, handles)
% hObject    handle to zoom (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
if hObject.Value == 1
    if handles.pan.Value == 1
        handles.pan.Value = 0;
        handles.pan.BackgroundColor = [0.0, 0.0, 0.0];
    end
    zoom on;
    hObject.BackgroundColor = [0.6, 0.6, 0.6];
else
    zoom off;
    hObject.BackgroundColor = [0.0, 0.0, 0.0];
end
end

% --------------------------------------------------------------------
% --- Executes on button press in pan.
function pan_Callback(hObject, ~, handles)
% hObject    handle to pan (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

if hObject.Value == 1
    if handles.zoom.Value == 1
        handles.zoom.Value = 0;
        handles.zoom.BackgroundColor = [0.0, 0.0, 0.0];
    end
    pan on;
    hObject.BackgroundColor = [0.6, 0.6, 0.6];
else
    pan off;
    hObject.BackgroundColor = [0.0, 0.0, 0.0];
end
end

% --------------------------------------------------------------------
% --- Executes on mouse key press with focus on figure1 or any of its controls.
function figure1_WindowButtonDownFcn(hObject, eventdata, varargin)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
global curr_im_ind displayProp

seltype = hObject.SelectionType;
currView = eventdata.Source.CurrentAxes;
firstClick = currView.CurrentPoint;
switch seltype
    case 'normal'
        hObject.WindowButtonMotionFcn = @windowingTool;
        
    case 'extend'
        set(gcf, 'pointer', 'custom', 'pointershapecdata', getappdata(gcf, 'panPointer'));
        hObject.WindowButtonMotionFcn = @paningTool;
end

% --------------------------------------------------------------------
% -- Nested function
    function windowingTool(varargin)
        currWindow=getappdata(gcf,'window');
        currLevel=getappdata(gcf,'level');
        currPoint = currView.CurrentPoint;
        newWindow=currWindow+(currPoint(1)-firstClick(1));
        if newWindow<0
            newWindow=2;
        end
        newLevel=currLevel-(firstClick(3)-currPoint(3));
        firstClick = currPoint;
        setappdata(gcf,'window',newWindow);
        setappdata(gcf,'level',newLevel);
        clim_min=newLevel-0.5*newWindow;
        clim_max=newLevel+0.5*newWindow;
        currView.CLim = [clim_min clim_max];
        displayProp(curr_im_ind).cLim = [clim_min clim_max];
    end

% --------------------------------------------------------------------
% -- Nested function
    function paningTool(varargin)
        currPoint = currView.CurrentPoint;
        newXLim = currView.XLim + firstClick(1,1) - (currPoint(1,1) + currPoint(2,1)) * 0.5;
        newYLim = currView.YLim + firstClick(1,2) - (currPoint(1,2) + currPoint(2,2)) * 0.5;
        currView.XLim = newXLim;
        currView.YLim = newYLim;
        displayProp(curr_im_ind).xLim = newXLim;
        displayProp(curr_im_ind).yLim = newYLim;
    end

end %end of figure1_WindowButtonDownFcn

% --------------------------------------------------------------------
% --- Executes on key press with focus on figure1 or any of its controls.
function figure1_WindowKeyPressFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
%	Key: name of the key that was pressed, in lower case
%	Character: character interpretation of the key(s) that was pressed
%	Modifier: name(s) of the modifier key(s) (i.e., control, shift) pressed
% handles    structure with handles and user data (see GUIDATA)
key = eventdata.Key;
slider_value = get(handles.slider1,'Value');
step_size = get(handles.slider1,'SliderStep'); step_size = step_size(1);
handles = guidata(hObject);
switch key
    case 'rightarrow'
        set(handles.slider1, 'Value', slider_value + step_size);
        slider1_Callback(handles.slider1, eventdata, handles);
    case 'leftarrow'
        set(handles.slider1, 'Value', slider_value - step_size);
        slider1_Callback(handles.slider1, eventdata, handles);
    case 'l'
        l_lung_button_Callback(handles.l_lung_button, eventdata, handles);
    case 'r'
        r_lung_button_Callback(handles.r_lung_button, eventdata, handles);
    case 'add'
        zoom(1.5);
    case 'subtract'
        zoom out;
end
guidata(hObject,handles);
end


% --- Executes on scroll wheel click while the figure is in focus.
function figure1_WindowScrollWheelFcn(hObject, eventdata, handles)
% hObject    handle to figure1 (see GCBO)
% eventdata  structure with the following fields (see MATLAB.UI.FIGURE)
%	VerticalScrollCount: signed integer indicating direction and number of clicks
%	VerticalScrollAmount: number of lines scrolled for each click
% handles    structure with handles and user data (see GUIDATA)

% Power law allows for the inverse to work:
%      C^(x) * C^(-x) = 1
% Choose C to get "appropriate" zoom factor
global curr_im_ind displayProp

C = 1.02;
currXLim = handles.image_axis.XLim; 
midX = mean(currXLim); 
rngXhalf = diff(currXLim) / 2;
currYLim = handles.image_axis.YLim; 
midY = mean(currYLim); 
rngYhalf = diff(currYLim) / 2;
currPt = mean(handles.image_axis.CurrentPoint); 
currPt = currPt(1:2);
currPt2 = (currPt - [midX, midY]) ./ [rngXhalf, rngYhalf];
currPt  = [currPt; currPt];
currPt2 = [-(1+currPt2).*[rngXhalf, rngYhalf];...
            (1-currPt2).*[rngXhalf, rngYhalf]];
        
r = C^(eventdata.VerticalScrollCount*eventdata.VerticalScrollAmount);
newLimSpan = r * currPt2;
% Determine new limits based on r
lims = currPt + newLimSpan;
handles.image_axis.XLim = lims(:,1);
handles.image_axis.YLim = lims(:,2);
displayProp(curr_im_ind).xLim = lims(:,1);
displayProp(curr_im_ind).yLim = lims(:,2);
end


% --------------------------------------------------------------------
function load_folder_menu_Callback(hObject, ~, handles)
% hObject    handle to load_folder_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global images curr_im_ind displayProp lung_masks ptx_masks other_masks currView

allDataList = getAllDataList();
N = numel(allDataList);

%Setting slider step based on images amount
handles.slider1.SliderStep = [1/N, 1/N];
handles.slider1.Value = 0;
setappdata(gcf, 'N', N);
setappdata(gcf, 'allDataList', allDataList);
curr_im_ind = 1;
emptyCell = cell(N,1);
images = struct('image', emptyCell);
displayProp = struct('xLim', emptyCell, 'yLim', emptyCell, 'cLim', emptyCell);
lung_masks = struct('l_lungs_mask', {}, 'r_lungs_mask', {});
ptx_masks = struct('ptx_mask', {});
other_masks = struct('other_mask', {});

loadImages(allDataList,getappdata(gcf, 'imSize'));
currView = display_image(handles);
guidata(hObject, handles);
end



% --------------------------------------------------------------------
function load_other_menu_Callback(hObject, ~, handles)
% hObject    handle to load_other_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)
global other_masks

[FileName,PathName] = uigetfile('*.mat','Select segmentation file');
if exist(fullfile(PathName, FileName), 'file')
    h = msgbox('Loading...');
    loaded_masks = load(fullfile(PathName, FileName));
    names = fieldnames(loaded_masks);
    loaded_masks = loaded_masks.(names{1});
    close(h);
    N = getappdata(gcf,'N');
    if ~isstruct(loaded_masks)
        msgbox('Wrong file, struct is expected\n');
    elseif numel(loaded_masks) ~= N
        str = sprintf('Wrong file, file has %d masks while %d are expected\n', numel(loaded_masks), N);
        msgbox(str, 'Error', 'error');
    else
        names = fieldnames(loaded_masks);
        for i = 1:N
            other_masks(i).other_mask = resizeWAspect(loaded_masks(i).(names{1}), getappdata(gcf,'imSize'));
        end
        guidata(hObject,handles);
        display_image(handles);
        msgbox('Segmentation loaded');
    end
else
    msgbox('No valid file was selected','Error','error');
end
end

% --------------------------------------------------------------------
function save_as_other_menu_Callback(varargin)
% hObject    handle to save_as_other_menu (see GCBO)
% eventdata  reserved - to be defined in a future version of MATLAB
% handles    structure with handles and user data (see GUIDATA)

global other_masks

[FileName,PathName] = uiputfile('*.mat', 'Save other segmentations as', 'other_masks.mat');
h = msgbox('Saving...');
save(fullfile(PathName, FileName) ,'other_masks');
close(h);
msgbox('Masks Saved');
end
