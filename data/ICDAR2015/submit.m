clear;
im_path = './test/';
target_zip_path = '/mnt/disk/xiaoxiannv/ICDAR2015/';
exp_name = 'sup_ic15_vgg';
DATA_DIR = ['/home/liming/project/text/rcnn/experiments/', exp_name,'/output/model_iter_200000/'];
sub_ori = [DATA_DIR, 'submit_roi/'];
sub_out = [DATA_DIR, 'submit/'];
list=textread('ImageSets/Main/test.txt','%s');

mkdir(sub_out);
rmdir(sub_out,'s');
mkdir(sub_out);

len = length(list);
not_circle_num =  0;
for i = 1 : len
    fid = fopen(fullfile(sub_ori,list{i}),'r');
    if fid<0
%         fprintf('%s\n',list{i});
        continue
    end
    im_name = fullfile(im_path,[list{i}, '.jpg']);
    iminfo = imfinfo(im_name);
    h=iminfo.Height;
    w=iminfo.Width;
    s1 = min(w, h);
    s2 = max(w, h);
    scale = min(1000.0/s2, 600.0/s1);
    data = textscan(fid,'%f %f %f %f %f %f %f %f %f','Delimiter',' ');
    fid = fclose(fid);
    x1 = int32(data{1} / scale);
    y1 = int32(data{2} / scale);
    x2 = int32(data{3} / scale);
    y2 = int32(data{4} / scale);
    x3 = int32(data{5} / scale);
    y3 = int32(data{6} / scale);
    x4 = int32(data{7} / scale);
    y4 = int32(data{8} / scale);
    score = data{9};
    res_name = sprintf('res_img_%d.txt',str2double(list{i})-1000);
    fid1 = fopen(fullfile(sub_out,res_name),'w');
    for j = 1:length(data{1})
        xx1=x1(j);xx2=x2(j);xx3=x3(j);xx4=x4(j);
        yy1=y1(j);yy2=y2(j);yy3=y3(j);yy4=y4(j);
        summatory = (xx2-xx1)*(yy2+yy1) + (xx3-xx2)*(yy3+yy2) + (xx4-xx3)*(yy4+yy3) + (xx1-xx4)*(yy1+yy4);
        if summatory > 0
            not_circle_num =  not_circle_num + 1;
        else
            fprintf(fid1,'%d,%d,%d,%d,%d,%d,%d,%d\r\n',xx1,yy1,xx2,yy2,xx3,yy3,xx4,yy4);
        end
    end
    fid1 = fclose(fid1);
end
fprintf('Non-valid lines number:%d\n',not_circle_num);
if ~isempty(strfind(exp_name,'/'))
    exp_name=exp_name(strfind(exp_name,'/')+1:end);
end
zip_name = ['./',exp_name,'.zip'];
status = system(['zip -q ',zip_name,' -j ', sub_out, '*.txt']);
system(['mv ',zip_name,' ', target_zip_path]);
system(['python /home/liming/temp/ic15/script.py -g=/home/liming/temp/ic15/gt.zip -s=', [target_zip_path,zip_name]]);
fprintf('\n');
