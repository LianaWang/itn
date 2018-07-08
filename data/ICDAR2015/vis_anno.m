list = dir('./JPGImages/*.jpg');
anno_list = dir('./Annotations/*.gt');
im_path = './JPGImages/';
anno_path = './Annotations/';

len = length(list);
for i = 90 %: len
    im = imread([im_path, list(i).name]);
    fid = fopen([anno_path,anno_list(i).name],'r+');
    data = textscan(fid,'%d %d %d %d %d %d %d %d %d');
    fid = fclose(fid);
    hard_flg = data{1};
    x1 = data{2};
    y1 = data{3};
    x2 = data{4};
    y2 = data{5};
    x3 = data{6};
    y3 = data{7};
    x4 = data{8};
    y4 = data{9};
    figure;
    imshow(im);
    hold on;
    for j = 1:length(data{1})
        plot([x1(j),x2(j),x3(j),x4(j)],[y1(j),y2(j),y3(j),y4(j)] );
    end

end