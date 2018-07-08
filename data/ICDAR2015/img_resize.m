list = dir('./JPGImages/*.jpg');
anno_list = dir('./Annotations/*.gt');
im_path = './JPGImages/';
anno_path = './Annotations/';

len = length(list);
parfor i = 1 : len
    im = imread([im_path, list(i).name]);
    [h, w, ~] = size(im);
    s1 = min(w, h);
    s2 = max(w, h);
    if s2 > 1000 || s1 > 600 
        disp(i);
        scale = min(1000*1.0/s2, 600*1.0/s1);
        im = imresize(im, scale);
        imwrite(im, [im_path, list(i).name]);

        fid = fopen([anno_path,anno_list(i).name],'r+');
        data = textscan(fid,'%d %d %d %d %d %d %d %d %d');
        fid = fclose(fid);
        hard_flg = data{1};
        x1 = data{2} * scale;
        y1 = data{3} * scale;
        x2 = data{4} * scale;
        y2 = data{5} * scale;
        x3 = data{6} * scale;
        y3 = data{7} * scale;
        x4 = data{8} * scale;
        y4 = data{9} * scale;
        fid1 = fopen([anno_path,anno_list(i).name],'wt+');
        for j = 1:length(data{1})
            fprintf(fid1,'%d %d %d %d %d %d %d %d %d \r\n',hard_flg(j),x1(j),y1(j),x2(j),y2(j),x3(j),y3(j),x4(j),y4(j));
        end
        fid1 = fclose(fid1);
    
    end
end



