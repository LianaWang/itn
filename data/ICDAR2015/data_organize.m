list = dir('/mnt/disk/xiaoxiannv/ICDAR2015/script_test_ch4_t1_e1-1501528344/ch4_test_images/*.jpg');
im_path = './test/';
input_path ='/mnt/disk/xiaoxiannv/ICDAR2015/script_test_ch4_t1_e1-1501528344/ch4_test_images/';

len = length(list);
for i = 1: len
    im = imread([input_path, list(i).name]);
    disp(list(i).name(5:end-4));
    idx = str2num(list(i).name(5:end-4));
    idx = idx+1000;
    disp(idx);
    im_name = [sprintf('%06d', idx), '.jpg'];
    imwrite(im, [im_path, im_name]);
    %delete([input_path, list(i).name]);
end