dir_name='/mnt/disk/wangfangfang/ICDAR2015/script_test_ch4_t1_e1-1501528344/gt/';
list = dir(fullfile(dir_name,'*.txt'));
output = './Annotations/';

for i = 1:length(list)
    disp(list(i).name);
    fileID = fopen(fullfile(dir_name,list(i).name),'r');
%     fgetl(fileID);
    data = textscan(fileID,'%d %d %d %d %d %d %d %d %s %*[^\r]','Delimiter',',');
    fclose(fileID);
    celldisp(data);
    x1 = data{1};
    y1 = data{2};
    x2 = data{3};
    y2 = data{4};
    x3 = data{5};
    y3 = data{6};
    x4 = data{7};
    y4 = data{8};
    trans = data{9};
    idx = str2num(list(i).name(8:end-4)) + 1000;
    anno_name = [sprintf('%06d', idx), '.gt'];
    fid1 = fopen([output, anno_name],'wt+');
    for j = 1:length(data{1})
        if strcmp(trans(j), '###')
            hard_flg = 1;
        else
            hard_flg = 0;
        end
        fprintf(fid1,'%d %d %d %d %d %d %d %d %d \r\n',hard_flg,x1(j),y1(j),x2(j),y2(j),x3(j),y3(j),x4(j),y4(j));
    end   
    fid1 = fclose(fid1);
end