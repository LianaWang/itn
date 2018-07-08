fid = fopen('./ImageSets/Main/train.txt','wt+');
fid1 = fopen('./ImageSets/Main/test.txt','wt+');
for i = 1:1000
    fprintf(fid,num2str(i, '%06d'));
    fprintf(fid,'\n');
end
fid = fclose(fid);
for j = 1001:1500
    fprintf(fid1,num2str(j, '%06d'));
    fprintf(fid1,'\n');
end
fid1 = fclose(fid1);