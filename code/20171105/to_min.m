path2015='E:\Tianchi\NEW_DATA2\224_rgb\unet_2015_224_red_new\';
path2017='E:\Tianchi\NEW_DATA2\224_rgb\unet_2017_224_red_new\';
store_path='E:\Tianchi\NEW_DATA2\224_rgb\min_unet_224_red_new\';
pic_all_2015 = dir(path2015);
pic_all_2017 = dir(path2017);
number=size(pic_all_2015,1)-2;
for i=1:number
    img15_s_path = [path2015,pic_all_2015(i+2).name];
    img15_s = imread(img15_s_path);
    img17_s_path = [path2017,pic_all_2017(i+2).name];
    img17_s = imread(img17_s_path);
    img15_s(img15_s>100)=255;
    img15_s(img15_s<=100)=0;
    img17_s(img17_s>100)=255;
    img17_s(img17_s<=100)=0;
    img_min=zeros(size(img17_s));
    [height,width]=size(img17_s);
    for k=1:height
        for j=1:width
            a=img17_s(k,j)-img15_s(k,j);
            if a>100
            img_min(k,j)=255;
            end
        end
    end
    store_path_s = [store_path,pic_all_2017(i+2).name];
    imwrite(img_min,store_path_s);
end

          
