function CropFaces
     %Read each image file from the folder Caltech_Webfaces
     %Read the lefteye_x,lefteye_y,righteye_x,righteye_y,nose_x,nose_y,
     %mouth_x,mouth_y coordinates from Webfaces_GroundTruth.txt
     imageDir='./Caltech_WebFaces\';
     
     T=readtable('WebFaces_GroundThruth.txt','ReadVariableNames',false);
     %equivalent to header=None from pandas.
     T(:,'Var10')=[];
     %remove the last column which is all nan.
     
     %find affine transform to mean_leye_x and so on.
     mean_leye_x=mean(T.Var2);
     mean_leye_y=mean(T.Var3);
     mean_reye_x=mean(T.Var4);
     mean_reye_y=mean(T.Var5);
     mean_nose_x=mean(T.Var6);
     mean_nose_y=mean(T.Var7);
     mean_mouth_x=mean(T.Var8);
     mean_mouth_y=mean(T.Var9);
     
     means=[mean_leye_x,mean_leye_y;
            mean_reye_x,mean_reye_y;
            mean_nose_x,mean_nose_y;
            mean_mouth_x,mean_mouth_y];
     
     for picture=1:5%size(T,1),
         
         image_=T(picture,1).Var1{1};
         img=imread(fullfile(imageDir,image_));
         fprintf('Image file: %s',image_);
         display(size(img));
         figure,imshow(img);
         locations=T(picture,2:end)
         leye_x=locations.Var2; %get this in float format.
         leye_y=locations.Var3;
         reye_x=locations.Var4;
         reye_y=locations.Var5;
         nose_x=locations.Var6;
         nose_y=locations.Var7;
         mouth_x=locations.Var8; 
         mouth_y=locations.Var9;
         
         img=imresize(img,[224,224]); %this has no effect on dim-3.
         actual=[leye_x,leye_y;
                        reye_x,reye_y;
                        nose_x,nose_y;
                        mouth_x,mouth_y];
         %Get the affine transform 
         tform=fitgeotrans(means,actual,'NonreflectiveSimilarity'); %they should have ncols=2
         img=imwarp(img,tform);
         %save all the images in a cell array called imarray.
         %img2=imcrop(img,[87,38,(140-87),(105-38)]);
         figure,imshow(img);
         imArray{picture}=img;
     end
end