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
            mean_mouth_x,mean_mouth_y]
        
     
     for picture=1:size(T,1),
         image_=T(picture,1).Var1{1};
         img=imread(fullfile(imageDir,image_));
         fprintf('Image file: %s\n',image_);
         
         if(numel(size(img))~=3),
             continue
         end    
         display(size(img));
          
         %figure,imshow(img); 
         locations=T(picture,2:end);
         leye_x=locations.Var2; %get this in float format.
         leye_y=locations.Var3;
         reye_x=locations.Var4;
         reye_y=locations.Var5;
         nose_x=locations.Var6;
         nose_y=locations.Var7;
         mouth_x=locations.Var8; 
         mouth_y=locations.Var9;
         
         actual=[leye_x,leye_y;
                        reye_x,reye_y;
                        nose_x,nose_y;
                        mouth_x,mouth_y];
                 
         %Get the affine transform 
         tform=fitgeotrans(actual,means,'nonreflectivesimilarity'); %they should have ncols=2
         img=imwarp(img,tform,'OutputView',imref2d(size(img)));

         %figure,imshow(img);%imtool(img);
         %save all the images in a cell array called imarray.
         img=imcrop(img,[mean_nose_x-30,mean_nose_y-40,60,70]);
         if(isempty(img)),
             fprintf('Empty..');
             continue;
         end   
         img=imresize(img,[224,224]); %this has no effect on dim-3.
         
         %{
         ci = [112, 112,80,120)];     
         [xx,yy] = ndgrid((1:size(img,1))-ci(1),(1:size(img,2))-ci(2));
         mask = uint8((ci(3)*xx.^2 + ci(4)*yy.^2)<(ci(3)*ci(4).^2));
         croppedImage = uint8(zeros(size(img)));
         croppedImage(:,:,1) = img(:,:,1).*mask;
         croppedImage(:,:,2) = img(:,:,2).*mask;
         croppedImage(:,:,3) = img(:,:,3).*mask;
         imshow(croppedImage);
         %}
         imArray{picture}=img;
     end
     save('Aligned_Images','imArray','T','-v7.3');
end