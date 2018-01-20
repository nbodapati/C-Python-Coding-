function[images,im]= EigenFaces(images)
    %Pca from matlab takes in input of shape NxP and
         %returns PxK where K<=P(strictly) where each column is a data vector and
         %each element of column weight for corresponding eigen vector.
         first_image=images(2,:);
         first_image=permute(reshape(first_image,[64,64,3]),[2,1,3]);
         figure,imshow(first_image/255);
         %images here is a NxP shape, transpose.
         fprintf('Original number of features:\n'); disp(size(images));
         im=pca(images,'NumComponents',20);%,'NumComponents',floor(num_comp*N));
         %im=im(:,10:end);
         disp(size(im)); 
         
         images=images*im;
         disp(size(images)); 
         first_image=images(2,:);
         
         first_image=first_image.*im;
         %first_image=sum(first_image,2);
         disp(size(first_image));
         
         for i=1:size(first_image,2),
             disp(i);disp(size(first_image(:,i)));
         first_image_=permute(reshape(first_image(:,i),[64,64,3]),[2,1,3]);
         figure,imshow(first_image_);
         end
         
         first_image=sum(first_image,2);
         first_image=permute(reshape(first_image,[64,64,3]),[2,1,3]);
         figure,imshow(first_image/255);
         fprintf('Final number of features:\n ');disp(size(images));

end