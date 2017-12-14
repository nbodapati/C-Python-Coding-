#include<iostream>
#include<cstdio>
#include<queue>
using namespace std;

class ArrayOP{
private:
  int* arr;
  int length;
  queue<int> desc_queue;

public:
  ArrayOP(int arr[],int n){
    this->arr=arr;
    this->length=n;
  }

//with-in class recursion works!
void get_descendants(int index)
{ //at leaf nodes.
  if(index==this->length or 2*index+1 >=this->length or 2*index+2 >=this->length)
  {
    return;
  }
    desc_queue.push(this->arr[2*index+1]);
    desc_queue.push(this->arr[2*index+2]);
    this->get_descendants(2*index+1);
    this->get_descendants(2*index+2);
}
void empty_queue(){
  while(!desc_queue.empty()){
       //just keep popping the front element of the queue.
       desc_queue.pop();
  }
}

bool is_array_maxheap(){
      //loop through each node in the tree and check
      //if its value is greater than its descendants.
      for(int i=0;i<this->length;i++){
          cout<<"i="<<arr[i]<<endl;
          get_descendants(i);
          while(!desc_queue.empty()){
            int desc=desc_queue.front();
            desc_queue.pop();
            cout<<"desc.."<<desc<<endl;

            if(desc>arr[i]){
              cout<<"Not max heap!"<<endl;
              //empty the queue since will be used by other function as well.
              this->empty_queue();
              return false;
            }
        }
      }
      //in case all nodes match the criteria.
      cout<<"Yes,max-heap!!"<<endl;
   }

  bool is_array_minheap(){
         //loop through each node in the tree and check
         //if its value is smaller than its descendants.
         for(int i=0;i<this->length;i++){
             cout<<"i="<<arr[i]<<endl;
             get_descendants(i);
             while(!desc_queue.empty()){
               int desc=desc_queue.front();
               desc_queue.pop();
               cout<<"desc.."<<desc<<endl;

               if(desc<arr[i]){
                 cout<<"Not min heap!"<<endl;
                 //empty the queue since will be used by other function as well.
                 this->empty_queue();
                 return false;
               }
           }
         }
         cout<<"Yes,a min-heap!!"<<endl;
      }
};

int main(){
int *arr1 =new int[12];
int *arr2 =new int[12];

for(int i=12;i>0;i--){
  arr1[12-i]=i; //will go as 0,1,2,3,4..
}
for(int i=0;i<12;i++){
  arr2[i]=i; //will go as 0,1,2,3,4..
}

ArrayOP aop=ArrayOP(arr1,12);
aop.is_array_maxheap();
aop.is_array_minheap();

ArrayOP aop2=ArrayOP(arr2,12);
aop2.is_array_minheap();
aop2.is_array_maxheap();

int *arr3=new int[4];
arr3[0]=100;
arr3[1]=101;
arr3[2]=99;
arr3[3]=10;
ArrayOP aop3=ArrayOP(arr3,4);
aop3.is_array_maxheap();
aop3.is_array_minheap();

return 0;
}
