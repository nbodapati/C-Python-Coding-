#include<iostream>
using namespace std;

void swap(int* arr,int i,int j){
  arr[i]=arr[i]^arr[j];
  arr[j]=arr[i]^arr[j];
  arr[i]=arr[i]^arr[j];
}

void waveform(int*arr,int n)
  {
    for(int i=0;i<n;i=i+2){
      if(i>0 && arr[i]<arr[i-1]){
        swap(arr,i,i-1);
      }
      if(i<n-1 && arr[i]<arr[i+1]){
        swap(arr,i,i+1);
      }
    }
  }

  int main(){
    int arr[10]={1,2,3,4,5,6,7,8,9,10 };
    waveform(arr,10);

    for(int i=0;i<10;i++){
      cout<<arr[i]<<"\t";
    }
    cout<<endl;
    return 0;
  }
