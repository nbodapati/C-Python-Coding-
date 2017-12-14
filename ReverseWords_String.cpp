#include<cstdio>
#include<iostream>
#include<stack>
using namespace std;

class StringOP{
private:
  char* string;
  int length;

public:
  StringOP(char* str){
    string=str;
    length=0;
    char* c=string;
    while(*c!='\0'){
      length++;
      c=c+1; //this increases the pointer by the number of bytes in character.
    }
    cout<<"String length: "<<this->length<<endl;
  }

  void WordReverse(){
    //this function revereses the words in the string in place.(?)
    stack<char*> wordqueue;
    char* start=string; //point to the start. 
    int i=0;int j=0;
    int num_words=0;

    while(*start!='\n')
     {
       if(*start==' '){
          wordqueue.push(this->extract_substring(i,(j-i)));
          i=j+1; //record the next location.  
          num_words++; 
         }
         j++;
         start++;
     }

    while(num_words!=0){
       char* el1=wordqueue.top();
       wordqueue.pop(); 
       cout<<el1; 
       num_words--;
      }
      cout<<endl;
  };

  void printString(){
    cout<<string<<endl;
  }

  char* extract_substring(int start,int length){
    cout<<"start: "<<start<<"length: "<<length<<endl;
    if(start>this->length or start+length>this->length)
    {
      cout<<"String not that long!"<<endl;
      return NULL;
    }
    char* st=string; //points to the start.
    char* end;
    char* substring = new char[length];

    //Navigate to the position of start.
    cout<<"Start: "<<st<<endl;
    st=st+start;
    cout<<"At position: "<<st<<endl;
    end=st+length;
    int i=0;
    while(st<=end){
      substring[i]=*st;
      st++;i++;
    }
    cout<<substring<<endl;
    return substring;
  }
};


int main()
{
  StringOP s = StringOP("I love you,Raghu very much. You are my krishna \n");
  s.printString();
  StringOP substr=StringOP(s.extract_substring(1,15));
  substr.printString();
  s.WordReverse();
  
  return 0;
}
