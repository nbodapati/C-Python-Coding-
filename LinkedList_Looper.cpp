#include<iostream>
#include<cstdio>
#include <ctime>
#include <sys/time.h>
using namespace std;

struct node{
 int value;
 struct node* next;
 //node* prev; //for a doubly linked list.
 node(){
   next=NULL;
}
};

class Linkedlist_Looper
{
private:
  node *head;
  node *tail;
  node * meet_node;
  int count; //count number of nodes in the list.
  bool loop_exists;
  int loop_where; //nth node from the start.

public:
  Linkedlist_Looper() //default constructor.
  {
     head=NULL;
     tail=NULL;
     meet_node=NULL;
     count=0;
     loop_exists=false;
     loop_where=-1;

  }

  void add_node(int value)
  {
    node* newnode=new node();
    newnode->value=value;

    node* header=head;
    if(header==NULL) //first  node to add to the list.
    {
      head=tail=newnode;
      this->count++;
      //cout<<this->count<<endl;
      return;
    }
    else{
      //add the element to the tail of the list.
      tail->next=newnode;
      tail=newnode;
      this->count++;
      //cout<<this->count<<endl;
    }
}
    //there cant be multiple loops in a list because all nexts are engaged.
    void connect_to_nth(int n)//tail next connects to nth node from start.
    {
      if(n >this->count){
        cout<<"Not enough nodes."<<endl;
        return;
      }

      int start=1;
      node * header=head;
      if(header==NULL)
      {
        printf("Empty list");
        return;
      }
      while(start!=(n-1))
      {
        if(head==tail)
        {
          printf("Not enough nodes");
          return;
        }
        header=header->next;
        start++;
      }
      //we have come to the  n-1th node.
      this->tail->next=header->next;
      this->loop_exists=true;
      this->loop_where=n; //the location of the loop.
      cout<<"Connected"<<n<<endl;
      return;
    }

    node* get_next_stop(node* runner,int hop)
    {
      //if the next location to any hop is null - return NULL
      int start=0;
      node* header=runner;
      while(start!=hop){
        if(header==NULL){
           return header;
        }
        header=header->next;
        start++;
      }
      return header;

    }

    void detect_loop(){
      node* _1x_runner=head;
      node* _2x_runner=head;

      if(head==NULL){
        printf("Empty list");
        return;
      }
      if(head==tail){
        printf("Only on node-no loop");
        return;
      }
      while(_1x_runner!=NULL and _2x_runner!=NULL){
        //if either has reached end - no loop.
        //cout<<"1x_runner: "<<_1x_runner<<endl;
        //cout<<"2x_runner: "<<_2x_runner<<endl;

        _1x_runner=get_next_stop(_1x_runner,1);
        _2x_runner=get_next_stop(_2x_runner,2);
        //cout<<"1x_runner: "<<_1x_runner<<endl;
        //cout<<"2x_runner: "<<_2x_runner<<endl;

        if(_1x_runner==_2x_runner){
          this->meet_node=_1x_runner;
          printf("Loop exists --meeting of two souls\n");
          return;
        }
      }
      cout<<"Loop ends - no loop. one of the runners exited."<<endl;
    }

    int detect_where(){
      //how far is the meet_node from head.
      int n_steps=0;
      if(head==tail or head==NULL or meet_node==NULL)
      {
        cout<<"No loop."<<endl;
        return -1;
      }
      node*finder=head;
      while(finder!=meet_node){
        finder=finder->next;
        n_steps++;
      }
      return n_steps+1;
    }
    //measure the time it takes to run the detect algorithm.
    int timeit(){
        //clock() on linux has a granularity of 1sec - anything less than 1sec goes unrecorded.
        struct timeval tv_start,tv_end;
        clock_t time_a = clock();
        gettimeofday(&tv_start,NULL);
        detect_loop();
        gettimeofday(&tv_end,NULL);
        clock_t time_b =clock();
        cout<<"Time to detect_loop in usec: "<<(unsigned int)(tv_end.tv_usec-tv_start.tv_usec)<<endl;
        cout<<"Time to detect_loop in sec: "<<(unsigned int)(tv_end.tv_sec-tv_start.tv_sec)<<endl;
        cout<<"Time to detect_loop: "<<(unsigned int)(time_b-time_a)<<endl;  //result in time ticks.

       time_a=clock();
       gettimeofday(&tv_start,NULL);
       int where=detect_where();
       cout<<"Meeting point: "<<where<<endl;
       cout<<"Will meet at(N-conn_point+2)"<<endl;
       gettimeofday(&tv_end,NULL);
       time_b =clock();

       cout<<"Time to detect_loop in usec: "<<(unsigned int)(tv_end.tv_usec-tv_start.tv_usec)<<endl;
       cout<<"Time to detect_loop in sec: "<<(unsigned int)(tv_end.tv_sec-tv_start.tv_sec)<<endl;
       cout<<"Time to detect_loop: "<<(unsigned int)(time_b-time_a)<<endl;  //result in time ticks.

    }
  };

int main(){
Linkedlist_Looper lst_looper=Linkedlist_Looper();
for(int i=0;i<10000;i++)
 {
  lst_looper.add_node(i+1);
 }

lst_looper.connect_to_nth(5); //tail to 5th.
lst_looper.timeit();

return 0;
}
