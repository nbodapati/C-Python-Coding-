#include<iostream>
#include<cstdio>
#include <ctime>
#include <sys/time.h>
using namespace std;


struct node{
 int value;
 struct node* next;
 node(){
   next=NULL;
}
};

class Linkedlist_Swapper
{
private:
  node *head,*tail;

public:
  Linkedlist_Swapper() //default constructor.
  {
     head=NULL;
     tail=NULL;
  }

  void add_node(int value)
  {
    node* newnode=new node();
    newnode->value=value;

    node* header=head;
    if(header==NULL) //first  node to add to the list.
    {
      head=tail=newnode;
      return;
    }
    else{
      //add the element to the tail of the list.
      tail->next=newnode;
      tail=newnode;
    }
}
    void print_contents(){
      node* header=head;
      while(header!=NULL)
      {
        cout<<header->value<<"\t";
        header=header->next; 
      }
      cout<<endl;
    }

    void swap_nodes(int v1,int v2){
      //Takes the values of two nodes.
      //swaps them using addresses - break and make links
      //instead of copy contents.
      //find hte locations of both - register the parent locations.

      node* parent1=NULL;
      node*parent2=NULL;
      node* header=head;
      int n_found=0;

      if(header==NULL)
      {
        cout<<"Empty list"<<endl;
        return;
      }
      if(header->value==v1 or header->value==v2){
        parent1=header;
        n_found++;
      }
      while(header->next!=NULL){
        if(header->next->value==v1 or header->next->value==v2){
          if(n_found==0){
          parent1=header;
          n_found++;
        }
        else if(n_found==1){
          parent2=header;
          break;
        }
        }
        header=header->next; 
      }
      if(parent1 ==NULL or parent2==NULL){
         cout<<"One or both values not present"<<endl;
         return; 
      }
      if(parent1==head and parent2!=NULL)
       {        
        //Exchange with first node.       
         node* next2=parent2->next->next;
         node* curr1=parent1;
         node* curr2=parent2->next;

         curr2->next=head->next;
         head=curr2;
         parent2->next=curr1;
         curr1->next=next2;
         return;
       }

      cout<<"Parent1: "<<parent1<<"\t"<<parent1->value<<endl;
      cout<<"Parent2: "<<parent2<<"\t"<<parent2->value<<endl;
      node* next1=parent1->next->next;
      node* next2=parent2->next->next;
      node* curr1=parent1->next;
      node* curr2=parent2->next;

      parent1->next=curr2;
      curr2->next=next1;
      parent2->next=curr1;
      curr1->next=next2;
    }

    //measure the time it takes to run the detect algorithm.
    int timeit(){
        //clock() on linux has a granularity of 1sec - anything less than 1sec goes unrecorded.
        struct timeval tv_start,tv_end;
        gettimeofday(&tv_start,NULL);
        this->swap_nodes(10,25);  
        gettimeofday(&tv_end,NULL);
        cout<<"Time to swap_nodes in usec: "<<(unsigned int)(tv_end.tv_usec-tv_start.tv_usec)<<endl;
        cout<<"Time to swap_nodes in sec: "<<(unsigned int)(tv_end.tv_sec-tv_start.tv_sec)<<endl;


    }
  };

int main()
{
  Linkedlist_Swapper lst_swapper=Linkedlist_Swapper();
  for(int i=0;i<100;i++)
    {
     lst_swapper.add_node(i+1);
    }

  lst_swapper.print_contents();
  //now swap nodes.
  lst_swapper.swap_nodes(45,25);

  lst_swapper.print_contents();

  lst_swapper.print_contents();
  //now swap nodes.
  lst_swapper.swap_nodes(1,25);

  lst_swapper.print_contents();

  lst_swapper.print_contents();
  //now swap nodes.
  lst_swapper.swap_nodes(100,25);

  lst_swapper.print_contents();

  lst_swapper.print_contents();
  //now swap nodes.
  lst_swapper.swap_nodes(101,25);

  lst_swapper.print_contents();
  return 0;
}
