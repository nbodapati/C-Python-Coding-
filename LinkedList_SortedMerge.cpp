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

class Linkedlist_Sorter
{
private:
  node *head,*tail;

public:
  //Make it friend so it can access the contents of both the entities
  //Reference so that need not make copies.
  friend Linkedlist_Sorter* SortedMerge_looping(Linkedlist_Sorter&,Linkedlist_Sorter&);
  Linkedlist_Sorter() //default constructor.
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

  };

Linkedlist_Sorter* SortedMerge_looping(Linkedlist_Sorter& slist1,Linkedlist_Sorter& slist2){
   //Use looping to run through both the lists and decide which node falls where.
    node* head1=slist1.head;
    node*head2=slist2.head;

    if(head1==NULL){
      return &slist2;
    }
    if(head2==NULL){
      return &slist1;
    }

    Linkedlist_Sorter* merged_list=new Linkedlist_Sorter();
    while(head1!=NULL and head2!=NULL){
      if(head1->value <= head2->value){
        cout<<head1->value<<"\t";
        merged_list->add_node(head1->value);
        head1=head1->next;
      }
      else{
        cout<<head2->value<<"\t";
        merged_list->add_node(head2->value);
        head2=head2->next;
      }
    }
    //one or both have become empty and we fill the rest with which ever is left.
    if(head1==NULL) //first one empty
    {
      while(head2!=NULL){
        cout<<head2->value<<"\t";
        merged_list->add_node(head2->value);
        head2=head2->next;
      }
    }
    else if(head2==NULL){
      while(head1!=NULL){
        cout<<head1->value<<"\t";
        merged_list->add_node(head1->value);
        head1=head1->next;
      }
    }
    cout<<endl;
    return merged_list;
}

int main()
{
  Linkedlist_Sorter lst_sorted1=Linkedlist_Sorter();
  Linkedlist_Sorter lst_sorted2=Linkedlist_Sorter();

  for(int i=0;i<=20;i=i+2)
    {
     lst_sorted1.add_node(i+1);
    }
  lst_sorted1.print_contents();

  for(int i=1;i<20;i=i+2)
      {
       lst_sorted2.add_node(i+1);
      }
  lst_sorted2.print_contents();

    struct timeval tv_start,tv_end;
    gettimeofday(&tv_start,NULL);
    Linkedlist_Sorter* merged_list=SortedMerge_looping(lst_sorted1,lst_sorted2);
    gettimeofday(&tv_end,NULL);
    cout<<"Time to swap_nodes in usec: "<<(unsigned int)(tv_end.tv_usec-tv_start.tv_usec)<<endl;
    cout<<"Time to swap_nodes in sec: "<<(unsigned int)(tv_end.tv_sec-tv_start.tv_sec)<<endl;

    merged_list->print_contents();
  return 0;
}
