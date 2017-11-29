#include<iostream>
#include<cstdio>
#include<stack>
#include<queue>
using namespace std;

struct node{
 int value;
 struct node* next;
 //node* prev; //for a doubly linked list.
 node(){
   next=NULL;
}
};

class list
{
private:
  node *head;
  bool ordered;
  //default no order. if True, ascending.
public:
  list() //default constructor.
  {
     head=NULL;
    ordered=false;
  }

  void makeOrdered(){
    ordered=true;
  }

  node* getNextNode(node* header){
    if(header==NULL)
      return NULL;
    else
      return header->next;
  }

  void isPalindrome() //pointer reference.
  {
    //put lst1 on a queue and lst2 on a stack.
    //not palindrome is lengths not equal.
    //pop each element and compare till the end.
    int num_nodes1=0;
    stack<int> stack_lst;
    queue<int> queue_lst;
    node* head1=this->head;
    while(head1!=NULL){
         stack_lst.push(head1->value);
         queue_lst.push(head1->value);
         num_nodes1++;
         head1=getNextNode(head1);
    }

    while(num_nodes1!=0){
       int el1=stack_lst.top();
       stack_lst.pop();
       int el2=queue_lst.front();
       queue_lst.pop();
       num_nodes1--;
       printf("poped values:%d , %d\n",el1,el2);
       if(el1!=el2)
       {
         printf("Not palindrome\n");
         return;
       }
     printf("Palindrome\n");
  }
}
  void insertAtTail(node* newnode){
    node* header=head;
    while(header->next!=NULL){
      header=header->next;
    }
    header->next=newnode;
    return;
  }

  void insertAtLocation(node* newnode){
       //order -ascending.
       node* header=head;
       //if head is smaller than new node value.
       if(header->value >= newnode->value)
       {
           newnode->next=header;
            head=newnode;
            return;
        }
      else {
        while(header->next!=NULL){
          if(header->next->value >= newnode->value ) {
            newnode->next=header->next;
            header->next=newnode;
            return;
        }
        else {
        //point new one to next node. header to new node.
        header=header->next;
        }
      }
       //check if greater than the last node value
      //add to the tail of the list.
      if(header->value < newnode->value)
        {
          header->next=newnode;
        }
  }
}

  void deleteNode(int value){
       //order -ascending.
       node* header=head;
       //if head is smaller than new node value.
       if(header->value == value)
           head=header->next;
    
      else {
        while(header->next != NULL ){
          if(header->next->value ==value){
             header->next=header->next->next;
             return ; 
            }
          else
             header=header->next; 
        }
      }
  }

  void displayContents(){
    node* header=this->head;
    if(header == NULL)
      return;
    while(header->next!=NULL)
    {
       printf("%d \t",header->value);
       header=header->next;
    }
    printf("%d\n",header->value); //last value.
  }

  void insertNode(int value){
    node* newnode=new node;
    newnode->value=value;

    if(head==NULL){
      head=newnode;
      printf("Inserted at %p\n",head);
    }
    else{
      //loop through the linked list and find an appropriate location to insert.
      //if ordered else insert at the tail in O(n)
      if(!ordered){
        insertAtTail(newnode);
      }
      else{
        insertAtLocation(newnode);
      }
    }
  }
};

int main()
{
 list linkedlist;
 linkedlist.makeOrdered();
 int values_to_add[14]={3,2,1,4,100,98,96,5,1,6,8,9,11,10};
 //Part-1: check if the insertion happens alright.
 for(int val=0;val<sizeof(values_to_add)/sizeof(int);val++)
 {
   int value=values_to_add[val];
   printf("Adding value: %d\n",value);
   linkedlist.insertNode(value);
   linkedlist.displayContents();
 }
  linkedlist.deleteNode(98);
  linkedlist.displayContents();
  linkedlist.isPalindrome();
  /*
  int values_to_add2[8]={6,3,2,1,1,2,3,6};
  list linkedlist2;
  linkedlist2.makeOrdered();

  for(int val=0;val<sizeof(values_to_add2)/sizeof(int);val++)
   {
    int value=values_to_add2[val];
    printf("Adding value: %d\n",value);
    linkedlist2.insertNode(value);
    linkedlist2.displayContents();
   }
    linkedlist2.displayContents();
    linkedlist2.isPalindrome();
  */
  
  int values_to_add2[8]={6,3,2,1,1,2,3,6};
  list linkedlist2;
  //linkedlist2.makeOrdered();

  for(int val=0;val<sizeof(values_to_add2)/sizeof(int);val++)
   {
    int value=values_to_add2[val];
    printf("Adding value: %d\n",value);
    linkedlist2.insertNode(value);
    linkedlist2.displayContents();
   }
    linkedlist2.displayContents();
    linkedlist2.isPalindrome();
    return 0;
}
