//This program creates a bst both with rules of bst and without
//checks if a given bst is binary search or not.
//1)insert
//2)Search
//3)Get parent
//4)Traversal: inorder,pre-order,post-order.

#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>
#include<vector>
#include<queue>
using namespace std;

struct tree_node{
 int value;
 struct tree_node* left;
 struct tree_node* right;
 //node* prev; //for a doubly linked list.
 tree_node(int value)
   {
    this->value=value;
    left=NULL;
    right=NULL;
   }
 };


class BST{
private:
  tree_node*root;
  int num_nodes;
  int depth;
  bool ordered;

public:
  BST(){
    root=NULL;
    num_nodes=0;
    depth=0;
    ordered=true;
  }

  friend bool is_mirror(BST &bst1,BST &bst2);
  tree_node* get_root(){
    return this->root;
  }
  int num_children(tree_node* node){
    return int(node->left!=NULL)+int(node->right!=NULL);
  }

  void print_tree(){
    //inorder : left->root->right
    if(this->root==NULL)
    {
       cout<<"Empty tree here"<<endl;
       return;
    }
    cout<<"Inorder"<<endl;
    this->inorder(this->root);
    cout<<endl;
  }

  void inorder(tree_node*& root){
        //cout<<"root inorder: "<<root<<endl;
        if(root==NULL){
          return;
        }
        else if(root->right==NULL and root->left==NULL){
                cout<<root->value<<"\t";
                return;
       }
       inorder(root->left);
       cout<<root->value<<"\t";
       inorder(root->right);
  }

 void get_inorder(tree_node* root,vector<int>& lst){
   //recursive - the inorder pattern stored in vector and sent back.
   //left - root - right.
   if(root ==NULL){
     return;
   }
   get_inorder(root->left,lst);
   lst.push_back(root->value);
   get_inorder(root->right,lst);
  }

  void insert(tree_node*&root,int value){
    tree_node* newnode=new tree_node(value);

    if(root==NULL){
      root=newnode;
      cout<<"root insert: "<<root<<"\t"<<root->value<<endl;
      this->num_nodes++;
      return;
    }
    //this here refers to bst.
    else if(value<=root->value){
        this->insert(root->left,value);
        }
    else{
        this->insert(root->right,value);
    }
    return;
  }

  void insert_node(int value){
    //if the root is null, add this as the root.
    //if the value is less, add to left subtree.
    //if the value is greater, add to the right subtree.
       this->insert(this->root,value);
   }

   tree_node* get_leftchild(tree_node*root){
      return root->left;
   }
   tree_node* get_rightchild(tree_node* root){
     return root->right;
   }

};

bool is_mirror(BST* &bst1,BST* &bst2){
//breadth-first method of checking if children at
//each node are in mirror.
   queue<tree_node*> queue1;
   queue<tree_node*> queue2;
   int lvalue1=-1,rvalue1=-1;
   int lvalue2=-1,rvalue2=-1;
   tree_node*head1,*head2;

   queue1.push(bst1->get_root());
   queue2.push(bst2->get_root());
   while (!queue1.empty() and !queue2.empty())
   {
     head1=queue1.front();
     queue1.pop();
     head2=queue2.front();
     queue2.pop();

     tree_node*lc1=bst1->get_leftchild(head1);
     tree_node*lc2=bst2->get_leftchild(head2);

     tree_node*rc1=bst1->get_rightchild(head1);
     tree_node*rc2=bst2->get_rightchild(head2);

     if(lc1!=NULL){
       lvalue1=lc1->value;
       queue1.push(lc1);
      }
     if(lc2!=NULL){
       lvalue2=lc2->value;
       queue2.push(lc2);
      }

      if(rc1!=NULL){
        rvalue1=rc1->value;
        queue1.push(rc1);
       }
      if(rc2!=NULL){
        rvalue2=rc2->value;
        queue2.push(rc2);
       }

      if(lc1!=rc2 or lc2!=rc1){
        return false;
      }
   }

return true;
}

int main()
{
  //manually build two trees that are mirrors - not bst just bt
  //since breaking rules!
BST *bst1=new BST();
BST *bst2=new BST();


  return 0;
};
