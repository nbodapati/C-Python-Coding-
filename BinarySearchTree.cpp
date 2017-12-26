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

  void print_tree(){
    //inorder : left->root->right
    if(this->root==NULL)
    {
       cout<<"Empty tree here"<<endl;
       return;
    }
    cout<<"Inorder"<<endl;
    this->inorder(this->root);
    cout<<"Pre-order"<<endl;
    this->preorder(this->root);
    cout<<"Postorder"<<endl;
    this->postorder(this->root);

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

  //Root->left->right
  void preorder(tree_node*& root){
        //cout<<"root inorder: "<<root<<endl;
        if(root==NULL){
          return;
        }

       cout<<root->value<<"\t";
       preorder(root->left);
       preorder(root->right);
  }

  //left->right->Root
  void postorder(tree_node*& root){
        //cout<<"root inorder: "<<root<<endl;
        if(root==NULL){
          return;
        }

       postorder(root->left);
       postorder(root->right);
       cout<<root->value<<"\t";
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

  tree_node* search(tree_node*& root,int value){

    if(root==NULL){
      return NULL;
    }
    else if(root->value ==value){
        return root;
    }
    //this here refers to bst.
    else if(value<root->value){
        this->search(root->left,value);
        }
    else{
        this->search(root->right,value);
    }
  }

  tree_node* search_node(int value){
    //if the root is null, add this as the root.
    //if the value is less, add to left subtree.
    //if the value is greater, add to the right subtree.
       tree_node* found=this->search(this->root,value);
       cout<<"Searching for..."<<value<<endl;
       if(found!=NULL)
       cout<<"Found at: "<<found<<endl;
       else
       cout<<"Doesnt exist in the tree"<<endl;

       return found;
    }

    tree_node* parent(tree_node* root,tree_node* current){
         if(current->value<=root->value){
           if(current==root->left){
             return root;
           }
           else{
             parent(root->left,current);
           }
         }
         else{
           if(current==root->right){
             return root;
           }
           else{
             parent(root->right,current);
           }
         }
    }

    tree_node* get_parent(tree_node* current){
      if(current==NULL){
        cout<<"Doesnt exist - not from this tree"<<endl;
        return NULL;
      }

      if(this->root==NULL){
        cout<<"Empty tree - no parent"<<endl;
        return NULL;
      }
      if(root==current){
        cout<<"Root itself"<<endl;
        return NULL;
      }
      return parent(this->root,current);
    }
};

int main()
{
BST *bst=new BST();
cout<<"BST: "<<bst<<endl;

for(int i=10,j=11;i>0,j<20;i=i-2,j=j+2)
  {
    bst->insert_node(i);
    //bst->insert_node(j-i);
    bst->insert_node(j);
  }

  bst->print_tree();

  tree_node*found=bst->search_node(11);
  cout<<"Parent to this: "<<bst->get_parent(found)<<endl;
  found=bst->search_node(5);
  cout<<"Parent to this: "<<bst->get_parent(found)<<endl;
  found=bst->search_node(15);
  cout<<"Parent to this: "<<bst->get_parent(found)<<endl;
  return 0;
};
