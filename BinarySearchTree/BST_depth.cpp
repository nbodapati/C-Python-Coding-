//This creates a bst from inorder list of elements.
//Prints the contents inorder.
//computes the depth and diameter.

#include<iostream>
#include<cstdio>
#include<cmath>
#include<cstdlib>
#include<algorithm> //this has the max() function.
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

  tree_node** get_at_root(){
    return &root;
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


};

void inorder_to_bst(tree_node** root,int arr[],int start,int end){
     //start - start of array.
     //end - end of array.
     //stop -condition when start==end
     //cout<<start<<end<<endl;
     if(start==end){
       cout<<"Leaf..."<<endl;
       return;
     }
     int middle=(start+end)/2;

      cout<<"root value: "<<arr[middle]<<endl;
     //this becomes the root->value.
     *root=new tree_node(arr[middle]);
     //cout<<"Root:"<<*root<<endl;
     inorder_to_bst(&((*root)->left),arr, start,middle);
     inorder_to_bst(&((*root)->right),arr,middle+1,end);
}

int calculate_depth(tree_node** root){
    if(*root==NULL){
        return 0;
    }
    return 1+max(calculate_depth(&((*root)->left)),calculate_depth(&((*root)->right)));
}

int sum_tree(tree_node** root){
  if(*root==NULL)
     return 0;
  return (*root)->value + sum_tree(&((*root)->left))+sum_tree(&((*root)->right));
}

void sum_k(tree_node** root,int &count,int sum,int k){
  //count - number of paths from root to leaf with sum==k
  //tracker of sum from root to current node - sum
  //k - the total we are looking for.
  if(*root==NULL){
    return ;
  }
  //at leaf - do the sum check.
  if((*root)->left==NULL and (*root)->right==NULL){
    cout<<"Sum from root to leaf: "<<(*root)->value + sum<<endl;
    if((*root)->value + sum ==k ){
      count++;
      cout<<"Current count: "<<count<<endl;
    }
  }
  //in iteration - pass on cumsum and count values.
  sum=sum+(*root)->value;
  sum_k(&((*root)->left),count,sum,k);
  sum_k(&((*root)->right),count,sum,k);

}

int main()
{
BST *bst=new BST();
cout<<"BST: "<<bst<<endl;

int arr[8];
for(int i=1;i<=sizeof(arr)/sizeof(int);i++){
  arr[i-1]=i;
}

tree_node **root=bst->get_at_root();
inorder_to_bst(root,arr,0,sizeof(arr)/sizeof(int));
bst->print_tree();
cout<<"Tree depth should be: log2(n)"<<calculate_depth(root)-1<<endl;
cout<<"sum of tree: "<<sum_tree(root)<<endl;

int c=0;
sum_k(root,c,0,66);
cout<<"Sum_k: "<<c<<endl;
return 0;
};
