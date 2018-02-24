#include<iostream>
using namespace std;


struct tree_node
{
int value;
tree_node* right;
tree_node* left;

//default as well as parametrised.
tree_node(int v=10)
{
value=v;
left=NULL;
right=NULL;
}
};

void preorder(tree_node * root){
if(root == NULL)
return;
preorder(root->left);
cout<<root->value<<"\t";
preorder(root->right);
}

bool is_leaf(tree_node*root){
  if(root->left==NULL && root->right==NULL){
    return true;
  }
  else
  return false;
}

short int remove_nodes(tree_node *& root,int k,int sum){
if(root==NULL)
return 1;
sum=sum+root->value;

//if all are only positive.
/*
if(sum==k){
  root->right=NULL;
  root->left=NULL;
  return 0;
}
else if(sum!=k){
*/

if(root->left){
  if(remove_nodes(root->left,k,sum)){
     root->left=NULL;
  }
}
if(root->right){
  if(remove_nodes(root->right,k,sum)){
    root->right=NULL;
  }
   }

if(is_leaf(root) && sum!=k){
    return 1;
  }

/*
}
*/
  return 0;
}

int main(int argc,char *argv[])
{
tree_node* root=new tree_node(10);
root->left=new tree_node(2);
root->left->left=new tree_node(1);
root->left->right=new tree_node(5);

root->right=new tree_node(18);
root->right->left=new tree_node(15);
root->right->right= new tree_node(20);

preorder(root);
cout<<endl;

int k=13;
int sum=0;
unsigned short int r=remove_nodes(root,k,sum);
if(r){
  root=NULL;
}
preorder(root);
}
